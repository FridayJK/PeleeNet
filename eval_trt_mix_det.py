# coding: utf-8
import os
import time
from PIL import Image
import numpy as np
from glob import glob
import cv2
from torch._C import ThroughputBenchmark, dtype
import struct
import pickle
import random
import  json

import torchvision
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import torch 
import torch.nn as nn 
import torch.nn.parallel 
import torch.backends.cudnn as cudnn 
import torch.distributed as dist 
import torch.optim 
import torch.utils.data 
import torch.utils.data.distributed 
import torchvision.transforms as transforms 
import torchvision.datasets as datasets 

from peleenet import PeleeNet
from eval import AverageMeter

# filename = '/mnt/data/JiaoTongDet/2019-08-12-07-55-0114.jpg'
filename = '/mnt/data/JiaoTongDet/video_19_0220.jpg'
max_batch_size = 16
# onnx_model_path = 'peleeDetBatch1_v9_test.onnx'
# onnx_model_path = 'peleeDetBatch1_v9_1.9.0_anchor.onnx'
onnx_model_path = 'peleeDetBatch16_v9_1.9.0_atss.onnx'
onnx_batch = 16
# onnx_model_path = 'resnet50.onnx'

calibDataPath   = "./data/cache_det/"
calibImagePath  = "./data/val_det/"
cacheFile       = calibDataPath + "calib.cache"
iGpu            = 0
calibCount      = 3000
calBatchSize    = 1
inputSize       = (3,544,960)

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine
# IInt8EntropyCalibrator2/IInt8EntropyCalibrator/IInt8MinMaxCalibrator/IInt8LegacyCalibrator
class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibCount, inputShape, dataPath, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)                                              # 基类默认构造函数
        self.calibCount     = calibCount
        self.shape          = inputShape
        self.calibDataSet   = self.loadData(dataPath)                                      # 需要自己实现一个读数据的函数
        self.cacheFile      = cacheFile
        self.calibData      = np.zeros(self.shape, dtype=np.float32)
        self.dIn            = cuda.mem_alloc(trt.volume(self.shape) * trt.float32.itemsize)     # 准备好校正用的设备内存
        self.oneBatch       = self.batchGenerator()

    def batchGenerator(self):                                                                   # calibrator 的核心，一个提供数据的生成器
        for i in range(self.calibCount):
            print("> calibration ", i)
            idx = np.random.choice(self.calibDataSet.shape[0], self.shape[0], replace=False)
            self.calibData = self.calibDataSet[idx,:]  # 随机选取数据
            yield np.ascontiguousarray(self.calibData, dtype=np.float32)                        # 调整数据格式后抛出

    def get_batch_size(self):                                                                   # TensorRT 会调用，不能改函数名
        return self.shape[0]

    def get_batch(self, names):                                                                 # TensorRT 会调用，不能改函数名，老版本 TensorRT 的输入参数个数可能不一样
        try:
            data = next(self.oneBatch)                                                          # 生成下一组校正数据，拷贝到设备并返回设备地址，否则退出
            cuda.memcpy_htod(self.dIn, data)
            return [int(self.dIn)]
        except StopIteration:
            return None

    def read_calibration_cache(self):                                                           # TensorRT 会调用，不能改函数名
        if os.path.exists(self.cacheFile):
            print( "cahce file: %s" %(self.cacheFile) )
            f = open(self.cacheFile, "rb")
            cache = f.read()
            f.close()
            return cache

    def write_calibration_cache(self, cache):                                                   # TensorRT 会调用，不能改函数名
        print( "cahce file: %s" %(self.cacheFile) )
        f = open(self.cacheFile, "wb")
        f.write(cache)
        f.close()

    def loadData(dataPath,imgformate = ".JPEG"):
        with open(calibImagePath + "callib_list_JiaoTong.txt","rt", encoding="utf-8") as f:
            images_list = f.readlines()
        images_list = random.sample(images_list,10)
        data_np = np.zeros([len(images_list), 3, 544, 960],dtype=np.float32)
        for i,filename in enumerate(images_list):
            data_np[i,:] = get_img_np_nchw_det((calibImagePath+filename).strip())
            if((i+1)%100==0):
                print("loaded data:{}".format(i))

        return data_np

def getBatchData(filelists):
    batch = max_batch_size
    out_batch = np.zeros([batch,inputSize[0],inputSize[1],inputSize[2]],dtype=np.float32)
    for i in range(len(filelists)//max_batch_size):
        for j, filename in enumerate(filelists[max_batch_size*i:max_batch_size*(i+1)]):
            out_batch[j,:] = get_img_np_nchw_det(filename).astype(dtype=np.float32)
        yield np.ascontiguousarray(out_batch)
           
def getBatchLabel(filelists):
    batch = max_batch_size
    out_batch = np.zeros([batch],dtype=np.int)
    for i in range(len(filelists)//max_batch_size):
        for j, lab in enumerate(filelists[max_batch_size*i:max_batch_size*(i+1)]):
            out_batch[j] = lab
        yield np.ascontiguousarray(out_batch)

def get_img_np_nchw_det(filename):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # off = 32
    image_cv = cv2.resize(image_cv, (960, 544))
    # image_cv = image_cv[16:240,16:240,:]
    miu = np.array([0.406, 0.456, 0.485])
    std = np.array([0.225, 0.224, 0.229])
    img_np = np.array(image_cv, dtype=np.float32) / 255.
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])
    img_np_nchw = np.expand_dims(img_np_t, axis=0)
    return img_np_nchw

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
# feat:['conv_282','conv_297','conv_312','conv_327']
# exp:['exp_288','exp_303','exp_318','exp_333']
# det_feat_map=['Conv_282','Conv_297','Conv_312','Conv_327','Relu_256','Relu_9']
det_feat_map=['Conv_282','Conv_297','Conv_312','Conv_327']
def setLayerPrecision(network):
    print("Setting layers precision, layers number:{}".format(network.num_layers))
    # for i in range(network.num_layers):
    for i in range(270):
    # for i in [0, 2, 4]:
        layer = network.get_layer(i)
        ltype_ = layer.name.split('_')[0]
        # if(ltype_=='Upsample' or ltype_=='Mul' or ltype_=='Clip' or ltype_=='Exp' or ltype_=='Concat' or ltype_=='Add' or ltype_=='AveragePool'):
        if(ltype_=='Upsample' or ltype_=='Mul' or ltype_=='Clip' or ltype_=='Exp' or ltype_=='Add' or ltype_=='AveragePool'):
            continue
        if(layer.name.split('.')[0]=='base' or (layer.name in det_feat_map)):
            continue
        if(layer.type!=trt.LayerType.CONSTANT and layer.type!=trt.LayerType.CONCATENATION and layer.type!=trt.LayerType.SHAPE \
            and layer.type!=trt.LayerType.GATHER and layer.type!=trt.LayerType.SHUFFLE):
            layer.precision = trt.int8
            # layer.precision = trt.float32
            # layer.precision = trt.float16
            for j in range(layer.num_outputs):
                tensorName = layer.get_output(j).name
                if(layer.get_output(j).is_execution_tensor):
                    layer.set_output_type(j,trt.int8)
                # layer.set_output_type(j,trt.float32)
                # layer.set_output_type(j,trt.float16)
        else:
            layer.precision = trt.float16
            for j in range(layer.num_outputs):
                tensorName = layer.get_output(j).name
                if(layer.get_output(j).is_execution_tensor):
                    layer.set_output_type(j,trt.float16)

    for i in range(270, network.num_layers):
        layer = network.get_layer(i)
        if(layer.get_output(0).dtype == trt.int32): #int32 can't be convert to others type
            continue
        layer.precision = trt.float16
        for j in range(layer.num_outputs):
            tensorName = layer.get_output(j).name
            if(layer.get_output(j).is_execution_tensor):
                layer.set_output_type(j,trt.float16)

def setDynamicRange(network, valMap):
    print("setDynamicRange, layers number:{}".format(network.num_layers))
    #input
    for i in range(network.num_inputs):
        inname = network.get_input(i).name
        input_max = valMap[inname]
        network.get_input(i).dynamic_range = [-input_max,input_max]
        print("input:{}".format(network.get_input(i).dtype))
    #layers
    # for i in range(network.num_layers):
    for i in range(270):
    # for i in [0, 2, 4]:
        layer = network.get_layer(i)
        ltype_ = layer.name.split('_')[0]
        if(ltype_=='Upsample' or ltype_=='Mul' or ltype_=='Clip' or ltype_=='Exp' or ltype_=='Add' or ltype_=='AveragePool' \
            or layer.name.split('.')[0]=='base' or layer.name in det_feat_map):
            continue
        for j in range(layer.num_outputs):
            tname = layer.get_output(j).name
            if(tname in valMap.keys() and layer.get_output(j).is_execution_tensor):
                layer_val = valMap[tname]
                layer.get_output(j).dynamic_range = [-layer_val,layer_val]

def readPerTensorDynamicRangeValues(mPerTensorDynamicRangeMap, filePath):
    print("parse DynamicRangeValues file")
    with open(filePath,"r") as f:
        lines = f.readlines()
        del lines[0]
        for i, line in enumerate(lines):
            key = line.split(":")[0]
            val = line.split(":")[1].strip()
            mPerTensorDynamicRangeMap[key] = struct.unpack('>f', bytes.fromhex(val))[0]*127
            # mPerTensorDynamicRangeMap[key] = struct.unpack('>f', bytes.fromhex(val))[0]

def get_engine(calib, max_batch_size=1, onnx_file_path="", engine_file_path="", \
               fp16_mode=False, int8_mode=False, save_engine=True):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(explicit_batch) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30  # Your workspace size
            builder.max_batch_size = max_batch_size
            builder.fp16_mode = fp16_mode  
            builder.int8_mode = int8_mode  
            builder.int8_calibrator = calib
            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            # network.mark_output(network.get_layer(network.num_layers-1).get_output(0))

            mPerTensorDynamicRangeMap = dict()
            cacheFilePath = "./data/cache_det/calib.cache"
            readPerTensorDynamicRangeValues(mPerTensorDynamicRangeMap, cacheFilePath)

            count =0
            matched = 0
            for i in range (network.num_layers):
                layer = network.get_layer(i)
                print('layer:{} name:{}'.format(i, layer.name))
                for j in range(layer.num_outputs):
                    count = count+1
                    print('output:{}, dtype:{}'.format(layer.get_output(j).name, layer.get_output(j).dtype, layer.get_output(j).is_execution_tensor))
                    
            print("count:{} matched:{}".format(count, matched))

            if(int8_mode):
                setLayerPrecision(network)            
                setDynamicRange(network, mPerTensorDynamicRangeMap)
                
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def non_max_suppression(prediction, xyxy=False, conf_thres=0.5, nms_thres=0.4, thresh_type='score'):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    
    Args:
        prediction: tensor with shape[batch,image_num,N],N=bbox+conf(centerness)+cls_num
        xyxy: bbox type
        conf_thres: blackground thresh(centerness_thresh)
        nms_thres: nms thresh
        thresh_type: "conf" or 'score'

    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    if xyxy == False:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        if thresh_type == 'conf':
            image_pred = image_pred[image_pred[:, 4] >= conf_thres]
            if image_pred.shape[0] <= 0:
                continue
            # Object confidence times class confidence
            score = (image_pred[:, 4].unsqueeze(1) * image_pred[:, 5:]).max(1)[0]
            # Sort by it
            image_pred = image_pred[(-score).argsort()]
            class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
            detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        elif thresh_type == 'score':
            # get the box cls is > 0.1
            per_box_score = image_pred[:, 4:5] * image_pred[:, 5:]
            if isinstance(conf_thres, list):
                per_candidate_inds = per_box_score > torch.as_tensor(conf_thres).cuda()
            elif isinstance(conf_thres, float):
                per_candidate_inds = per_box_score > conf_thres
            # multiply the classification scores with centerness scores
            per_box_score = per_box_score[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_box_class = per_candidate_nonzeros[:, 1]
            per_box_regression = image_pred[per_box_loc, :4]
            image_pred = torch.cat(
                [per_box_regression, per_box_score.unsqueeze(-1), per_box_class.unsqueeze(-1).float()], dim=-1)
            # Sort by it
            _, indices = per_box_score.sort(descending=True)
            detections = image_pred[indices]
        else:
            raise ValueError
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output

def write_json():

    return

def postprocess_the_outputs(outputs):
    # with open("fcos_anchors_locations.pkl","rb") as f:
    #     anchors_locations = pickle.load(f)
    # anchors_locations = [torch.from_numpy(x) for x in anchors_locations]
    outputs = torch.from_numpy(outputs)
    # idx1=0
    # for _, (locations, idx2) in enumerate(zip(anchors_locations, [8160,2040,510,135])):
    #     idx2 = idx2 + idx1
    #     pos_anchors_x = (locations[..., 0] + locations[..., 2])/2
    #     pos_anchors_y = (locations[..., 1] + locations[..., 3])/2
    #     outputs[:,idx1:idx2,0] = (outputs[:,idx1:idx2,0] + pos_anchors_x).clamp(min=0)
    #     outputs[:,idx1:idx2,1] = (outputs[:,idx1:idx2,1] + pos_anchors_y).clamp(min=0)
    #     idx1 = idx2
    outputs2 = non_max_suppression(outputs, xyxy=False,
                                        conf_thres=0.22,
                                        nms_thres=0.5,
                                        thresh_type='score')
    if(outputs2[0]==None):
        return None
    outputs2 = [outp.numpy() for outp in outputs2]
    return outputs2

#data loader config
test_data_path = "/mnt/data/ILSVRC2012/ILSVRC2012_val/"

def main():
    #---------------------------------------
    # calib = MyCalibrator(calibCount, (calBatchSize,) + inputSize, calibDataPath, cacheFile)
    calib = None

    # These two modes are dependent on hardwares
    fp16_mode = True
    int8_mode = True
    # fp16_mode = True
    # int8_mode = False
    model_type = "mix"
    trt_engine_path = './model_fp16_{}_int8_{}_maxbatch{}_{}_{}.trt'.format(fp16_mode, int8_mode, max_batch_size,onnx_model_path,model_type)
    # Build an engine
    engine = get_engine(calib, max_batch_size, onnx_model_path, trt_engine_path, fp16_mode, int8_mode)
    context = engine.create_execution_context()
    # Allocate buffers for input and output
    inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

    shape_of_output = (max_batch_size, 10845, 9)

    with open("./test_list_JiaoTong_v2.txt", encoding="utf-8") as f:
        img_list = f.readlines()

    infer_time = 0
    detect_miss = 0
    img_root_path = "/mnt/data/JiaoTongDet2/"
    labels = {0: "person", 1: "non-motor", 2: "car", 3: "tricycle", 4: "motorcycle"}
    json_writer = open("det_res_"+model_type+"_batch"+str(max_batch_size)+ "_v3_newCab" +".json", "w")
    for i, filename in enumerate(img_list):
        # if(i>=1000):
        #     continue
        filename = img_root_path + filename.strip()
        img_np_nchw = get_img_np_nchw_det(filename)
        img_np_nchw = img_np_nchw.astype(dtype=np.float32)
        # # Load data to the buffer
        inputs[0].host = img_np_nchw.reshape(-1)
        
        t1 = time.time()
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
        t2 = time.time()
        infer_time = infer_time + t2-t1
        outputs_res = trt_outputs[0].reshape(*shape_of_output)
        # print(i,filename)
        outputs2 = postprocess_the_outputs(outputs_res[0].reshape((1,10845,9)))

        dict_res = {}
        dict_res["url"] = filename.strip()
        dict_res["type"] = "image"
        label_dict = {}
        
        if(outputs2 == None):
            print("{}:no object detected".format(filename.encode("utf-8")))
            detect_miss = detect_miss+1
            label_dict["data"] = []
        else:
            for j, boxes in enumerate(outputs2):#for batch
                label_dict["name"] = "general"
                label_dict["type"] = "detection"
                label_dict["version"] = "1"
                data_list = []
                if boxes is None:
                    continue
                for box in boxes:
                    data_dict = {}
                    xmin = int(box[0] * 1)*2
                    ymin = int(box[1] * 1)*2
                    xmax = int(box[2] * 1)*2
                    ymax = int(box[3] * 1)*2
                    data_dict["bbox"] = [[xmin, ymin],[xmax, ymin],[xmax, ymax],[xmin, ymax]]
                    data_dict["class"] = [labels[int(box[-1])]]
                    data_dict["scores"] = [float(box[4])]
                    data_list.append(data_dict)
                label_dict["data"] = data_list
        
        dict_res["label"] = [label_dict]
        json_str = json.dumps(dict_res)
        json_writer.write(json_str+"\n")

        if((i+1)%100==0):
            print("processed:{}".format(i))

    json_writer.close()
    print("ave times:{}".format(infer_time/len(img_list)))
    # print("ave times:{}".format(infer_time/1000))
    print("{} images no obj detected".format(detect_miss))
    print('TensorRT ok')

    # top1 = AverageMeter()
    # top5 = AverageMeter()
    # batch_time = AverageMeter()


if __name__ == '__main__':
    main()