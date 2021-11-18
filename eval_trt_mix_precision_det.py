# coding: utf-8
import os
import time
from PIL import Image
import numpy as np
from glob import glob
import cv2
import struct
import pickle
import random

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

filename = '/mnt/data/JiaoTongDet/2019-08-12-07-55-0114.jpg'
max_batch_size = 16
profile_shape = (1,8,16)   #(min, opt, max)
onnx_model_path = 'peleeDet_atss_v11_20211118_544x960_dynamic.onnx'   #960*544
# onnx_model_path = 'peleeDet_atss_v11_20211118_544x960_nodynamic.onnx'   #960*544

# img_size = [480, 864]
# out_len = 8617
img_size = [544, 960]
out_len = 10845

calibDataPath   = "./data/cache_det/"
calibImagePath  = "./data/val_det/"
cacheFile       = calibDataPath + "calib.cache"
iGpu            = 0
calibCount      = 4000
calBatchSize    = 1
inputSize       = (3,img_size[0],img_size[1])

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
        data_np = np.zeros([len(images_list), 3, img_size[0], img_size[1]],dtype=np.float32)
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
    image_cv = cv2.resize(image_cv, (img_size[1], img_size[0]))
    miu = np.array([0.406, 0.456, 0.485])
    std = np.array([0.225, 0.224, 0.229])
    # img_np = np.array(image_cv, dtype=np.float32) / 255.
    img_np = np.array(image_cv, dtype=np.float32) / 1.
    # r = (img_np[:, :, 0] - miu[0]) / std[0]
    # g = (img_np[:, :, 1] - miu[1]) / std[1]
    # b = (img_np[:, :, 2] - miu[2]) / std[2]
    r = img_np[:, :, 0]
    g = img_np[:, :, 1]
    b = img_np[:, :, 2]
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
        size = trt.volume(engine.get_binding_shape(binding)[1:]) * engine.max_batch_size
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
LAYERS_TH = 287 #262 #282 #285 #287:31.2ms |#288:30.3ms #289:30.4 #290:29.6ms
det_feat_map=['Conv_282','Conv_297','Conv_312','Conv_327']
def setLayerPrecision(network):
    print("Setting layers precision, layers number:{}".format(network.num_layers))
    # for i in range(network.num_layers):
    for i in range(LAYERS_TH):
        layer = network.get_layer(i)
        ltype_ = layer.name.split('_')[0]
        # if(ltype_=='Upsample' or ltype_=='Mul' or ltype_=='Clip' or ltype_=='Exp' or ltype_=='Concat' or ltype_=='Add' or ltype_=='AveragePool'):
        if(ltype_=='Resize' or ltype_=='Upsample' or ltype_=='Clip' or ltype_=='Exp'):
            continue
        if(layer.name.split('.')[0]=='base' or (layer.name in det_feat_map) or layer.get_output(0).dtype == trt.int32):
            continue
        if(layer.type!=trt.LayerType.CONCATENATION and layer.type!=trt.LayerType.SHAPE \
            and layer.type!=trt.LayerType.GATHER and layer.type!=trt.LayerType.SHUFFLE):  #layer.type!=trt.LayerType.CONSTANT and
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

    for i in range(LAYERS_TH, network.num_layers):
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
    # for i in range(network.num_layers): # ltype_=='AveragePool' # ltype_=='Add'
    for i in range(LAYERS_TH):
        layer = network.get_layer(i)
        ltype_ = layer.name.split('_')[0]
        if(ltype_=='Resize' or ltype_=='Upsample' or ltype_=='Clip' or ltype_=='Exp' \
            or layer.name.split('.')[0]=='base' or layer.name in det_feat_map or layer.get_output(0).dtype == trt.int32):
            continue
        for j in range(layer.num_outputs):
            tname = layer.get_output(j).name
            if(tname in valMap.keys() and layer.get_output(j).is_execution_tensor):
                print("out_blob:{}".format(tname))
                layer_val = valMap[tname]
                layer.get_output(j).dynamic_range = [-layer_val,layer_val]
            else:
                print("layer output:{} not set!@@@".format(tname))

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

            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            # network.mark_output(network.get_layer(network.num_layers-1).get_output(0))

            count =0
            matched = 0
            for i in range (network.num_layers):
                layer = network.get_layer(i)
                print('layer:{} name:{}'.format(i, layer.name))
                for j in range(layer.num_outputs):
                    count = count+1
                    print('output:{}, dtype:{}'.format(layer.get_output(j).name, layer.get_output(j).dtype, layer.get_output(j).is_execution_tensor))    
            print("count:{} matched:{}".format(count, matched))

            if(int8_mode and calib == None):
                mPerTensorDynamicRangeMap = dict()
                cacheFilePath = cacheFile
                readPerTensorDynamicRangeValues(mPerTensorDynamicRangeMap, cacheFilePath)
                setLayerPrecision(network)            
                setDynamicRange(network, mPerTensorDynamicRangeMap)

            config = builder.create_builder_config()
            profile = builder.create_optimization_profile()
            profile.set_shape(network.get_input(0).name, (profile_shape[0], 3, img_size[0], img_size[1]), (profile_shape[1], 3, img_size[0], img_size[1]), (profile_shape[2], 3, img_size[0], img_size[1]))
            config.add_optimization_profile(profile)
            if(fp16_mode):
                config.set_flag(trt.BuilderFlag.FP16)
            if(int8_mode):
                config.set_flag(trt.BuilderFlag.INT8)            
            config.int8_calibrator = calib

            #stage1
            # engine = builder.build_cuda_engine(network)
            #stage2
            engine = builder.build_engine(network,config = config)
            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
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
    # y = x.new(x.shape)
    y = np.zeros_like(x)
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
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, 10000) * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, 10000)
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
            detections = np.concatenate((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        elif thresh_type == 'score':
            # get the box cls is > 0.1
            per_box_score = image_pred[:, 4:5] * image_pred[:, 5:]
            if isinstance(conf_thres, list):
                per_candidate_inds = per_box_score > np.array(conf_thres)
            elif isinstance(conf_thres, float):
                per_candidate_inds = per_box_score > conf_thres
            # multiply the classification scores with centerness scores
            per_box_score = per_box_score[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_candidate_nonzeros = np.vstack((per_candidate_nonzeros[0],per_candidate_nonzeros[1])).T
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_box_class = per_candidate_nonzeros[:, 1]
            per_box_regression = image_pred[per_box_loc, :4]
            image_pred = np.concatenate((per_box_regression, per_box_score.reshape(per_box_score.shape[0],1), per_box_class.reshape(per_box_class.shape[0],1)), axis=1)

            # Sort by it
            indices = per_box_score.argsort()[::-1]
            # _, indices = per_box_score.sort(descending=True)
            detections = image_pred[indices]
        else:
            raise ValueError
        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue

        # Perform non-maximum suppression
        keep_boxes = []
        while detections.shape[0]:
            large_overlap = bbox_iou(detections[0, :4].reshape(1,4), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = np.vstack(keep_boxes)
    return output

def postprocess_the_outputs(outputs):
    outputs2 = non_max_suppression(outputs, xyxy=False,
                                        conf_thres=[0.2754, 0.2283, 0.3089, 0.3933],
                                        nms_thres=0.5,
                                        thresh_type='score')
    outputs2 = [outp for outp in outputs2]
    return outputs2

#data loader config
test_data_path = "/mnt/data/ILSVRC2012/ILSVRC2012_val/"

def main():
    #---------------------------------------
    # calib = MyCalibrator(calibCount, (calBatchSize,) + inputSize, calibDataPath, cacheFile)
    calib = None

    fp16_mode = True
    int8_mode = True

    model_type = "int8" #"f32" "mix" "int8" "f16"
    version_ = "v0." + str(LAYERS_TH)
    trt_engine_path = './model_mb{}_{}_{}_{}_{}_{}_{}.trt'.format(max_batch_size,onnx_model_path,model_type,profile_shape[0],profile_shape[1],profile_shape[2],version_)

    # Build an engine
    engine = get_engine(calib, max_batch_size, onnx_model_path, trt_engine_path, fp16_mode, int8_mode)
    context = engine.create_execution_context()
    exex_batch = 8
    context.set_binding_shape(0, (exex_batch, 3, img_size[0], img_size[1]))
    # Allocate buffers for input and output
    inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

    img_np_nchw = get_img_np_nchw_det(filename)
    img_np_nchw = img_np_nchw.astype(dtype=np.float32)

    shape_of_output = (max_batch_size, out_len, 9)

    # # Do inference
    # # Load data to the buffer
    img_np_nchw = np.ascontiguousarray(np.repeat(img_np_nchw,exex_batch,axis=0))
    inputs[0].host = img_np_nchw.reshape(-1)

    # inputs[1].host = ... for multiple input
    times = 1000
    t1 = time.time()
    for i in range(times):
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
    t2 = time.time()
    print("average time:{}".format((t2-t1)/times))
    outputs = trt_outputs[0].reshape(*shape_of_output)
    outputs2 = postprocess_the_outputs(outputs[0].reshape((1,out_len,9)))

    #show
    COLOR = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
    for j, boxes in enumerate(outputs2):
        cv_img = img_np_nchw[j].transpose(1, 2, 0)
        cv_img = cv_img.astype(np.uint8)[:, :, [2, 1, 0]]
        cv_img = cv2.cvtColor(np.asarray(cv_img), cv2.COLOR_RGB2BGR)
        if boxes is None:
            continue
        for box in boxes:
            xmin = int(box[0] * 1)
            ymin = int(box[1] * 1)
            xmax = int(box[2] * 1)
            ymax = int(box[3] * 1)
            cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), COLOR[int(box[-1])], 1)
            cv2.putText(cv_img, str(int(box[-1])), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        COLOR[int(box[-1])], 1)

        # cv2.imshow("show", cv_img)
        cv2.imwrite("./output/2019-08-12-07-55-0114_res.jpg",cv_img)
        if cv2.waitKey() == ord("c"):
            continue
        elif cv2.waitKey() == ord("q"):
            exit()

    print('TensorRT ok')


if __name__ == '__main__':
    main()