import os
import time
from PIL import Image
import numpy as np
from glob import glob
import cv2
from torch._C import dtype

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
# /mnt/data/ILSVRC2012/ILSVRC2012_val/val/n01440764/ILSVRC2012_val_00000293.JPEG 0
# /mnt/data/ILSVRC2012/ILSVRC2012_val/val/n01531178/ILSVRC2012_val_00004243.JPEG 11
# /mnt/data/ILSVRC2012/ILSVRC2012_val/val/n02097474/ILSVRC2012_val_00000802.JPEG 200
# /mnt/data/ILSVRC2012/ILSVRC2012_val/val/n02669723/ILSVRC2012_val_00000851.JPEG 400
# /mnt/data/ILSVRC2012/ILSVRC2012_val/val/n03532672/ILSVRC2012_val_00000575.JPEG 600
# /mnt/data/ILSVRC2012/ILSVRC2012_val/val/n04243546/ILSVRC2012_val_00000857.JPEG 800

filename = '/mnt/data/ILSVRC2012/ILSVRC2012_val/val/n02669723/ILSVRC2012_val_00000851.JPEG'
max_batch_size = 1
onnx_model_path = 'peleeNetBatch1.onnx'
# onnx_model_path = 'resnet50.onnx'


calibDataPath   = "./data/cache/"
cacheFile       = calibDataPath + "calib.cache"
iGpu            = 0
calibCount      = 10
batchSize       = max_batch_size
inputSize       = (3,224,224)

TRT_LOGGER = trt.Logger()  # This logger is required to build an engine

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
        images_list = glob("./data/val/" + "*.JPEG")
        data_np = np.zeros([len(images_list), 3, 224, 224])
        for i,filename in enumerate(images_list):
            data_np[i,:] = get_img_np_nchw(filename)

        return data_np

def getBatchData(filelists):
    batch = max_batch_size
    out_batch = np.zeros([batch,inputSize[0],inputSize[1],inputSize[2]],dtype=np.float32)
    for i in range(len(filelists)//max_batch_size):
        for j, filename in enumerate(filelists[max_batch_size*i:max_batch_size*(i+1)]):
            out_batch[j,:] = get_img_np_nchw(filename).astype(dtype=np.float32)
        yield np.ascontiguousarray(out_batch)
           
def getBatchLabel(filelists):
    batch = max_batch_size
    out_batch = np.zeros([batch],dtype=np.int)
    for i in range(len(filelists)//max_batch_size):
        for j, lab in enumerate(filelists[max_batch_size*i:max_batch_size*(i+1)]):
            out_batch[j] = lab
        yield np.ascontiguousarray(out_batch)

def get_img_np_nchw(filename):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    off = 32
    image_cv = cv2.resize(image_cv, (224+off, 224+off))
    image_cv = image_cv[16:240,16:240,:]
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(image_cv, dtype=float) / 255.
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
            # pdb.set_trace()
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
            network.mark_output(network.get_layer(network.num_layers-1).get_output(0))
            last_layer = network.get_layer(network.num_layers - 1)
            network.mark_output(last_layer.get_output(0))

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

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

#data loader config
test_data_path = "/mnt/data/ILSVRC2012/ILSVRC2012_val/"
input_dim = 224
workers = 1

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    valdir = os.path.join(test_data_path, 'val')
    # valdir = args.data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(input_dim+32),
            transforms.CenterCrop(input_dim),
            transforms.ToTensor(),
            normalize,
        ]))

    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #     batch_size=max_batch_size, shuffle=False,
    #     num_workers=workers, pin_memory=True)

    num_classes = len(val_dataset.classes)
    print('Total classes: ',num_classes)

    #---------------------------------------
    calib = MyCalibrator(calibCount, (batchSize,) + inputSize, calibDataPath, cacheFile)

    # These two modes are dependent on hardwares
    fp16_mode = False
    int8_mode = True
    trt_engine_path = './model_fp16_{}_int8_{}_maxbatch{}_{}.trt'.format(fp16_mode, int8_mode, max_batch_size,onnx_model_path)
    # Build an engine
    engine = get_engine(calib, max_batch_size, onnx_model_path, trt_engine_path, fp16_mode, int8_mode)
    context = engine.create_execution_context()
    # Allocate buffers for input and output
    inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

    img_np_nchw = get_img_np_nchw(filename)
    img_np_nchw = img_np_nchw.astype(dtype=np.float32)

    shape_of_output = (max_batch_size*1, 1000)
    # shape_of_output = (max_batch_size, 1000)

    # # Do inference
    # # Load data to the buffer
    inputs[0].host = img_np_nchw.reshape(-1)

    # inputs[1].host = ... for multiple input
    t1 = time.time()
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
    t2 = time.time()
    feat = trt_outputs[0].reshape(*shape_of_output)
    # feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)
    print(feat.argmax())
    print('TensorRT ok')

    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    

    # use dataLoader gen input
    # for i, (input, target) in enumerate(val_loader):
    imglists = [filename[0] for filename in val_dataset.imgs]
    imglabels = [filename[1] for filename in val_dataset.imgs]
    batch_data_gen = getBatchData(imglists)
    batch_lab_gen = getBatchLabel(imglabels)
    i=0
    for batch_data in batch_data_gen:
        batch_lab = next(batch_lab_gen)
        inputs[0].host = batch_data.reshape(-1)
        end = time.time()
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream,batch_size=batchSize)
        feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)

        tensor_output = torch.from_numpy(feat[0:batchSize,:])
        # np.savetxt('b32_res.txt',feat[0:32,:])
        target = torch.from_numpy(batch_lab)

        prec1, prec5 = accuracy(tensor_output.data, target, topk=(1, 5))
        top1.update(prec1[0], (len(imglists)//max_batch_size)*max_batch_size)
        top5.update(prec5[0], (len(imglists)//max_batch_size)*max_batch_size)
        batch_time.update(time.time() - end)
        end = time.time()


        i = i+1
        if i % 10 == 0:
            print('Test: [{0}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, batch_time=batch_time, 
                #    loss=losses,
                   top1=top1, top5=top5))
    
    print("Inference time with the TensorRT engine: {}".format(t2-t1))
    print('All completed!')


if __name__ == '__main__':
    main()