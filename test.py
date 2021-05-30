import os
import sys
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from datetime import datetime as dt

import loadPara as ld
import calibrator

DEBUG           = True
testDataPath    = "./"
calibDataPath   = "./"
tempPath        = "./"
paraFile        = tempPath + "para.h5"
cacheFile       = tempPath + "calib.cache"
outputFile      = tempPath + "output.txt"

iGpu            = 0
calibCount      = 10                        # int8 校正次数
inputSize       = (1,1,1)                   # 输入数据尺寸，CHW

class TrtPredictor:
    def __init__(self, batchSize, dataType):
        self.logger     = trt.Logger(trt.Logger.ERROR)                      # 创建 logger
        self.batchSize  = batchSize
        self.dataType   = dataType
        self.h5f, ...   = fld.loadPara(paraFile)                            # 读取训练好的参数

        trtFilePath = tempPath + "engine-" + self.dataType + ".trt"         # 尝试读取创建好的引擎，没有则现场创建引擎
        if os.path.isfile(trtFilePath) and not DEBUG:
            f =  open(trtFilePath, 'rb')
            engineStr = f.read()                                            # enginStr 不作为成员变量
            self.runtime = trt.Runtime(self.logger)                         # 运行时读取文件中的引擎
            self.engine = self.runtime.deserialize_cuda_engine(engineStr)
            f.close()
            print("succeeded loading engine!")
        else:
            self.create_engine()                                            # 创建 engine，并写入文件，方便下次调用
            if self.engine == None:
                print("failed building engine!")
                return
            engineStr = self.engine.serialize()
            f = open(trtFilePath, 'wb')
            f.write(engineStr)
            f.close()
            print("succeeded building engine!")

        self.context = self.engine.create_execution_context()               # 创建 CUDA 上下文和流
        self.stream = cuda.Stream()

    def __del__(self):
        self.context = None
        self.engine  = None
        ld.close(self.h5f)

    def create_engine(self):                                                # 构造引擎
        self.builder = trt.Builder(self.logger)
        self.builder.max_batch_size     = 16
        self.builder.max_workspace_size = 1 << 30
        self.builder.fp16_mode          = self.dataType == 'float16'
        self.builder.int8_mode          = self.dataType == 'int8'
        self.network                    = self.builder.create_network()
        self.builder.strict_type_constraints = True

        h0 = self.network.add_input("h0", trt.DataType.FLOAT, (1,) + inputSize) # 强制 N 为 1，多的数据堆在更高维度上

        #...                                                                # 中间层

        self.network.mark_output(h0.get_output(0))                          # 标记输出层

        if self.dataType == 'int8':                                         # int8 需要额外的校正，放到 builder 中
            self.builder.int8_calibrator = calibrator.MyCalibrator(calibCount, (self.batchSize,) + inputSize, calibDataPath, cacheFile)

        self.engine = self.builder.build_cuda_engine(self.network)          # 创建引擎（最容易失败的地方，返回构造函数后要检查是否成功）

    def infer(self, hInPart, dIn, dOut, hOut):                              # 推理
        cuda.memcpy_htod_async(dIn, hInPart, self.stream)
        self.context.execute_async(len(hInPart), [int(dIn), int(dOut)], self.stream.handle)
        cuda.memcpy_dtoh_async(hOut, dOut, self.stream)
        self.stream.synchronize()

def predict(hIn, batchSize, dataType):
    predictor = TrtPredictor(batchSize, dataType)                           # 构造一个预测器

    dIn  = cuda.mem_alloc(hIn[0].nbytes * batchSize)                        # 准备主机和设备内存
    hOut = np.empty((batchSize,) + tuple(predictor.engine.get_binding_shape(1)), dtype = np.float32)
    dOut = cuda.mem_alloc(hOut.nbytes)                                      # dOut 和 hOut 的大小一定是相同的
    res=[]
    for i in range(0, len(hIn), batchSize):                                 # 分 batch 喂入数据
        predictor.infer(hIn[i:i+batchSize], dIn, dOut, hOut)
        res.append( hOut )

    return res

if __name__ == "__main__":                                                  # main 函数负责管理 cuda.Device 和 cuda.Context
    _ = os.system("clear")
    batchSize = int(sys.argv[1])    if len(sys.argv) > 1 and sys.argv[1].isdigit()                         else 1
    dataType  = sys.argv[2]         if len(sys.argv) > 2 and sys.argv[2] in ['float32', 'float16', 'int8'] else 'float32'
    DEBUG     = int(sys.argv[3])>0  if len(sys.argv) > 3 and sys.argv[3].isdigit()                         else False
    if DEBUG:                                                               # 清除建好的 engine 和 校正缓存，重头开始建立
        oldEngineEAndCache = glob(tempPath+"*.trt") + glob(tempPath+"*.cache")
        [ os.remove(oldEngineEAndCache[i]) for i in range(len(oldEngineEAndCache))]
    print( "%s, start! GPU =  %s, batchSize = %2d, dataType  = %s" %( dt.now(), cuda.Device(iGpu).name(), batchSize, dataType ) )

    inputData = loadData(testDataPath)                                      # 读取数据
    oF = open(outputFile, 'w')
    cuda.Device(iGpu).make_context()

    res = predict(inputData, batchSize, dataType)
    for i in range(len(res)):
        print( "%d -> %s" % (i,res[i]) )
        oF.write(res[i] + '\n')

    oF.close()
    cuda.Context.pop()
    print( "%s, finish!" %(dt.now()) )