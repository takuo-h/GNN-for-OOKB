import numpy
from chainer import Variable, cuda

class XP:
    def __init__(self,args):
        self.is_gpu = args.use_gpu
        if args.use_gpu:
            self.lib = cuda.cupy
            cuda.get_device(args.gpu_device).use()
        else:
            self.lib = numpy

    def array_int(self,xs,is_train=True):
        if is_train:    return Variable(self.lib.array(xs, dtype=self.lib.int32))
        else:           return Variable(self.lib.array(xs, dtype=self.lib.int32),volatile='on')

    def array_float(self,xs,is_train=True):
        if is_train:    return Variable(self.lib.array(xs, dtype=self.lib.float32))
        else:           return Variable(self.lib.array(xs, dtype=self.lib.float32),volatile='on')
    
    def get_max(self,x):
        x = x.data.argmax(1)
        if self.is_gpu: x = cuda.to_cpu(x)
        return list(x)
