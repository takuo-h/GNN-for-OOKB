import numpy
from chainer import Variable
#from chainer import cuda # in default, cupy (gpu) is tuned off. please remove the first # to use gpu

class Backend:
	def __init__(self,args):
		self.is_gpu = args.use_gpu
		if args.use_gpu:
			self.lib = cuda.cupy
			cuda.get_device(args.gpu_device).use()
		else:
			self.lib = numpy

	def array_int(self,values):
		return Variable(self.lib.array(values, dtype=self.lib.int32))

	def array_float(self,values):
		return Variable(self.lib.array(values, dtype=self.lib.float32))
	
	def get_max(self,values):
		values = values.data.argmax(1)
		if self.is_gpu: values = cuda.to_cpu(values)
		return list(values)
