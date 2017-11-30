import numpy, random
from chainer import Variable

class Backend:
	def __init__(self,args):
		self.is_gpu = args.use_gpu
		if args.use_gpu:
			from chainer import cuda
			import cupy
			cuda.get_device(args.gpu_device).use()
			self.lib = cuda.cupy
		else:
			self.lib = numpy
		
		if args.seed!=-1:
			random.seed(args.seed)
			numpy.random.seed(args.seed)
			if args.use_gpu: cupy.random.seed(args.seed)

	def array_int(self,xs):
		return Variable(self.lib.array(xs, dtype=self.lib.int32))

	def array_float(self,xs):
		return Variable(self.lib.array(xs, dtype=self.lib.float32))

	def get_max(self,x):
		x = x.data.argmax(1)
		if self.is_gpu: x = cuda.to_cpu(x)
		return list(x)
