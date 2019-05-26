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

	def array(self,xs,dtype):
		if dtype in ['float','f']:	return Variable(self.lib.array(xs, dtype=self.lib.float32))
		if dtype in ['int','i']:	return Variable(self.lib.array(xs, dtype=self.lib.int32))
