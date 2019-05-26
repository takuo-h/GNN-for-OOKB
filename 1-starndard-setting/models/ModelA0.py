#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L

from collections import defaultdict
import sys,random

class Module(chainer.Chain):
	def __init__(self, dim, dropout_rate, activate, isR, isBN):
		super(Module, self).__init__()
		with self.init_scope():
			self.x2z	= L.Linear(dim,dim),
			self.bn		= L.BatchNormalization(dim)
	def __call__(self, x):
		if self.dropout_rate!=0:
			x = F.dropout(x,ratio=self.dropout_rate)
		z = self.x2z(x)
		if self.is_batchnorm:
			z = self.bn(z)
		if self.activate=='tanh': z = F.tanh(z)
		if self.activate=='relu': z = F.relu(z)
		if self.is_residual:	return z+x
		else: return z
class Block(chainer.Chain):
	def __init__(self, dim, dropout_rate, activate, layer, isR, isBN):
		super(Block, self).__init__()
		links = [('m{}'.format(i), Module(dim,dropout_rate,activate, isR, isBN)) for i in range(layer)]
		for link in links:
			self.add_link(*link)
		self.forward = links
	def __call__(self,x):
		for name, _ in self.forward:
			x = getattr(self,name)(x)
		return x

class Tunnel(chainer.Chain):
	def __init__(self, dim, dropout_rate, activate, layer, isR, isBN, relation_size, pooling_method):
		super(Tunnel, self).__init__()
		linksH = [('h{}'.format(i), Block(dim,dropout_rate,activate,layer,isR,isBN)) for i in range(relation_size)]
		for link in linksH:
			self.add_link(*link)
		self.forwardH = linksH
		linksT = [('t{}'.format(i), Block(dim,dropout_rate,activate,layer,isR,isBN)) for i in range(relation_size)]
		for link in linksT:
			self.add_link(*link)
		self.forwardT = linksT
		self.pooling_method = pooling_method
		self.layer = layer

	def maxpooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = []
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:
				x = F.concat(xxs,axis=0)					# -> (b,d)
				x = F.swapaxes(x,0,1)						# -> (d,b)
				x = F.maxout(x,len(xxs))					# -> (d,1)
				x = F.swapaxes(x,0,1)						# -> (1,d)
				result.append(x)
		return result

	def averagepooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = []
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:			result.append(sum(xxs)/len(xxs))
		return result

	def sumpooling(self,xs,neighbor):
		sources = defaultdict(list)
		for ee in neighbor:
			for i in neighbor[ee]:
				sources[i].append(xs[ee])
		result = []
		for i,xxs in sorted(sources.items(),key=lambda x:x[0]):
			if len(xxs)==1: result.append(xxs[0])
			else:			result.append(sum(xxs))
		return result

	def easy_case(self,x,neighbor_entities,neighbor_dict,assign,entities,relations):
		x = F.split_axis(x,len(neighbor_dict),axis=0)

		assignR = dict()
		bundle = defaultdict(list)
		for v,k in enumerate(neighbor_entities):
			for i in assign[v]:
				e = entities[i]
				if (e,k) in relations:	r =  relations[(e,k)]*2
				else:					r =  relations[(k,e)]*2+1
				assignR[(r,len(bundle[r]))] = v
				bundle[r].append(x[v])

		result = [0 for i in range(len(neighbor_dict))]
		for r in bundle:
			rx = bundle[r]
			if len(rx)==1:	result[assignR[(r,0)]] = rx[0]
			else:
				for i,x in enumerate(rx):
					result[assignR[(r,i)]] = x

		if self.pooling_method=='max':
			result = self.maxpooling(result,assign)
		if self.pooling_method=='avg':
			result = self.averagepooling(result,assign)
		if self.pooling_method=='sum':
			result = self.sumpooling(result,assign)
		result = F.concat(result,axis=0)
		return result



	"""
	# neighbor_entities=[(k,v)]
	# (e,k) in links
	# e = entities[i]
	# i in assing[v]
	"""
	"""
	source entityから出てるedgeが無い
	"""
	def __call__(self,x,neighbor_entities,neighbor_dict,assign,entities,relations):
		if self.layer==0:
			return self.easy_case(x,neighbor_entities,neighbor_dict,assign,entities,relations)

		if len(neighbor_dict)==1:
			x=[x]
		else:
			x = F.split_axis(x,len(neighbor_dict),axis=0)

		assignR = dict()
		bundle = defaultdict(list)
		for v,k in enumerate(neighbor_entities):
			for i in assign[v]:
				e = entities[i]
				if (e,k) in relations:	r =  relations[(e,k)]*2
				else:					r =  relations[(k,e)]*2+1
				assignR[(r,len(bundle[r]))] = v
				bundle[r].append(x[v])

		result = [0 for i in range(len(neighbor_dict))]
		for r in bundle:
			rx = bundle[r]
			if len(rx)==1:
				rx=rx[0]
				if r%2==0:	rx = getattr(self,self.forwardH[r//2][0])(rx)
				else:		rx = getattr(self,self.forwardT[r//2][0])(rx)
				result[assignR[(r,0)]] = rx
			else:
				size = len(rx)
				rx = F.concat(rx,axis=0)
				if r%2==0:	rx = getattr(self,self.forwardH[r//2][0])(rx)
				else:		rx = getattr(self,self.forwardT[r//2][0])(rx)
				rx = F.split_axis(rx,size,axis=0)
				for i,x in enumerate(rx):
					result[assignR[(r,i)]] = x

		if self.pooling_method=='max':
			result = self.maxpooling(result,assign)
		if self.pooling_method=='avg':
			result = self.averagepooling(result,assign)
		if self.pooling_method=='sum':
			result = self.sumpooling(result,assign)
		result = F.concat(result,axis=0)
		return result



class Model(chainer.Chain):
	def __init__(self, args):
		super(Model, self).__init__(
			embedE	= L.EmbedID(args.entity_size,args.dim),
			embedR	= L.EmbedID(args.rel_size,args.dim),
		)
		linksB = [('b{}'.format(i), Tunnel(args.dim,args.dropout_block,args.activate,args.layerR,args.is_residual,args.is_batchnorm, args.rel_size, args.pooling_method)) for i in range(args.order)]
		for link in linksB:
			self.add_link(*link)
		self.forwardB = linksB

		self.sample_size = args.sample_size
		self.dropout_embed = args.dropout_embed
		self.dropout_decay = args.dropout_decay
		self.depth = args.order
		self.is_embed = args.is_embed
		self.is_known = args.is_known
		self.threshold = args.threshold
		self.objective_function = args.objective_function
		self.is_bound_wr = args.is_bound_wr
		if args.use_gpu: self.to_gpu()


	def get_context(self,entities,links,relations,edges,order,xp):
		if self.depth==order:
			return self.embedE(xp.array(entities,'i'))

		assign = defaultdict(list)
		neighbor_dict = defaultdict(int)
		for i,e in enumerate(entities):
			"""
			(not self.is_known)
				unknown setting
			(not is_train)
				in test time
			order==0
				in first connection
			"""
			if e in links:
				if len(links[e])<=self.sample_size:	nn = links[e]
				else:		nn = random.sample(links[e],self.sample_size)
				if len(nn)==0:
					print('something wrong @ modelS')
					print('entity not in links',e,self.is_known,order)
					sys.exit(1)
			else:
				if len(edges[e])<=self.sample_size:	nn = edges[e]
				else:		nn = random.sample(edges[e],self.sample_size)
				if len(nn)==0:
					print('something wrong @ modelS')
					print('entity not in edges',e,self.is_known,order)
					sys.exit(1)
			for k in nn:
				if k not in neighbor_dict:
					neighbor_dict[k] = len(neighbor_dict)	# (k,v)
				assign[neighbor_dict[k]].append(i)
		neighbor = []
		for k,v in sorted(neighbor_dict.items(),key=lambda x:x[1]):
			neighbor.append(k)
		x = self.get_context(neighbor,links,relations,edges,order+1,xp)
		x = getattr(self,self.forwardB[order][0])(x,neighbor,neighbor_dict,assign,entities,relations)
		return x

	def train(self,positive,negative,links,relations,edges,xp):
		self.cleargrads()

		entities= set()
		for h,r,t in positive:
			entities.add(h)
			entities.add(t)
		for h,r,t in negative:
			entities.add(h)
			entities.add(t)

		entities = list(entities)

		x = self.get_context(entities,links,relations,edges,0,xp)
		x = F.split_axis(x,len(entities),axis=0)
		edict = dict()
		for e,x in zip(entities,x):
			edict[e]=x

		pos,rels = [],[]
		for h,r,t in positive:
			rels.append(r)
			pos.append(edict[h]-edict[t])
		pos = F.concat(pos,axis=0)
		xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		pos = F.batch_l2_norm_squared(pos+xr)

		neg,rels = [],[]
		for h,r,t in negative:
			rels.append(r)
			neg.append(edict[h]-edict[t])
		neg = F.concat(neg,axis=0)
		xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		neg = F.batch_l2_norm_squared(neg+xr)

		if self.objective_function=='relative': return sum(F.relu(self.threshold+pos-neg))
		if self.objective_function=='absolute': return sum(pos+F.relu(self.threshold-neg))


	def get_scores(self,candidates,links,relations,edges,xp,mode):
		entities = set()
		for h,r,t,l in candidates:
			entities.add(h)
			entities.add(t)
		entities = list(entities)
		xe = self.get_context(entities,links,relations,edges,0,xp)
		xe = F.split_axis(xe,len(entities),axis=0)
		edict = dict()
		for e,x in zip(entities,xe):
			edict[e]=x
		diffs,rels = [],[]
		for h,r,t,l in candidates:
			rels.append(r)
			diffs.append(edict[h]-edict[t])
		diffs = F.concat(diffs,axis=0)
		xr = self.embedR(xp.array(rels,'i'))
		if self.is_bound_wr:	xr = F.tanh(xr)
		scores = F.batch_l2_norm_squared(diffs+xr)
		return scores
