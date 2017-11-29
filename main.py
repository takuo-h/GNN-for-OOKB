# -*- coding: utf-8 -*-

import os
os.environ['CHAINER_TYPE_CHECK'] = '0'

import chainer
#import cupy as cp # in default, cupy (gpu) is tuned off. please remove the first # to use gpu
import numpy as np
from chainer import serializers
import util.general_tool as tool
from util.backend import Backend
from util.vocabulary import Vocabulary
from util.optimizer_manager import get_opt
from models.manager import get_model

#----------------------------------------------------------------------------
from collections import defaultdict

candidate_heads=defaultdict(set)
gold_heads = defaultdict(set)
candidate_tails=defaultdict(set)
gold_tails = defaultdict(set)
black_set = set()

tail_per_head=defaultdict(set)
head_per_tail=defaultdict(set)

train_data,dev_data,test_data=list(),list(),list()
trfreq = defaultdict(int)

glinks, grelations, gedges = 0,0,0

from more_itertools import chunked
import random

def initilize_dataset():
	global candidate_heads,gold_heads,candidate_tails,gold_tails
	global glinks, grelations, gedges

	# get properties of knowledge graph
	tool.trace('load train')
	grelations = dict()
	glinks = defaultdict(set)
	for line in tool.read(args.train_file):
		h,r,t = list(map(int,line.strip().split('\t')))
		grelations[(h,t)]=r
		glinks[t].add(h)
		glinks[h].add(t)
		gold_heads[(r,t)].add(h)
		gold_tails[(h,r)].add(t)
		candidate_heads[r].add(h)
		candidate_tails[r].add(t)
		tail_per_head[h].add(t)
		head_per_tail[t].add(h)
	for e in glinks:
		glinks[e] = list(glinks[e])

	for r in candidate_heads:		candidate_heads[r] = list(candidate_heads[r])
	for r in candidate_tails:		candidate_tails[r] = list(candidate_tails[r])
	for h in tail_per_head:			tail_per_head[h] = len(tail_per_head[h])+0.0
	for t in head_per_tail:			head_per_tail[t] = len(head_per_tail[t])+0.0

	tool.trace('set axiaulity')
	# switch standard setting or OOKB setting
	if args.train_file==args.auxiliary_file:
		tool.trace('standard setting, use: edges=links')
		gedges = glinks
	else:
		tool.trace('OOKB esetting, use: different edges')
		gedges = defaultdict(set)
		for line in tool.read(args.auxiliary_file):
			h,r,t = list(map(int,line.strip().split('\t')))
			grelations[(h,t)]=r
			gedges[t].add(h)
			gedges[h].add(t)
		for e in gedges:
			gedges[e] = list(gedges[e])

	global train_data,dev_data,test_data,trfreq
	# load train
	train_data = set()
	for line in open(args.train_file):
		h,r,t = list(map(int,line.strip().split('\t')))
		train_data.add((h,r,t,))
		trfreq[r]+=1
	train_data = list(train_data)
	for r in trfreq:
		trfreq[r] = args.train_size/(float(trfreq[r])*len(trfreq))

	# load dev
	tool.trace('load dev')
	dev_data = set()
	for line in open(args.dev_file):
		h,r,t,l = list(map(int,line.strip().split('\t')))
		dev_data.add((h,r,t,l,))
	dev_data=list(dev_data)
	print('dev size:',len(dev_data))

	# load test
	tool.trace('load test')
	test_data = [line.strip().split('\t') for line in open(args.test_file)]
	test_data = [tuple(map(int,x)) for x in test_data]
	test_data = list(set(test_data))
	print('test size:',len(test_data))


def generator_train_with_corruption(args):
	skip_rate = args.train_size/float(len(train_data))

	positive,negative=list(),list()
	random.shuffle(train_data)
	for i in range(len(train_data)):
		h,r,t = train_data[i]
		if (-r,t) in black_set: continue
		if (h, r) in black_set: continue
		if args.is_balanced_tr:
			if random.random()>trfreq[r]: continue
		else:
			if random.random()>skip_rate: continue

		# tph/Z
		head_ratio = 0.5
		if args.is_bernoulli_trick:
			head_ratio = tail_per_head[h]/(tail_per_head[h]+head_per_tail[t])
		if random.random()>head_ratio:
			cand = random.choice(candidate_heads[r])
			while cand in gold_heads[(r,t)]:
				cand = random.choice(candidate_heads[r])
			h = cand
		else:
			cand = random.choice(candidate_tails[r])
			while cand in gold_tails[(h,r)]:
				cand = random.choice(candidate_tails[r])
			t = cand
		if len(positive)==0 or len(positive) <= args.batch_size:
			positive.append(train_data[i])
			negative.append((h,r,t))
		else:
			yield positive,negative
			positive,negative = [train_data[i]],[(h,r,t)]
	if len(positive)!=0:
		yield positive,negative

#----------------------------------------------------------------------------

def train(args,m,xp,opt):
	Loss,N = list(),0
	for positive, negative in generator_train_with_corruption(args):
		loss = m.train(positive,negative,glinks,grelations,gedges,xp)
		loss.backward()
		opt.update()
		Loss.append(float(loss.data)/len(positive))
		N += len(positive)
		del loss
	return sum(Loss),N

def dump_current_scores_of_devtest(args,m,xp):
	if args.margin_file=='': return
	for mode in ['dev','test']:
		if mode=='dev': 	current_data = dev_data
		if mode=='test': 	current_data = test_data

		scores = list()
		for batch in  chunked(current_data, args.test_batch_size):
			with chainer.using_config('train',False), chainer.no_backprop_mode():
				current_score = m.get_scores(batch,glinks,grelations,gedges,xp,mode)
			for v,(h,r,t,l) in zip(current_score.data, batch):
				values = (h,r,t,l,v)
				values = map(str,values)
				values = ','.join(values)
				scores.append(values)
			del current_score
		with open(args.margin_file,'a') as wf:
			wf.write(mode+':'+' '.join(scores)+'\n')

def get_sizes(args):
	relation,entity=-1,-1
	for line in open(args.train_file):
		h,r,t = list(map(int,line.strip().split('\t')))
		relation = max(relation,r)
		entity = max(entity,h,t)
	return relation+1,entity+1

import sys
def main(args):
	initilize_dataset()
	args.rel_size,args.entity_size = get_sizes(args)
	print('relation size:',args.rel_size,'entity size:',args.entity_size)

	xp = Backend(args)
	m = get_model(args)
	opt = get_opt(args)
	opt.setup(m)
	for epoch in range(args.epoch_size):
		opt.alpha = args.beta0/(1.0+args.beta1*epoch)
		trLoss,Ntr = train(args,m,xp,opt)
		tool.trace('epoch:',epoch,'tr Loss:',tool.dress(trLoss),Ntr)
		dump_current_scores_of_devtest(args,m,xp)


#----------------------------------------------------------------------------
"""
	-tF dataset/data/Freebase13/train \
	-dF dataset/data/Freebase13/train \
	-eF dataset/data/Freebase13/test \
"""
from argparse import ArgumentParser
def argument():
	p = ArgumentParser()
	p.add_argument('--train_file',      '-tF',  default='dataset/standard/Freebase13/serialized/train')
	p.add_argument('--dev_file',        '-vF',  default='dataset/standard/Freebase13/serialized/dev')
	p.add_argument('--test_file',       '-eF',  default='dataset/standard/Freebase13/serialized/test')
	p.add_argument('--auxiliary_file',  '-aF',  default='dataset/standard/Freebase13/serialized/train')
	p.add_argument('--param_dir',       '-pD',  default='')
	p.add_argument('--margin_file',     '-mF',  default='')

	p.add_argument('--use_gpu',     '-g',   default=False,  action='store_true')
	p.add_argument('--gpu_device',  '-gd',  default=0,      type=int)

	p.add_argument('--seed',        '-seed',default=0,      type=int)

	p.add_argument('--rel_size',  '-Rs',       default=10,    type=int)
	p.add_argument('--entity_size', '-Es',       default=18,    type=int)

	p.add_argument('--is_residual',     '-iR',   default=False,  action='store_true')
	p.add_argument('--is_batchnorm',    '-iBN',  default=False,  action='store_true')
	p.add_argument('--is_embed',      	'-nE',   default=True,   action='store_false')
	p.add_argument('--is_known',    	'-nK',   default=True,   action='store_false')
	p.add_argument('--is_balanced_tr',    '-nBtr',   default=True,   action='store_false')
	p.add_argument('--is_balanced_dev',   '-iBde',   default=False,   action='store_true')
	p.add_argument('--is_bernoulli_trick',  '-iBeT',   default=False,   action='store_true')

	p.add_argument('--is_bound_wr',   '-iRB',   default=False,   action='store_true')
	p.add_argument('--object_kind',   '-oK',    default=1,  type=int)

	p.add_argument('--layerR' ,      '-Lr',      default=1,      type=int)

	p.add_argument('--dim',         '-D',       default=200,    type=int)
	p.add_argument('--order',       '-O',       default=1,      type=int)
	p.add_argument('--threshold',   '-T',       default=1.0,   type=float)

	p.add_argument('--dropout_block','-dBR',     default=0.05,  type=float)
	p.add_argument('--dropout_decay','-dDR',     default=0.0,   type=float)
	p.add_argument('--dropout_embed','-dER',     default=0.0,   type=float)

	p.add_argument('--train_size',  '-trS',  default=1000,       type=int)

	p.add_argument('--batch_size',  '-bS',  default=64,        type=int)
	p.add_argument('--test_batch_size',  '-tbS',  default=64,        type=int)
	p.add_argument('--sample_size', '-sS',  default=64,        type=int)
	p.add_argument('--pool_size',   '-pS',  default=128*5,      type=int)
	p.add_argument('--epoch_size',  '-eS',  default=1000,       type=int)
	p.add_argument('--nn_model',    '-nn',  default='I')
	p.add_argument('--activate',    '-af',  default='tanh')
	p.add_argument('--pooling_method',    '-pM',  default='max')

	p.add_argument('--opt_model',   "-Op",  default="Adam")
	p.add_argument('--alpha0',      "-a0",  default=0,      type=float)
	p.add_argument('--alpha1',      "-a1",  default=0,      type=float)
	p.add_argument('--alpha2',      "-a2",  default=0,      type=float)
	p.add_argument('--alpha3',      "-a3",  default=0,      type=float)
	p.add_argument('--beta0',       "-b0",  default=0.01,   type=float)
	p.add_argument('--beta1',       "-b1",  default=0.001,  type=float)

	args = p.parse_args()
	return args

import random
if __name__ == '__main__':
	args = argument()
	print(args)
	print(' '.join(sys.argv))
	main(args)
