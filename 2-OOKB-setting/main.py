# -*- coding: utf-8 -*-

import os
os.environ['CHAINER_TYPE_CHECK'] = '0'

import chainer
from util.backend import Backend
from util.vocabulary import Vocabulary
from util.optimizer_manager import get_opt
import model
from more_itertools import chunked
import random
from collections import defaultdict



import datetime
def trace(*args):
	print(datetime.datetime.now().strftime('%H:%M:%S')+' '+' '.join(map(str,args)))



train_data	= list()
dev_data	= list()
test_data	= list()

gold_heads		= defaultdict(set)
gold_tails		= defaultdict(set)
gold_relations	= dict()

tail_per_head	= defaultdict(set)
head_per_tail	= defaultdict(set)


candidate_heads	= defaultdict(set)
candidate_tails	= defaultdict(set)

train_link			= defaultdict(set)
aux_link 			= defaultdict(set)
trfreq = defaultdict(int)

def init_property_of_dataset():
	global gold_heads, gold_tails, gold_relations
	global candidate_heads,candidate_tails
	global train_link, aux_link

	# get properties of knowledge graph
	trace('load train')
	for line in open(args.train_file):
		# parse items
		items = line.strip().split('\t')
		items = list(map(int,items))
		h,r,t = items
		# properties about correct triplets for negative sampling (corruption)
		candidate_heads[r].add(h)
		candidate_tails[r].add(t)
		gold_heads[(r,t)].add(h)
		gold_tails[(h,r)].add(t)
		tail_per_head[h].add(t)
		head_per_tail[t].add(h)
		train_link[t].add(h)
		train_link[h].add(t)
		gold_relations[(h,t)]=r
	for e in train_link:
		train_link[e] = list(train_link[e])
#	assert 37112 in train_link, ('something wrong')

	# convert a set into a list
	for r in candidate_heads:		candidate_heads[r] = list(candidate_heads[r])
	for r in candidate_tails:		candidate_tails[r] = list(candidate_tails[r])
	# convert a set into its size
	for h in tail_per_head:			tail_per_head[h] = len(tail_per_head[h])+0.0
	for t in head_per_tail:			head_per_tail[t] = len(head_per_tail[t])+0.0

	trace('set axiaulity')
	# set aux_link based on standard setting or OOKB setting
	trace('OOKB esetting, use: different edges')
	aux_link = defaultdict(set)
	for line in open(args.auxiliary_file):
		items = line.strip().split('\t')
		items = list(map(int,items))
		h,r,t = items
		gold_relations[(h,t)]=r
		aux_link[t].add(h)
		aux_link[h].add(t)
	for e in aux_link:
		aux_link[e] = list(aux_link[e])


def parse_line(line):
	items = line.strip().split('\t')
	items = list(map(int,items))
	return items

def load_dataset():
	# load datasets as standard machine learning settings
	global train_data,dev_data,test_data,trfreq
	trace('load train')
	for line in open(args.train_file):
		h,r,t = parse_line(line)
		train_data.append((h,r,t,))
		trfreq[r]+=1
	train_data = list(train_data)
	for r in trfreq:
		trfreq[r] = args.train_size/(float(trfreq[r])*len(trfreq))

	trace('load dev')
	for line in open(args.dev_file):
		h,r,t,l = parse_line(line)
		dev_data.append((h,r,t,l,))
	trace('dev size:',len(dev_data))

	trace('load test')
	for line in open(args.test_file):
		h,r,t,l = parse_line(line)
		test_data.append((h,r,t,l,))
	trace('test size:',len(test_data))


def generator_train_with_corruption(args):
	skip_rate = args.train_size/float(len(train_data))

	positive,negative = list(),list()
	random.shuffle(train_data)
	for i in range(len(train_data)):
		h,r,t = train_data[i]
		if args.is_balanced_tr:
			# if negative sampling (corruption) is based on balanced (reflect frequency of relations)
			if random.random()>trfreq[r]: continue
		else:
			if random.random()>skip_rate: continue

		# tph/Z
		head_ratio = 0.5
		# if negative sampling is based on bernoulli trick,
		#  sampling ratio will be modified
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
		# collect positive triples until the size is over batch size
		if len(positive)==0 or len(positive) <= args.batch_size:
			positive.append(train_data[i])
			negative.append((h,r,t))
		else:
			# positive triplet's size is over batch size
			yield positive,negative
			positive,negative = [train_data[i]],[(h,r,t)]
	if len(positive)!=0:
		yield positive,negative

#----------------------------------------------------------------------------

def train(args,m,xp,opt):
	Loss,N = list(),0
	for positive, negative in generator_train_with_corruption(args):
		loss = m.train(positive,negative,train_link,gold_relations,aux_link,xp)
		loss.backward()
		opt.update()
		Loss.append(float(loss.data)/len(positive))
		N += len(positive)
		del loss
	return sum(Loss),N

def dump_current_scores_of_devtest(args,m,xp):
	for mode in ['dev','test']:
		if mode=='dev': 	current_data = dev_data
		if mode=='test': 	current_data = test_data

		scores, accuracy = list(),list()
		for batch in  chunked(current_data, args.test_batch_size):
			with chainer.using_config('train',False), chainer.no_backprop_mode():
				current_score = m.get_scores(batch,train_link,gold_relations,aux_link,xp,mode)
			for v,(h,r,t,l) in zip(current_score.data, batch):
				values = (h,r,t,l,v)
				values = map(str,values)
				values = ','.join(values)
				scores.append(values)
				if v < args.threshold:
					if l==1:	accuracy.append(1.0)
					else: 		accuracy.append(0.0)
				else:
					if l==1:	accuracy.append(0.0)
					else: 		accuracy.append(1.0)
			del current_score
		trace('\t ',mode,sum(accuracy)/len(accuracy))
		if args.margin_file!='':
			with open(args.margin_file,'a') as fp:
				fp.write(mode+':'+' '.join(scores)+'\n')

import sys
def main(args):
	init_property_of_dataset()
	load_dataset()
	print('relation size:',args.rel_size,'entity size:',args.entity_size)

	xp = Backend(args)
	m = model.Model(args)
	opt = get_opt(args)
	opt.setup(m)
	for epoch in range(args.epoch_size):
		opt.alpha = args.beta0/(1.0+args.beta1*epoch)
		trLoss,Ntr = train(args,m,xp,opt)
		trace('epoch:',epoch,'tr Loss:',trLoss,Ntr)
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

	# GPU	
	p.add_argument('--use_gpu',     '-g',   default=False,  action='store_true')
	p.add_argument('--gpu_device',  '-gd',  default=0,      type=int)

	# dirs
	p.add_argument('--target_dir',      '-tD',  default='head-1000')
	p.add_argument('--param_dir',       '-pD',  default='')
	p.add_argument('--margin_file',     '-mF',  default='')

	# entity and relation sizes
	p.add_argument('--rel_size',  	'-Rs',      default=11,			type=int)
	p.add_argument('--entity_size', '-Es',      default=38195,		type=int)

	# model parameters (neural network)
	p.add_argument('--pooling_method',	'-pM',  default='avg')
	p.add_argument('--dim',         '-D',       default=200,    type=int)
	p.add_argument('--order',       '-O',       default=1,      type=int)
	p.add_argument('--threshold',   '-T',       default=300.0,  type=float)
	p.add_argument('--layerR' ,		'-Lr',      default=1,      type=int)

	# parameters for negative sampling (corruption)
	p.add_argument('--is_balanced_tr',    '-iBtr',   default=False,   action='store_true')
	p.add_argument('--is_bernoulli_trick', '-iBeT',  default=True,   action='store_false')

	# sizes
	p.add_argument('--train_size',  	'-trS',  default=1000,       type=int)
	p.add_argument('--batch_size',		'-bS',	 default=5000,        type=int)
	p.add_argument('--test_batch_size', '-tbS',  default=20000,        type=int)
	p.add_argument('--sample_size',		'-sS',  default=64,        type=int)
	p.add_argument('--pool_size',		'-pS',  default=128*5,      type=int)
	p.add_argument('--epoch_size',		'-eS',  default=10000,       type=int)

	# optimization
	p.add_argument('--opt_model',   "-Op",  default="Adam")
	p.add_argument('--alpha0',      "-a0",  default=0,      type=float)
	p.add_argument('--alpha1',      "-a1",  default=0,      type=float)
	p.add_argument('--alpha2',      "-a2",  default=0,      type=float)
	p.add_argument('--alpha3',      "-a3",  default=0,      type=float)
	p.add_argument('--beta0',       "-b0",  default=0.1,   type=float)
	p.add_argument('--beta1',       "-b1",  default=0.00001,  type=float)

	# seed to control random variables
	p.add_argument('--seed',        '-seed',default=0,      type=int)

	p = p.parse_args()


	p.train_file		= 'datasets/OOKB/'+p.target_dir+'/train'
	p.dev_file			= 'datasets/OOKB/'+p.target_dir+'/dev'
	p.test_file			= 'datasets/OOKB/'+p.target_dir+'/test'
	p.auxiliary_file	= 'datasets/OOKB/'+p.target_dir+'/aux'

	return p

import random
if __name__ == '__main__':
	args = argument()
	print(args)
	print(' '.join(sys.argv))
	main(args)
