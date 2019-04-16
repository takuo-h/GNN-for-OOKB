# -*- coding: utf-8 -*-

import os
os.environ['CHAINER_TYPE_CHECK'] = '0'

import chainer
from util.backend import Backend
from util.vocabulary import Vocabulary
from util.optimizer_manager import get_opt
from models.manager import get_model
from more_itertools import chunked
import random
from collections import defaultdict



import datetime
def trace(*args):
	print(datetime.datetime.now().strftime('%H:%M:%S')+' '+' '.join(map(str,args)))



# standard datasets
train_data	= list()
dev_data	= list()
test_data	= list()

# information to avoid using correct entities as negative samples
gold_heads		= defaultdict(set)	# entity set {h | (h,r,t) ∈ TrainData} for given (r,t)
									#  i.e., map : (r,t) -> {h | (h,r,t) ∈ TrainData}
gold_tails		= defaultdict(set)	# entity set {t | (h,r,t) ∈ TrainData} for given (h,r)
									#  i.e., map : (h,r) -> {t | (h,r,t) ∈ TrainData}
gold_relations	= dict()			# relation r in (h,r,t) ∈ TrainData    for given (h,t)
									#  i.e., map : (h,t) -> r

# information for bernoulli trick (reduce sampling ratio for common entities and vice versa )
tail_per_head	= defaultdict(set)	# size of entity set {t | (h,･,t) ∈ TrainData} for given h
									#  i.e., map : h -> #{t | (h,･,t) ∈ TrainData}
head_per_tail	= defaultdict(set)	# size of entity set {h | (h,･,t) ∈ TrainData} for given t
									#  i.e., map : t -> #{h | (h,･,t) ∈ TrainData}

# candidate for negative sampling focusing on given relation r
candidate_heads	= defaultdict(set)	# entity set {h | (h,r,･) ∈ TrainData} for given r
									#  i.e., map : r -> {h | (h,r,･) ∈ TrainData}
candidate_tails	= defaultdict(set)	# entity set {t | (･,r,t) ∈ TrainData} for given r
									#  i.e., map : r -> {t | (･,r,t) ∈ TrainData}

# connection between entities
glinks			= defaultdict(set)	# entity set {x | (x,･,e) or (e,･,x) ∈ TrainData} for given entity e
									#  i.e., map : e -> {x | (x,･,e) or (e,･,x) ∈ TrainData}
									#   which indicates neightborhoods for given entity e in TrainData
gedges 			= defaultdict(set)	# links between entities
									#  1. in starndard setting (no OOKB entity), gedges is same as glinks
									#  2. otherwise, gedges contains auxiliary edges 
									#      about known entities and unknown entities

trfreq = defaultdict(int)			# frequency of relations for negative sampling

def init_property_of_dataset():
	global gold_heads, gold_tails, gold_relations
	global candidate_heads,candidate_tails
	global glinks, gedges

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
		glinks[t].add(h)
		glinks[h].add(t)
		gold_relations[(h,t)]=r
	for e in glinks:
		glinks[e] = list(glinks[e])

	# convert a set into a list
	for r in candidate_heads:		candidate_heads[r] = list(candidate_heads[r])
	for r in candidate_tails:		candidate_tails[r] = list(candidate_tails[r])
	# convert a set into its size
	for h in tail_per_head:			tail_per_head[h] = len(tail_per_head[h])+0.0
	for t in head_per_tail:			head_per_tail[t] = len(head_per_tail[t])+0.0

	trace('set axiaulity')
	# set gedges based on standard setting or OOKB setting
	if args.train_file==args.auxiliary_file:
		trace('standard setting, use: edges=links')
		gedges = glinks
	else:
		trace('OOKB esetting, use: different edges')
		gedges = defaultdict(set)
		for line in open(args.auxiliary_file):
			items = line.strip().split('\t')
			items = list(map(int,items))
			h,r,t = items
			gold_relations[(h,t)]=r
			gedges[t].add(h)
			gedges[h].add(t)
		for e in gedges:
			gedges[e] = list(gedges[e])


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
		if h not in glinks or t not in glinks: continue
		dev_data.append((h,r,t,l,))
	print('dev size:',len(dev_data))

	trace('load test')
	for line in open(args.test_file):
		h,r,t,l = parse_line(line)
		if h not in glinks or t not in glinks: continue
		test_data.append((h,r,t,l,))
	print('test size:',len(test_data))


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
		loss = m.train(positive,negative,glinks,gold_relations,gedges,xp)
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
				current_score = m.get_scores(batch,glinks,gold_relations,gedges,xp,mode)
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

def get_sizes(args):
	relation,entity=-1,-1
	for line in open(args.train_file):
		h,r,t = list(map(int,line.strip().split('\t')))
		relation = max(relation,r)
		entity = max(entity,h,t)
	return relation+1,entity+1

import sys
def main(args):
	init_property_of_dataset()
	load_dataset()
	args.rel_size,args.entity_size = get_sizes(args)
	print('relation size:',args.rel_size,'entity size:',args.entity_size)

	xp = Backend(args)
	m = get_model(args)
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

	# trian, dev, test, and other filds
	p.add_argument('--train_file',      '-tF',  default='datasets/standard/WordNet11/serialized/train')
	p.add_argument('--dev_file',        '-vF',  default='datasets/standard/WordNet11/serialized/dev')
	p.add_argument('--test_file',       '-eF',  default='datasets/standard/WordNet11/serialized/test')
	p.add_argument('--auxiliary_file',  '-aF',  default='datasets/standard/WordNet11/serialized/train')
	# dirs
	p.add_argument('--param_dir',       '-pD',  default='')
	p.add_argument('--margin_file',     '-mF',  default='')

	# entity and relation sizes
	p.add_argument('--rel_size',  	'-Rs',      default=11,			type=int)
	p.add_argument('--entity_size', '-Es',      default=38194,		type=int)

	# model parameters (neural network)
	p.add_argument('--nn_model',		'-nn',  default='A0')
	p.add_argument('--activate',		'-af',  default='relu')
	p.add_argument('--pooling_method',	'-pM',  default='max')
	p.add_argument('--dim',         '-D',       default=200,    type=int)
	p.add_argument('--order',       '-O',       default=1,      type=int)
	p.add_argument('--threshold',   '-T',       default=300.0,  type=float)
	p.add_argument('--layerR' ,		'-Lr',      default=1,      type=int)

	# objective function
	p.add_argument('--objective_function',   '-obj',    default='absolute')

	# dropout rates
	p.add_argument('--dropout_block','-dBR',     default=0.0,  type=float)
	p.add_argument('--dropout_decay','-dDR',     default=0.0,   type=float)
	p.add_argument('--dropout_embed','-dER',     default=0.0,   type=float)

	# model flags
	p.add_argument('--is_residual',     '-nR',   default=False,   action='store_true')
	p.add_argument('--is_batchnorm',    '-nBN',  default=True,   action='store_false')
	p.add_argument('--is_embed',      	'-nE',   default=True,   action='store_false')
	p.add_argument('--is_known',    	'-iK',   default=False,   action='store_true')
	p.add_argument('--is_bound_wr',		'-iRB',  default=True,   action='store_false')

	# parameters for negative sampling (corruption)
	p.add_argument('--is_balanced_tr',    '-iBtr',   default=False,   action='store_true')
	p.add_argument('--is_bernoulli_trick', '-iBeT',  default=True,   action='store_false')

	# sizes
	p.add_argument('--train_size',  	'-trS',  default=1000,       type=int)
	p.add_argument('--batch_size',		'-bS',	 default=5000,        type=int)
	p.add_argument('--test_batch_size', '-tbS',  default=20000,        type=int)
	p.add_argument('--sample_size',		'-sS',  default=64,        type=int)
	p.add_argument('--pool_size',		'-pS',  default=128*5,      type=int)
	p.add_argument('--epoch_size',		'-eS',  default=1000,       type=int)

	# optimization
	p.add_argument('--opt_model',   "-Op",  default="Adam")
	p.add_argument('--alpha0',      "-a0",  default=0,      type=float)
	p.add_argument('--alpha1',      "-a1",  default=0,      type=float)
	p.add_argument('--alpha2',      "-a2",  default=0,      type=float)
	p.add_argument('--alpha3',      "-a3",  default=0,      type=float)
	p.add_argument('--beta0',       "-b0",  default=0.01,   type=float)
	p.add_argument('--beta1',       "-b1",  default=0.0001,  type=float)

	# seed to control random variables
	p.add_argument('--seed',        '-seed',default=0,      type=int)

	args = p.parse_args()
	return args

import random
if __name__ == '__main__':
	args = argument()
	print(args)
	print(' '.join(sys.argv))
	main(args)
