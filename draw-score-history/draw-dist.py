#! -*- coding:utf-8 -*-

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.ticker import *

from collections import defaultdict
import random
import os,sys

import datetime
def trace(*args):
	print(datetime.datetime.now().strftime('%H:%M:%S')+' '+' '.join(map(str,args)))

def draw(source_file,target_file,ymin=0,ymax=100000,is_logscale=False,sample_size=50,threshold=1):
	assert os.path.exists(source_file), ('cannot such file',source_file)

	plt.figure(figsize=(7,3))
	mode = 'test'
	for mode in ['dev:','test:']:
		if mode=='dev:': continue
		trace('\tcollecting properties')
		xmax,tuples = 0,[defaultdict(set),defaultdict(set)]
		relation_set = set()
		for text in open(source_file):
			if mode not in text: continue
			xmax+=1
			if xmax!=1: continue
			for terms in text[len(mode):].split(' '):
				h,r,t,l = map(int,terms.split(',')[:-1])
				tuples[l][r].add((h,r,t))
				relation_set.add(r)
		Rsize = len(relation_set)

		trace('\tinitlizing data')
		values = dict()
		for l in [0,1]:
			for r in tuples[l]:
				tuples[l][r] = list(tuples[l][r])
				for t in tuples[l][r]:
					values[t] = [0.0 for j in range(xmax)]
		trace('\tloading data')
		count=0
		for text in open(source_file):
			if mode not in text: continue
			for terms in text[len(mode):].split(' '):
				terms = terms.split(',')
				t = tuple(map(int,terms[:-2]))
				v = float(terms[-1])
				if is_logscale: v = math.copysign(math.log(v),v)
				values[t][count] = v
			count+=1

		trace('\tdrawing')
		Wsize=4
		if is_logscale: threshold = math.log(threshold)
		for r in range(Rsize):
			trace('\t\t relation:',r)
			# draw scores
			ax1 = plt.subplot(1+Rsize//Wsize,Wsize,r+1)
			ax1.grid(which='major', alpha=0.9,linestyle='-')
			if is_logscale:	  ax1.set_xticks(np.arange(0, 50*(1+xmax//50), 50))
			else:					ax1.set_xticks(np.arange(0, 1000*(1+xmax//1000), 1000))
			ax1.set_ylim(ymin, ymax)
			ax1.set_yticks(np.arange(ymin, ymax, 2))
			for _ in range(2*sample_size):
				l = random.randint(0,1)
				if len(tuples[l][r])==0: continue
				t = random.sample(tuples[l][r],1)[0]
				if l==1:	 ax1.plot(values[t],'b-',alpha=0.8,linewidth=0.1)
				if l==0:	 ax1.plot(values[t],'r-',alpha=0.8,linewidth=0.1)


			# draw threshold
			threshold_value=[threshold for i in range(xmax)]
			ax1.plot(threshold_value,'k-',alpha=1.0,linewidth=0.3)

			accuracy = [0.0 for i in range(xmax)]
			for i in range(xmax):
				for l in [0,1]:
					for t in tuples[l][r]:
						if values[t][i]<=threshold and l==1:
							accuracy[i]+=1
						if values[t][i]>=threshold and l==0:
							accuracy[i]+=1
			N = len(tuples[0][r])+len(tuples[1][r])+0.0
			if N==0.0:continue
			accuracy = [100.0-100.0*x/N for x in accuracy]
			ax2 = ax1.twinx()
			ax2.set_ylim([0,100.0])
			ax2.set_ylim([0,60.0])
			ax2.set_ylim([0,40.0])
			ax2.plot(accuracy,'g-',alpha=1.0,linewidth=0.7)

		plt.savefig(target_file+'-'+mode[:-1]+'.png',dpi=720)
		plt.clf()

import os
if __name__ == '__main__':
	source_file = 'score/example'
	target_file = 'history/example'
	ymin 			= 0
	ymax 			= 10
	is_logscale 	= True
	sample_size 	= 30
	draw(source_file, target_file, ymin, ymax, is_logscale, sample_size)

