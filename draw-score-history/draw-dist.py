#! -*- coding:utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.ticker import *

import random,os,sys,math
from collections import defaultdict

relationsWN11={0:'type_of',1:'synset_domain_topic',2:'has_instance',3:'member_holonym',4:'part_of',5:'has_part',6:'member_meronym',7:'similar_to',8:'subordinate_instance_of',9:'domain_region',10:'domain_topic'}

import datetime
def trace(*args):
	print(datetime.datetime.now().strftime('%H:%M:%S')+' '+' '.join(map(str,args)))

def draw(source_file,target_file,mode,ymin=0,ymax=100000,is_logscale=False,sample_size=50,threshold=300,best_accuracy=90.0):
	assert os.path.exists(source_file), ('cannot such file',source_file)

	plt.figure(figsize=(7,3))

	trace('\tcollecting properties')
	xmax,tuples = 0,[defaultdict(set),defaultdict(set)]
	relation_set = set()
	for text in open(source_file):
		if mode not in text: continue
		xmax+=1
		if xmax!=1: continue
		for terms in text[len(mode)+1:].split(' '):
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
		for terms in text[len(mode)+1:].split(' '):
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
		ax1.set_title(relationsWN11[r],fontsize=3)
		for _ in range(2*sample_size):
			l = random.randint(0,1)
			if len(tuples[l][r])==0: continue
			t = random.sample(tuples[l][r],1)[0]
			if l==1:     ax1.plot(values[t],'b-',alpha=0.8,linewidth=0.1)
			if l==0:     ax1.plot(values[t],'r-',alpha=0.8,linewidth=0.1)

		# draw threshold
		ax1.plot([threshold for i in range(xmax)],'k-',alpha=1.0,linewidth=0.3)

		# draw accuracy
		accuracy = [0.0 for i in range(xmax)]
		for i in range(xmax):
			for l in [0,1]:
				for t in tuples[l][r]:
					if values[t][i]<=threshold and l==1:
						accuracy[i]+=1
					if values[t][i]> threshold and l==0:
						accuracy[i]+=1
		N = len(tuples[0][r])+len(tuples[1][r])+0.0
		if N==0.0:continue

		ax2 = ax1.twinx()
		ax2.set_ylim([0,100.0])
		ax2.plot([100.0*x/N for x in accuracy],'g-',alpha=1.0,linewidth=0.5)

		ax1.tick_params(labelleft=False, labelright=False, labeltop=False,labelbottom=False,bottom=False)
		ax2.tick_params(labelleft=False, labelright=False, labeltop=False,labelbottom=False,bottom=False)
		ax1.yaxis.set_visible(False)
		ax2.yaxis.set_visible(False)
		ax1.spines["left"].set_color("none")
		ax2.spines["left"].set_color("none")
		ax1.spines["right"].set_color("none")
		ax2.spines["right"].set_color("none")
		ax1.spines["top"].set_color("none")
		ax2.spines["top"].set_color("none")




	trace('\t\t total')
	ax1 = plt.subplot(1+Rsize//Wsize,Wsize,r+2)
	ax1.set_title('total',fontsize=3)

	ax2 = ax1.twinx()
	ax2.set_ylim([0,100.0])

	ax1.tick_params(labelleft=False, labelright=False, labeltop=False,labelbottom=False,bottom=False)
	ax2.tick_params(labelleft=False, labelright=False, labeltop=False,labelbottom=False,bottom=False)
	ax1.yaxis.set_visible(False)
	ax2.yaxis.set_visible(False)
	ax1.spines["left"].set_color("none")
	ax2.spines["left"].set_color("none")
	ax1.spines["right"].set_color("none")
	ax2.spines["right"].set_color("none")
	ax1.spines["top"].set_color("none")
	ax2.spines["top"].set_color("none")

	# draw accuracy
	accuracy = [0.0 for i in range(xmax)]
	N = 0.0
	for r in range(Rsize):
		for i in range(xmax):
			for l in [0,1]:
				for t in tuples[l][r]:
					if values[t][i]<=threshold and l==1:
						accuracy[i]+=1
					if values[t][i]> threshold and l==0:
						accuracy[i]+=1
		N += len(tuples[0][r])+len(tuples[1][r])
	ax2.plot([100.0*x/N for x in accuracy],         'g-',alpha=0.8,linewidth=0.5)
	
	# draw best accuracy
	ax2.plot([best_accuracy for _ in range(xmax)],  'k:',alpha=1.0,linewidth=0.5)

	plt.savefig(target_file+'-'+mode+'.png',dpi=720)
	plt.clf()

import os
if __name__ == '__main__':
	source_file 	= 'score/example'
	target_file		= 'history/example'
	mode 			= 'test'
	ymin            = 0
	ymax            = 10
	is_logscale     = True
	sample_size     = 30
	threshold       = 200
	draw(source_file, target_file, mode, ymin, ymax, is_logscale, sample_size, threshold)

