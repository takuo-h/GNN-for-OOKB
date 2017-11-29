#! -*- coding:utf-8 -*-

import random,os,sys,math
from collections import defaultdict

def get_threshold_example(source_file):
	assert os.path.exists(source_file), ('cannot such file',source_file)

	mode = 'dev'
	for text in open(source_file):
		if mode not in text: continue
		last_text = text

	positive_scores, negative_scores = list(),list()
	for items in last_text[len(mode)+1:].split(' '):
		items = items.split(',')
		l,v = int(items[-2]),float(items[-1])
		if l==0: negative_scores.append(v)
		if l==1: positive_scores.append(v)
	averaged_positive_scores = sum(positive_scores)/len(positive_scores)
	averaged_negative_scores = sum(negative_scores)/len(negative_scores)
	print('E[ps]=',averaged_positive_scores)
	print('E[ns]=',averaged_negative_scores)
	return (averaged_positive_scores+averaged_negative_scores)/2.0

import os
if __name__ == '__main__':
	source_file 	= 'score/example'
	threshold = get_threshold_example(source_file)
	print('get threshood:',threshold)

