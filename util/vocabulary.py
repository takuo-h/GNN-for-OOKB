# -*- coding: utf-8 -*-

from collections import defaultdict,Counter
import os,codecs
class Vocabulary:
	def __init__(self,gen,vocab):
		word_freq = Counter()
		for line in gen:
			word_freq.update(line)
		_s2i = {'<EOS>':0,'<UNK>':1,'<TAIL>':2,'<HEAD>':3,}
		_i2s = {0:'<EOS>',1:'<UNK>',2:'<TAIL>',3:'<HEAD>',}
		for sc in word_freq.most_common(vocab-len(_s2i)):
			i,s = len(_s2i), sc[0]
			_s2i[s],_i2s[i] = i,s
		self._s2i,self._i2s = _s2i,_i2s

	def __len__(self):
		return len(self._s2i)

	def s2i(self, s):
		if s in self._s2i:  return self._s2i[s]
		return 1    # return index of <unk>

	def i2s(self, i):
		if i in self._i2s:  return self._i2s[i]
		return '<OOV>'  # maybe this will not be used

	def dump(self):
		result = []
		for i,s in sorted(self._i2s.items(),key=lambda x:x[0]):
			result.append(s)
		return result
