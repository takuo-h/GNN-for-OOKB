import sys,os
def check_exist(path):
	if path!='' and not os.path.exists(path):
		print('no such file or dir',path)
		sys.exit(1)
def check_exists(*path_list):
	for path in path_list:
		check_exist(path)

#----------------------------------------------------

def addmatrix(A,B):
	if len(A)==0:	return [[x for x in row] for row in B]
	else:	return [[x+y for x,y in zip(rowA,rowB)] for rowA,rowB in zip(A,B)]

#----------------------------------------------------

import datetime
def trace(*args):
	print(datetime.datetime.now().strftime('%H:%M:%S')+' '+' '.join(map(str,args)))
def dress(value,size=10):
	return (str(float(value))+'0'*size)[:size]
def drawConfusion(value,size=5):
	for row in value:
		print(', '.join([dress(x,size) for x in row]))
def drawConfusionDoubly(value1,value2,size=5):
	for row1,row2 in zip(value1,value2):
		tmp  = ', '.join([dress(x,size) for x in row1])
		tmp += '  :  '
		tmp += ', '.join([dress(x,size) for x in row2])
		print(tmp)

def drawLine(value,size=5):
	print(', '.join([dress(x,size) for x in value]))
def drawLineFinely(value,title,acc=-1):
	size=5
	if acc==-1:
		print(title+': '+', '.join([dress(x,size) for x in value]))
	else:
		print(title+': '+', '.join([dress(x,size) for x in value])+" acc:"+str(acc))

#----------------------------------------------------

import codecs
def read(file_path):
	with codecs.open(file_path, 'r', 'utf-8', errors='ignore') as read_f:
		for line in read_f:
			yield line.strip()
def write(file_path,text):
	with codecs.open(file_path, 'w', 'utf-8') as text_f:
		text_f.write('\n'.join(text))

#----------------------------------------------------

def gen_text(file_path,parse_func):
	for line in read(file_path):
		yield parse_func(line)
def pack(generator,batch_size):
	batch=[]
	for data in generator:
		batch.append(data)
		if len(batch)==batch_size:
			yield batch
			batch=[]
	if len(batch)!=0:yield batch
import random
def gen_spool(file_path,pool_size,batch_size):
	for pool in pack(read(file_path),pool_size):
		random.shuffle(pool)
		for batch in pack(pool,batch_size):
			yield batch

#----------------------------------------------------
