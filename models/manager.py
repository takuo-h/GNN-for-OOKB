#! -*- coding:utf-8 -*-

from models.ModelI import Model as I

import sys
def get_model(args):
	if args.nn_model=='I':			return I(args)
	print('no such model:',args.nn_model)
	sys.exit(1)
