#!/bin/bash  

nohup python -u main.py -tD head-1000 > LOG/head_1000.log &
nohup python -u main.py -tD head-3000 > LOG/head_3000.log &
nohup python -u main.py -tD head-5000 > LOG/head_5000.log &
nohup python -u main.py -tD tail-1000 > LOG/tail_1000.log &
nohup python -u main.py -tD tail-3000 > LOG/tail_3000.log &
nohup python -u main.py -tD tail-5000 > LOG/tail_5000.log &
nohup python -u main.py -tD both-1000 > LOG/both_1000.log &
nohup python -u main.py -tD both-3000 > LOG/both_3000.log &
nohup python -u main.py -tD both-5000 > LOG/both_5000.log &