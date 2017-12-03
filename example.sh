#!/bin/bash  
cd datasets/standard
unzip Freebase13.zip
unzip WordNet11.zip
cd ..
unzip OOKB.zip
cd ..
python main.py -nn I -mF margins -T 300 > log
#nohup python -uO main.py -nn I -mF margins -T 300 > log & 
cp margins draw-score-history/scores
cd draw-score-history
python draw.py
