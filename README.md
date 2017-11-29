# GNN-for-OOKB  


#How to use  
 download or clone this repository  
 unzip each dataset in dataset dir   
 python main.py  

#Requirements  
 chainer, cuda or numpy, more_itertools  

#How to modify this model or develop your models  
 add your models in models dir  
 register your models in models/manager.py       
 use option -nn to use your models, e.g., "python -nn X" runs the model X  

#How to undarstand results   
 run draw-score-history/draw.py with your threshold    
 this script shows learning history of positive and negative triplets's scores
 ( next image is an example of it )  
  
<img src="https://user-images.githubusercontent.com/17702908/33366678-4acd11de-d52f-11e7-842c-08bd52ebfce7.png" width="600px">


#How to cite this work  
official paper: https://www.ijcai.org/proceedings/2017/0250.pdf  
official bibtex : https://www.ijcai.org/proceedings/2017/bibtex/250 (directly download the bibtex file)
