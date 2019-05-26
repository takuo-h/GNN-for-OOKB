# GNN-for-OOKB  


#How to use  
 download or clone this repository.  
 unzip each dataset in dataset dir.   
 then type "python main.py"  

#Requirements  
 chainer, cuda or numpy, more_itertools  

#How to modify this model or develop your models  
 add your models in models dir.  
 register your models in models/manager.py      
 use option -nn to use your models, e.g., "python -nn X" runs the model X  

#How to analyse and investigate results   
 apply draw-score-history/draw.py to your results(scores) with thresholds.      
 this script shows an image (following image is an example), that is how the scores are changed in the learning.   
 in particular, red and blue lines indicate negative and positive triplet's scores, respectively. the black line is your threshold, and the green line is accuracy using the threshold, i.e., how well the threshold splits triplets. this drawing is not the contribution of my paper, but i think it may help us to understand model's behavior.  

<img src="https://user-images.githubusercontent.com/17702908/33417466-e1fa11b4-d5e4-11e7-8bdd-6bf4f97325a8.png" width="600px">


#How to cite this work  
official paper: https://www.ijcai.org/proceedings/2017/0250.pdf  
official bibtex : https://www.ijcai.org/proceedings/2017/bibtex/250 (directly download the bibtex file)
