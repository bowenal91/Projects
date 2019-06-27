import numpy as np
import random
import glob
import shutil
import os
os.chdir("pos_data")
names_pos = glob.glob("*")
os.chdir("../neg_data/")
names_neg = glob.glob("*")
os.chdir("../")

random.shuffle(names_pos)
random.shuffle(names_neg)

N1 = len(names_pos)
N2 = len(names_neg)

frac_train = 0.8
N1train = frac_train*N1
N1test = N1-N1train

N2train = frac_train*N2
N2test = N2-N2train

for i,filename in enumerate(names_pos):
    if i<N1train:
        shutil.copyfile("pos_data/"+filename,"train/pos/"+filename)
    else:
        shutil.copyfile("pos_data/"+filename,"test/pos/"+filename)

for i,filename in enumerate(names_neg):
    if i<N2train:
        shutil.copyfile("neg_data/"+filename,"train/neg/"+filename)
    else:
        shutil.copyfile("neg_data/"+filename,"test/neg/"+filename)


