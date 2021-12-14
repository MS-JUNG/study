import os 
import shutil
import pandas as pd 


#1
df = pd.read_csv('BCS-DBT labels-train-v2.csv')

files = os.listdir('D:/mammo.png')

benign_list = []


for i in range(19148):
    if df['Benign'][i] == 1:
        benign_list.append(df.loc[i][0])
        pass

    else:
        pass

S_benign = list(set(benign_list))


for file in files:  
     k = file.split('-')[0]+'-'+file.split('-')[1]




     if k in S_benign:
        dir_to_move= os.path.join('D:/mammo.png', file)
        shutil.move(dir_to_move,'D:/benign') 
     else:
         pass 
    
   