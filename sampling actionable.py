import os 
import shutil
import pandas as pd 


#1
df = pd.read_csv('BCS-DBT labels-train-v2.csv')

files = os.listdir('D:/mammo.png')

actionable_list = []


for i in range(19148):
    if df['Actionable'][i] == 1:
        actionable_list.append(df.loc[i][0])
        pass

    else:
        pass

S_actionable = list(set(actionable_list))


for file in files:  
     k = file.split('-')[0]+'-'+file.split('-')[1]




     if k in S_actionable:
        dir_to_move= os.path.join('D:/mammo.png', file)
        shutil.move(dir_to_move,'D:/actionable') 
     else:
         pass 
    
   