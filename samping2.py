import os 
import shutil
import pandas as pd 




files = sorted(os.listdir('D:/normal'))


l = []
Counts = []
dict = {}
for file in files:
     view = file.split('_')[0]
    
     if not l:
         l.append(view)
     elif view == l[0]:
         l.append(view)
     elif view != l[0]:    
         count = len(l)//2
         Counts.append(count)
         dict[l[0]] = count
         l.clear()
         l.append(view)


for key, values in dict.items():

        dir_to_move = os.path.join('D:\\normal', key+'_'+str(values)+'.png')
        shutil.move(dir_to_move, 'D:\\normal_center')



# # DBT-P00003-lmlo_33
# D:/normal\DBT-P00003-lmlo_33