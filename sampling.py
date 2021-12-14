import os 
import shutil
import pandas as pd 


#1
df = pd.read_csv('BCS-DBT labels-train-v2.csv')

files = os.listdir('D:/mammo.png')

normal_list = []


for i in range(19148):
    if df['Normal'][i] == 1:
        normal_list.append(df.loc[i][0])
        pass

    else:
        pass

S_normal = list(set(normal_list))


for file in files:  
     k = file.split('-')[0]+'-'+file.split('-')[1]




     if k in S_normal:
        dir_to_move= os.path.join('D:/mammo.png', file)
        shutil.move(dir_to_move,'D:/normal') 
     else:
         pass 
    
   
#2
# files = os.listidr('D:/normal')


# l = []
# Counts = []
# dict = {}
# for file in files:
#      view = file.split('_')[0]
     
#      if not l:
#          l.append(view)
#      elif view ==l[0]:
#          l.append(view)
#      else:
#          count = len(l)//2
#          Counts.append(count)
#          dict[view] = count
#          l.clear()
#          l.append(view)



# for key, values in dict.items():

#         dir_to_move = os.path.join('D:/normal', str(key)+'_'+values)
#         shutil.move(dir_to_move, 'D:/normal_center')






