import os, sys
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import imageio
import glob
import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image
import pydicom

class tommography(Dataset):
    def __init__(self,mode ='Train' ,trans_func =  None,csv_label_file ='C:/Users/pc/Desktop/csv/BCS-DBT labels-train-v2', csv_path_file ='C:/Users/pc/Desktop/csv/BCS-DBT labels-train-v2', path_dir ='C:\Users\pc\Desktop\새 폴더 (2)'):
        self.path_dir = glob.glob(path_dir+'/*')
        self.classify = pd.read_csv(csv_label_file)
        self.path = pd.read_csv()
        self.mode = mode
      



    

        self.trans_func = trans_func

    


    def __len__(self):
        return len(self.path_dir)
    




    
    def __getitem__(self,idx):

        image = Image.open(self.path_dir[idx])

        pixel = np.array(image)
        Patient_id = None
        Patient_id = (self.path_dir[idx].split('/')[-1]).split('-')[0]+'-'+(self.path_dir[idx].split('/')[-1]).split('-')[1]




        label = None
        
        if self.mode is 'Test':
            Patient_id = self.classify.loc[idx][0]


        
        if self.classify['Normal'][idx] == 1:
            label = 0
            pass
        elif self.classify['Actionable'][idx] ==1 :
            label = 1
            pass
        elif self.classify['Benign'][idx] ==1:
            label = 2
            pass
        else :
            label =3
            pass

        if self.trans_func is not None:
            pixel = self.trans_func(pixel)

      
        

     
        return pixel, label


    # 'C:\Users\pc\Desktop\새 폴더 (2)\DBT-P00006-lcc'

        # key = []
        #     for i in range(19152):

        #         l = self.classify.values.tolist()[i][0]
        #         key.append(l)





        





# def main():
#     test =label()

#     image, label=test[0]
    


