import os
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.models as models
#import models.resnet as resnet
from datasets.dataset_new import tommography
import numpy as np
import pandas as pd
import argparse
import random
from tqdm import tqdm
from datasets.dataset_CM import CMMD



torch.manual_seed(1995)
torch.cuda.manual_seed(1995)
torch.cuda.manual_seed_all(1995)
np.random.seed(1995)
# torch.cudnn.benchmark = False
# torch.cudnn.deterministic = True
random.seed(1995)

parser = argparse.ArgumentParser(description='multi_label classification')
parser.add_argument('--epoch', type= int, default= 100)
parser.add_argument('--model',type= str , default= 'resnet50')
parser.add_argument('--path', type = str , default = '')
parser.add_argument('--batch_size', type = int, default = 16)


def train(model,train_dl,valid_dl,epochs= 100, batch_size = None, writer= None):
    os.makedirs('D:/checkpoint/1115_checkpoint', exist_ok= True)
    # train scheme
    optimizer= optim.Adam(model.parameters(), lr= 0.0001 , weight_decay = 1e-8)
    ce_loss =nn.CrossEntropyLoss(weight= None)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
    dl_len = len(train_dl) * batch_size
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0
        train_acc = []    
        pbar = tqdm(train_dl, desc=f'Train Epoch {epoch+1}/{epochs}')
       
        for i,(image,label) in enumerate(pbar):
            # input to cuda
            image = image.cuda()
            label = label.cuda()
            # output 
            
            cls = F.softmax(model(image))
          
            batch_loss = ce_loss(cls,label)
          
            
            epoch_loss += batch_loss.item()
            cls_item = cls.detach()
  

            
            optimizer.zero_grad(True)
            batch_loss.backward()
            optimizer.step()
            
            arg = torch.argmax(cls, dim = 1)
            accuracy = 100*torch.sum(arg == label) // batch_size


            pbar.set_postfix({'loss(batch)' :batch_loss.item(), 'accuracy(batch)' : accuracy.item()})
        
        
        if writer is not None :
            writer.add_scalar('epoch/train', epoch_loss// dl_len , epoch+1)
        pbar.set_postfix({'loss(epoch)' : epoch_loss // dl_len})

        # eval
        with torch.no_grad():
            acc = torch.tensor(0)
            
            val_size = len(valid_dl)* batch_size
            pbar = tqdm(valid_dl, desc=f'Validation Epoch {epoch+1}/{epochs}')
            for i,(image,label) in enumerate(pbar):
                image = image.cuda()
                
                # hard vote
                cls = F.softmax(model(image)).cpu() 
                arg = torch.argmax(cls, dim = 1)
                #  torch.argmax(cls, dim = 1)
                accuracy = torch.sum(arg == label)
                
                 
                acc = acc + accuracy
                # 이미지당 맞는지 , image size = (N,C,H,W)

                pbar.set_postfix({'accuracy(batch) ' : 100*accuracy/batch_size })

            pbar.set_postfix({'accuracy(epoch) ' : 100*acc/val_size })
            print(f'accuracy(epoch{epoch+1}) : ' + str(100*acc/val_size) )
            if writer is not None :
                writer.add_scalar('epoch/valid_acc', 100*acc // val_size, epoch+1)
            #breakpoint()
        
        writer.flush()

        # save 
        if (epoch+1) % 5 == 0:
            scheduler.step()
        if (epoch) % 10 == 0 :
            torch.save(model, f'D:/checkpoint/1115_checkpoint/{epoch}.pth')

    torch.save(model, f'D:/checkpoint/1115_checkpoint/whole.pth')




def main(arg):
    epoch = arg.epoch
    mode = arg.model
    path = arg.path
    batch_size = arg.batch_size
    model = None



    # 모델 초기화
    if mode == 'resnet18':
        model = models.resnet18(pretrained = True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(2048,3)
    elif mode =='resnet34':
        model = models.resnet34(pretrained = True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(2048,3)
    elif mode =='resnet50':
        model = models.resnet50(pretrained = True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(2048,3)
    elif mode =='resnet101':
        model = models.resnet101(pretrained = True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(2048,3)
    elif mode =='resnet152':
        model = models.resnet152(pretrained = True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(2048,3)

    model.cuda()
    
    # 토치 데이터 전처리
    trans_fn = transforms.Compose([
        transforms.Resize((499,614)),
        
        transforms.Normalize((0.5,),(0.5,))


    ])

    # random Split
    
    # data = tommography(trans_func=trans_fn) 
    # train_size = int(len(data)*0.8)
    # valid_size = len(data) - train_size

    # train_data , valid_data = torch.utils.data.random_split(data, [train_size, valid_size])

    # 데이터셋 초기화
    train_data = CMMD(mode = 'train', trans_func=trans_fn)
    valid_data = CMMD(mode = 'test', trans_func=trans_fn)
     
    train_dl = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    valid_dl = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
    writer = SummaryWriter()
    train(model,train_dl,valid_dl,epoch, batch_size = batch_size, writer = writer)


    return model



if __name__ == '__main__':
    
    arg = parser.parse_args()

    main(arg)