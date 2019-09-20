#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:22:32 2019

@author: aayush
"""

from models import model_dict
from torch.utils.data import DataLoader 
from dataset import IrisDataset
import torch
from utils import mIoU, CrossEntropyLoss2d,total_metric,get_nparams,Logger,GeneralizedDiceLoss,SurfaceLoss
import numpy as np
from dataset import transform
from opt import parse_args
import os
from utils import get_predictions
from tqdm import tqdm
import matplotlib.pyplot as plt
#%%

def lossandaccuracy(loader,model,factor):
    epoch_loss = []
    ious = []    
    model.eval()
    with torch.no_grad():
        for i, batchdata in enumerate(loader):
#            print (len(batchdata))
            img,labels,index,spatialWeights,maxDist=batchdata
            data = img.to(device)

            target = labels.to(device).long()  
            output = model(data)
            
            ## loss from cross entropy is weighted sum of pixel wise loss and Canny edge loss *20
            CE_loss = criterion(output,target)
            loss = CE_loss*(torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device)+(spatialWeights).to(torch.float32).to(device))
            
            loss=torch.mean(loss).to(torch.float32).to(device)
            loss_dice = criterion_DICE(output,target)
            loss_sl = torch.mean(criterion_SL(output.to(device),(maxDist).to(device)))
            
            ##total loss is the weighted sum of suface loss and dice loss plus the boundary weighted cross entropy loss
            loss = (1-factor)*loss_sl+factor*(loss_dice)+loss 
            
            epoch_loss.append(loss.item())
            predict = get_predictions(output)
            iou = mIoU(predict,labels)
            ious.append(iou)
    return np.average(epoch_loss),np.average(ious)

#%%
if __name__ == '__main__':
    
    args = parse_args()
    kwargs = vars(args)

#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    if args.useGPU:
        device=torch.device("cuda")
        torch.cuda.manual_seed(12)
    else:
        device=torch.device("cpu")
        torch.manual_seed(12)
        
    torch.backends.cudnn.deterministic=False
    
    if args.model not in model_dict:
        print ("Model not found !!!")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)
    
    LOGDIR = 'logs/{}'.format(args.expname)
    os.makedirs(LOGDIR,exist_ok=True)
    os.makedirs(LOGDIR+'/models',exist_ok=True)
    logger = Logger(os.path.join(LOGDIR,'logs.log'))
    
    model = model_dict[args.model]
    model  = model.to(device)
    torch.save(model.state_dict(), '{}/models/dense_net{}.pkl'.format(LOGDIR,'_0'))
    model.train()
    nparams = get_nparams(model)
    
    try:
        from torchsummary import summary
        summary(model,input_size=(1,640,400))
        print("Max params:", 1024*1024/4.0)
        logger.write_summary(str(model.parameters))
    except:
        print ("Torch summary not found !!!")
    
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5)

    criterion = CrossEntropyLoss2d()
    criterion_DICE = GeneralizedDiceLoss(softmax=True, reduction=True)
    criterion_SL = SurfaceLoss()
    
    Path2file = args.dataset
    train = IrisDataset(filepath = Path2file,split='train',
                             transform = transform, **kwargs)
    
    valid = IrisDataset(filepath = Path2file , split='validation',
                            transform = transform, **kwargs)
    
    trainloader = DataLoader(train, batch_size = args.bs,
                             shuffle=True, num_workers = args.workers)
    
    validloader = DataLoader(valid, batch_size = args.bs,
                             shuffle= False, num_workers = args.workers)
 
    test = IrisDataset(filepath = Path2file , split='test',
                            transform = transform, **kwargs)
    
    testloader = DataLoader(test, batch_size = args.bs,
                             shuffle=False, num_workers = args.workers)


#    alpha = 1 - np.arange(1,args.epochs)/args.epoch
    ##The weighing function for the dice loss and surface loss 
    alpha=np.zeros(((args.epochs)))
    alpha[0:np.min([125,args.epochs])]=1 - np.arange(1,np.min([125,args.epochs])+1)/np.min([125,args.epochs])
    if args.epochs>125:
        alpha[125:]=1
    ious = []        
    for epoch in range(args.epochs):
        for i, batchdata in enumerate(trainloader):
#            print (len(batchdata))
            img,labels,index,spatialWeights,maxDist= batchdata
            data = img.to(device)
            target = labels.to(device).long()  
            optimizer.zero_grad()            
            output = model(data)
            ## loss from cross entropy is weighted sum of pixel wise loss and Canny edge loss *20
            CE_loss = criterion(output,target)
            loss = CE_loss*(torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device)+(spatialWeights).to(torch.float32).to(device))
            
            loss=torch.mean(loss).to(torch.float32).to(device)
            loss_dice = criterion_DICE(output,target)
            loss_sl = torch.mean(criterion_SL(output.to(device),(maxDist).to(device)))
            
            ##total loss is the weighted sum of suface loss and dice loss plus the boundary weighted cross entropy loss
            loss = (1-alpha[epoch])*loss_sl+alpha[epoch]*(loss_dice)+loss 
#            
            predict = get_predictions(output)
            iou = mIoU(predict,labels)
            ious.append(iou)
    
            if i%10 == 0:
                logger.write('Epoch:{} [{}/{}], Loss: {:.3f}'.format(epoch,i,len(trainloader),loss.item()))
    
            loss.backward()
            optimizer.step()
            
        logger.write('Epoch:{}, Train mIoU: {}'.format(epoch,np.average(ious)))
        lossvalid , miou = lossandaccuracy(validloader,model,alpha[epoch])
        totalperf = total_metric(nparams,miou)
        f = 'Epoch:{}, Valid Loss: {:.3f} mIoU: {} Complexity: {} total: {}'
        logger.write(f.format(epoch,lossvalid, miou,nparams,totalperf))
        
        scheduler.step(lossvalid)
            
        ##save the model every epoch
        if epoch %1 == 0:
            torch.save(model.state_dict(), '{}/models/dense_net{}.pkl'.format(LOGDIR,epoch))

        ##visualize the ouput every 5 epoch
        if epoch %5 ==0:
            os.makedirs('test/epoch/labels/',exist_ok=True)
            os.makedirs('test/epoch/output/',exist_ok=True)
            os.makedirs('test/epoch/mask/',exist_ok=True)
            
            with torch.no_grad():
                for i, batchdata in tqdm(enumerate(testloader),total=len(testloader)):
                    img,labels,index,x,maxDist= batchdata
                    data = img.to(device)       
                    output = model(data)            
                    predict = get_predictions(output)
                    for j in range (len(index)):       
                        np.save('test/epoch/labels/{}.npy'.format(index[j]),predict[j].cpu().numpy())
                        try:
                            plt.imsave('test/epoch/output/{}.jpg'.format(index[j]),255*labels[j].cpu().numpy())
                        except:
                            pass
                        pred_img = predict[j].cpu().numpy()/3.0
                        inp = img[j].squeeze() * 0.5 + 0.5
                        img_orig = np.clip(inp,0,1)
                        img_orig = np.array(img_orig)
                        combine = np.hstack([img_orig,pred_img])
                        plt.imsave('test/epoch/mask/{}.jpg'.format(index[j]),combine)

