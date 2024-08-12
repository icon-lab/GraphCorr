from tqdm import tqdm
import torch
import numpy as np
import os

def getAtlasRegions():
    arr = np.loadtxt("/auto/data2/irmak/ROIextracted/Atlasses/schaefer_2018/Schaefer2018_400Parcels_7Networks_order.txt", delimiter="\t", dtype=str)
    lut = {}
    for row in arr:
        roi = row[0]
        region = row[1]
        lut[int(roi)]=region
    
    return(lut)


def train(trainDict):
    
    dataset = trainDict.dataset
    fold = trainDict.fold
    model = trainDict.model
    dname = trainDict.dname
    saveName = trainDict.saveName
    
    if 'corr' in saveName:
        corr = True
    else:
        corr = False

    dataLoader = dataset.getFold(fold, train=True)

    saliencyNodes_f = []
    saliencyNodes_m = []
    wBOLD_f = []
    wBOLD_m = []
    BOLD_f =[]
    BOLD_m =[]

    salFCPerWindow_f = []
    salFCPerWindow_m = []


    for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'Saliency for fold:{fold}')):

        xTrain = data["timeseries"]
        yTrain = data["label"]
        
        
        # NOTE: xTest and yTest are still on "cpu" at this point
        

        grads, pred, wBOLD = model.step(xTrain, yTrain)
        x = xTrain.squeeze(dim=0)
        torch.cuda.empty_cache()

        if pred == 0:
            
            if corr:
                salFCPerWindow_f.append(torch.mean(grads,dim=0).unsqueeze(dim=-1)) 
                wBOLD_f.append(wBOLD.unsqueeze(dim=-1))
            else:
                saliencyNodes_f.append(grads.unsqueeze(dim=-1))
            BOLD_f.append(x.unsqueeze(dim=-1))

        else:
            
            if corr:
                salFCPerWindow_m.append(torch.mean(grads,dim=0).unsqueeze(dim=-1))
                wBOLD_m.append(wBOLD.unsqueeze(dim=-1))
            else:
                saliencyNodes_m.append(grads.unsqueeze(dim=-1))
            BOLD_m.append(x.unsqueeze(dim=-1))

        
    
    saveDir = "/auto/k2/irmak/Projects/GraphCorr_3/SaliencyNew/SaliencyData/{}/".format(saveName)
    os.makedirs(saveDir + dname + "/fold_{}/TRAIN".format(fold),exist_ok=True)
   
    if corr:
        if len(salFCPerWindow_m) != 0:            
            salFCPerWindow_m = torch.cat(salFCPerWindow_m, dim=-1)
            print(salFCPerWindow_m.shape)
            wBOLD_m = torch.cat(wBOLD_m, dim=-1)

            torch.save(salFCPerWindow_m, saveDir + dname + "/fold_{}/TRAIN/saliencyAvgWindow_m.save".format(fold))    
            torch.save(wBOLD_m, saveDir + dname + "/fold_{}/TRAIN/wBOLD_m.save".format(fold))
            torch.save(BOLD_m, saveDir + dname + "/fold_{}/TRAIN/BOLD_m.save".format(fold))
        else:           
            wBOLD_m = None
            salFCPerWindow_m = None
            print('No males predicted in this fold: {}'.format(fold))

    
        if len(salFCPerWindow_f) != 0:           
            salFCPerWindow_f = torch.cat(salFCPerWindow_f, dim=-1)
            print(salFCPerWindow_f.shape)            
            wBOLD_f = torch.cat(wBOLD_f, dim=-1)

            torch.save(salFCPerWindow_f, saveDir + dname + "/fold_{}/TRAIN/saliencyAvgWindow_f.save".format(fold)) # avgd over windows, (N,N,subj)
            torch.save(wBOLD_f, saveDir + dname + "/fold_{}/TRAIN/wBOLD_f.save".format(fold))
            torch.save(BOLD_f, saveDir + dname + "/fold_{}/TRAIN/BOLD_f.save".format(fold))
        else:
            wBOLD_f = None
            salFCPerWindow_f = None
            print('No females predicted in this fold: {}'.format(fold))
    else:
        if len(saliencyNodes_m) != 0:
            salFC_m = torch.cat(saliencyNodes_m, dim=-1)   

            torch.save(salFC_m, saveDir + dname + "/fold_{}/TRAIN/saliencyPerSubject_m.save".format(fold))
            torch.save(BOLD_m, saveDir + dname + "/fold_{}/TRAIN/BOLD_m.save".format(fold))         
        else:
            salFC_m = None
            print('No males predicted in this fold: {}'.format(fold))
   
        if len(saliencyNodes_f) != 0:
            salFC_f = torch.cat(saliencyNodes_f, dim=-1)

            torch.save(BOLD_f, saveDir + dname + "/fold_{}/TRAIN/BOLD_f.save".format(fold))
            torch.save(salFC_f, saveDir + dname + "/fold_{}/TRAIN/saliencyPerSubject_f.save".format(fold))
        else:
            salFC_f = None
            print('No females predicted in this fold: {}'.format(fold))

            

    

   
    




    
    