from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def getAtlasRegions():
    arr = np.loadtxt("/auto/data2/irmak/ROIextracted/Atlasses/schaefer_2018/Schaefer2018_400Parcels_7Networks_order.txt", delimiter="\t", dtype=str)
    lut = {}
    for row in arr:
        roi = row[0]
        region = row[1]
        lut[int(roi)]=region
    
    return(lut)


def test(testDict):
    
    dataset = testDict.dataset
    fold = testDict.fold
    model = testDict.model
    saveName = testDict.saveName
    dname = testDict.dname

    if 'corr' in saveName:
        corr = True
    else:
        corr = False
    

    dataLoader = dataset.getFold(fold, train=False)

    saliencyNodes_f = []
    saliencyNodes_m = []
    wBOLD_f = []
    wBOLD_m = []
    BOLD_f =[]
    BOLD_m =[]


    for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'Saliency for fold:{fold}')):

        xTest = data["timeseries"]
        yTest = data["label"]
        # subjId = data["subjId"][0]

        # xTest.shape: 1, 400, 1200

        # NOTE: xTest and yTest are still on "cpu" at this point
        # test set size : 218 HCP REST

        grads, pred, wBOLD = model.step(xTest, yTest)
        
        x = xTest.squeeze(dim=0)
        
        torch.cuda.empty_cache()

        if pred == 0:
            saliencyNodes_f.append(grads.unsqueeze(dim=-1))
            BOLD_f.append(x.unsqueeze(dim=-1))
            if corr:
                wBOLD_f.append(wBOLD.unsqueeze(dim=-1)) # only needed for corr
        else:
            saliencyNodes_m.append(grads.unsqueeze(dim=-1))
            BOLD_m.append(x.unsqueeze(dim=-1))
            if corr:
                wBOLD_m.append(wBOLD.unsqueeze(dim=-1)) # only needed for corr

        

    saveDir = "/auto/k2/irmak/Projects/GraphCorr_3/SaliencyNew/SaliencyData/{}/".format(saveName)
    
    
    if len(saliencyNodes_m) != 0:
        saliencyNodes_m = torch.cat(saliencyNodes_m, dim=-1)
        print(saliencyNodes_m.shape)
        if corr:
            wBOLD_m = torch.cat(wBOLD_m, dim=-1)
    else:
        saliencyNodes_m = None
        if corr:
            wBOLD_m = None

    
    if len(saliencyNodes_f) != 0:
        saliencyNodes_f = torch.cat(saliencyNodes_f, dim=-1)
        print(saliencyNodes_f.shape)
        if corr:
            wBOLD_f = torch.cat(wBOLD_f, dim=-1)
    else:
        saliencyNodes_f = None
        if corr:
            wBOLD_f = None

    

    os.makedirs(saveDir + dname + "/fold_{}/TEST".format(fold),exist_ok=True)
    if saliencyNodes_f != None:

        torch.save(saliencyNodes_f, saveDir + dname + "/fold_{}/TEST/saliencyPerSubject_f.save".format(fold))
        if corr:
            torch.save(wBOLD_f, saveDir + dname + "/fold_{}/TEST/wBOLD_f.save".format(fold))
        torch.save(BOLD_f, saveDir + dname + "/fold_{}/TEST/BOLD_f.save".format(fold))

    if saliencyNodes_m != None:

        torch.save(saliencyNodes_m, saveDir + dname + "/fold_{}/TEST/saliencyPerSubject_m.save".format(fold))
        if corr:    
            torch.save(wBOLD_m, saveDir + dname + "/fold_{}/TEST/wBOLD_m.save".format(fold))
        torch.save(BOLD_m, saveDir + dname + "/fold_{}/TEST/BOLD_m.save".format(fold))
    

    
    






          

        
