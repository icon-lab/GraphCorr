import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ks_2samp, wilcoxon, ttest_ind
import os

class Saliency(object):

    def __init__(self, options):

        self.options = options
        
        self.saliencyDir = options.workDir + "SaliencyNew/SaliencyData/{}/".format(options.saveName) + options.datasets[0]
        self.lut = self.getAtlasRegions()
        self.networks = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
        self.hemispheres = ["LH", "RH"]

        if 'corr' in options.saveName:
            self.corr = True
        else:
            self.corr = False


    def getAtlasRegions(self):
        arr = np.loadtxt(".../Atlasses/schaefer_2018/Schaefer2018_400Parcels_7Networks_order.txt", delimiter="\t", dtype=str) # atlas dir
        lut = {}
        for row in arr:
            roi = row[0]
            region = row[1]
            lut[int(roi)]=region
        
        return(lut)

    def getOverallSaliency(self): # returns node scores based on gradients (Num nodes, Num subj)
        
        salientNodes_fs = []
        salientNodes_ms = []
        for fold in range(self.options.kFold):

            if self.corr:  
                saliencyFC_f = torch.load(self.saliencyDir + "/fold_{}/TEST/saliencyPerSubject_f.save".format(fold))
                saliencyFC_m = torch.load(self.saliencyDir + "/fold_{}/TEST/saliencyPerSubject_m.save".format(fold))
                
                salFCPerWindow_f = torch.load(self.saliencyDir + "/fold_{}/TRAIN/saliencyAvgWindow_f.save".format(fold)) # avgd over windows, (N,N,subj)
                salFCPerWindow_m = torch.load(self.saliencyDir + "/fold_{}/TRAIN/saliencyAvgWindow_m.save".format(fold))

                salientNodes_f = torch.mean(saliencyFC_f, dim=0) # take window avg (for test)
                salientNodes_m = torch.mean(saliencyFC_m, dim=0)

                # now test and train samples follow the same pipeline

                salientNodes_f = torch.sum(salientNodes_f, dim=0) 
                salientNodes_m = torch.sum(salientNodes_m, dim=0)
                
                salientNodes_f_train = torch.sum(salFCPerWindow_f, dim=0) # node scores
                salientNodes_m_train = torch.sum(salFCPerWindow_m, dim=0)
                
                
            else:
                saliencyFC_f = torch.load(self.saliencyDir + "/fold_{}/TEST/saliencyPerSubject_f.save".format(fold)) # (N, N, num subj)
                saliencyFC_m = torch.load(self.saliencyDir + "/fold_{}/TEST/saliencyPerSubject_m.save".format(fold))

                saliencyFC_f_train = torch.load(self.saliencyDir + "/fold_{}/TRAIN/saliencyPerSubject_f.save".format(fold))
                saliencyFC_m_train = torch.load(self.saliencyDir + "/fold_{}/TRAIN/saliencyPerSubject_m.save".format(fold))

                salientNodes_f = torch.sum(saliencyFC_f, dim=1) # node scores
                salientNodes_m = torch.sum(saliencyFC_m, dim=1)

                salientNodes_f_train = torch.sum(saliencyFC_f_train, dim=1) # node scores
                salientNodes_m_train = torch.sum(saliencyFC_m_train, dim=1)


            salientNodes_fs.append(salientNodes_f)
            salientNodes_fs.append(salientNodes_f_train)
            salientNodes_ms.append(salientNodes_m)
            salientNodes_ms.append(salientNodes_m_train)

        
        salientNodes_fs = torch.cat(salientNodes_fs, dim=-1)
        salientNodes_ms = torch.cat(salientNodes_ms, dim=-1)
        
       
        return salientNodes_fs.numpy(), salientNodes_ms.numpy()

    

    def wilcoxonTest(self, samp1, samp2):
        
        if len(samp1) > len(samp2):
            samp1 = samp1[:len(samp2)]
        if len(samp2) > len(samp1):
            samp2 = samp2[:len(samp1)]
        res = wilcoxon(samp1, samp2)

        return res.pvalue 

    def ksTest(self, samp1, samp2):
        _, pval = ks_2samp(samp1, samp2)
        return pval

    def tTest(self, samp1, samp2):
        _, pval = ttest_ind(samp1, samp2, equal_var=False)
        return pval
    
    def histogram(self, samp1, samp2, network, hem, test):

        n_bins = 25
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
        colors = ['red', 'skyblue']

        ax0.hist(samp1, n_bins, density=True, histtype='bar', color=colors[0])
        ax0.set_title('female distribution')

        ax1.hist(samp2, n_bins, density=True, histtype='bar', color=colors[1])
        ax1.set_title('male distribution')
        fig.tight_layout()
        
        os.makedirs(self.saliencyDir+"/Results/"+ test, exist_ok=True)
        plt.savefig(self.saliencyDir+"/Results/{}/{}_{}.png".format(test,network, hem))


    def getNetworkMask(self, network, hemisphere):

        mask = np.zeros(400)

        for roi in range(400):
            region = self.lut[roi+1]
            if (network in region) and (hemisphere in region):
                mask[roi] = 1
        
        mask = np.expand_dims(mask, axis=1)

        return mask


    def getClassNetworks(self, classMatrix):
        
        subjNum = classMatrix.shape[1]
        classNetworkDict = {"LH":{}, "RH":{}}
        
        for network in self.networks:

            for hem in self.hemispheres:

                mask = self.getNetworkMask(network, hem)
                mask = np.repeat(mask, subjNum, axis=1) # mask for all subjects, 1 network
                ones = np.count_nonzero(mask[:,0])
                maskedScore = classMatrix * mask
                subjScore = np.sum(maskedScore, axis=0)
                subjScore /= ones
                classNetworkDict[hem][network] = subjScore
        
        return classNetworkDict
    
    def getDist(self):

        femaleMatrix, maleMatrix = self.getOverallSaliency()
    
        # normalize each subject 
        
        # meanf = np.mean(femaleMatrix, axis=0)
        # stdf = np.std(femaleMatrix, axis=0)
        # femaleMatrix = (femaleMatrix - meanf) / stdf

        femaleMatrix /= np.sum(femaleMatrix,axis=0, keepdims=True)
        maleMatrix /= np.sum(maleMatrix,axis=0, keepdims=True)
        
        # meanm = np.mean(maleMatrix, axis=0)
        # stdm = np.std(maleMatrix, axis=0)
        # maleMatrix = (maleMatrix - meanm) / stdm

        # can look at networks by using self.getClassNetworks(), or do the following analysis node-wise.
        
        NetworkDict_f = self.getClassNetworks(femaleMatrix)
        NetworkDict_m = self.getClassNetworks(maleMatrix)

        res = {"tTest":[], "wilcoxon":[], "ksTest": []}
        
        for network in self.networks:

            for hem in self.hemispheres:

                samples_f = NetworkDict_f[hem][network]
                samples_m = NetworkDict_m[hem][network]
                
                pval = self.tTest(samples_f,samples_m)
                testType = "tTest"

                if pval < 0.05:
                    res[testType].append({"hemisphere": hem, "network": network, "pval": pval})

                    # self.histogram(samples_f,samples_m, network, hem, testType)

                pval = self.wilcoxonTest(samples_f,samples_m)
                testType = "wilcoxon"

                if pval < 0.05:
                    res[testType].append({"hemisphere": hem, "network": network, "pval": pval})

                    # self.histogram(samples_f,samples_m, network, hem, testType)
                
                pval = self.ksTest(samples_f,samples_m)
                testType = "ksTest"

                if pval < 0.05:
                    res[testType].append({"hemisphere": hem, "network": network, "pval": pval})

                    # self.histogram(samples_f,samples_m, network, hem, testType)
                
        return(res)


