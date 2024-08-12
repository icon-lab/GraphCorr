from tqdm import tqdm
import torch
import numpy as np

def train(trainDict):
    
    # load what is inside trainDict
    dataset = trainDict.dataset


    targetFolds = trainDict.targetFolds

    startEpoch = trainDict.startEpoch
    totalEpoch = trainDict.totalEpoch

    modelTracker = trainDict.modelTracker



    for fold in targetFolds:

        dataLoader = dataset.getFold(fold, train=True)  
        model = modelTracker.train_setFold(fold, len(dataLoader))


        for epoch_ in range(totalEpoch - startEpoch):
            epoch = epoch_ + startEpoch

            modelTracker.train_setEpoch(epoch)


            preds = []
            probs = []
            groundTruths = []
            losses = []
            nodeInsight = []
            edgeInsight = []

            for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
                
                xTrain = data["timeseries"] # (batchSize, N, dynamicLength)
                yTrain = data["label"] # (batchSize, )


                # NOTE: xTrain and yTrain are still on "cpu" at this point

                train_loss, train_preds, train_probs, model_train_insight, yTrain = model.step(xTrain, yTrain, train=True)

                torch.cuda.empty_cache()

                preds.append(train_preds)
                probs.append(train_probs)
                groundTruths.append(yTrain)
                losses.append(train_loss)

                for insight in model_train_insight:
                    if(insight["name"] == "subgraph_nodes"):
                        nodeInsight.append(insight["value"])
                    if(insight["name"] == "subgraph_edges"):
                        edgeInsight.append(insight["value"])
                

            preds = torch.cat(preds, dim=0).numpy()
            probs = torch.cat(probs, dim=0).numpy()
            groundTruths = torch.cat(groundTruths, dim=0).numpy()
            losses = torch.tensor(losses).numpy()
            if len(nodeInsight) != 0:
                nodeInsight = np.stack(nodeInsight)
                
                nodeInsight = np.mean(nodeInsight, axis=0) 
                model_train_insight.append({"name": "epochNodeLog", "value": nodeInsight, "type": "array"})
            if len(edgeInsight) != 0:
                edgeInsight = np.stack(edgeInsight)
                
                edgeInsight = np.mean(edgeInsight, axis=0) 
                model_train_insight.append({"name": "epochEdgeLog", "value": edgeInsight, "type": "array"})
            

            modelTracker.log_epoch_train(model_train_insight, np.mean(losses), preds, probs, groundTruths)



        modelTracker.finalizeTrainFold()
