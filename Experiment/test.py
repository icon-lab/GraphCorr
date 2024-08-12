from tqdm import tqdm
import torch
import numpy as np

def test(testDict):
    
    dataset = testDict.dataset
    modelTracker = testDict.modelTracker

    targetFolds = testDict.targetFolds

    for fold in targetFolds:

        model = modelTracker.test_setFold(fold)
        dataLoader = dataset.getFold(fold, train=False)


        preds = []
        probs = []
        groundTruths = []
        losses = []
        nodeInsight = []
        edgeInsight = []      

        for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'Testing fold:{fold}')):

            xTest = data["timeseries"]
            yTest = data["label"]

            # NOTE: xTrain and yTrain are still on "cpu" at this point

            test_loss, test_preds, test_probs, model_test_insight, yTest = model.step(xTest, yTest, train=False)
            
            torch.cuda.empty_cache()

            preds.append(test_preds)
            probs.append(test_probs)
            groundTruths.append(yTest)
            losses.append(test_loss)

            for insight in model_test_insight:
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
            model_test_insight.append({"name": "epochNodeLog", "value": nodeInsight, "type": "array"})
        if len(edgeInsight) != 0:
            edgeInsight = np.stack(edgeInsight)
            
            edgeInsight = np.mean(edgeInsight, axis=0) 
            model_test_insight.append({"name": "epochEdgeLog", "value": edgeInsight, "type": "array"})


        modelTracker.log_test(model_test_insight, np.mean(losses), preds, probs, groundTruths)
        
