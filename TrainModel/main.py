from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import json
import os
import pickle
data = None

TRAIN_ON_FINGER_TIPS = True

def getFingerTipData(data):
    tipsData = []
    for samp in data:
        tips = []
        for idx in [4,8,12,16]:
            tips.append(samp[idx*3])
            tips.append(samp[idx*3+1])
            tips.append(samp[idx*3+2])
        tipsData.append(tips)
    return tipsData

with open(os.getcwd()+"/data.json","r") as f:
    data = json.load(f)
dataX = data["X"]
dataY = data["Y"]



print("Started training")
model = MLPClassifier(random_state=1, max_iter=2000,activation="tanh")
model.fit(dataX,dataY)


with open(os.getcwd()+"/model.pickle","wb") as f:
    pickle.dump(model,f)

print("Finished Training")