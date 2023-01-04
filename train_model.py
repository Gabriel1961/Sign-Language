from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import json
import os
import pickle
data = None



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

xTrain, xTest, yTrain, yTest = train_test_split(dataX, dataY,train_size=1)


print("Started training")
#model = MLPClassifier(random_state=2, max_iter=10000,activation="tanh",solver="lbfgs",verbose=True) #kinda bad
model = KNeighborsClassifier(n_neighbors=5)
model.fit(dataX,dataY)

#print("Model Score: ",model.score(xTest,yTest))

with open(os.getcwd()+"/model.pickle","wb") as f:
    pickle.dump(model,f)

print("Finished Training")