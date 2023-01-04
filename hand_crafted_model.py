import numpy as np
import json 
import pickle
import math
import os 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

with open(os.getcwd()+"/data.json","r") as f:
    data = json.load(f)

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

def plotHand(hand,ax):
    pairs = [[0,1],[1,2],[2,3],[3,4], 
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],
            [0,17]
        ]
    for pair in pairs:
        p1 = hand[pair[0]]
        p2 = hand[pair[1]]
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]],'k-')
    ax.scatter(hand[:,0],hand[:,1],hand[:,2])

class CustomModel:
    def fit(self,trainX:np.ndarray,trainY:np.ndarray) -> None:
        assert trainX.shape[1] == 21*3
        assert len(trainY.shape) == 1
        assert len(trainY) == len(trainX)
        trainX = trainX.reshape(-1,21,3)
        lableCount = trainY[-1]+1
        self.meanData = np.zeros((lableCount,21,3))
        cnt = np.zeros(lableCount)
        # accumulate
        for i in range(len(trainX)):
            self.meanData[trainY[i]] += trainX[i]
            cnt[trainY[i]] += 1 
        # normalize 
        for i in range(len(cnt)):
            self.meanData[i] /= cnt[i]
        
    
    def predict(self,testX:np.ndarray) -> np.ndarray:
        assert testX.shape[1] == 63
        testX = testX.reshape(-1,21,3)
        Y = []
        for sample in testX:
            minp = np.inf
            minIdx = -1
            # for each point in the mean positions of the hand we compute  the inverse distance sqr, and acc. as a score for the current prediction
            for i,mean in enumerate(self.meanData):
                distAcc = 0
                for j,point in enumerate(sample):
                    dif = point-mean[j]
                    distAcc += np.sqrt(np.dot(dif,dif))               
                if distAcc < minp:
                    minp = distAcc
                    minIdx = i
            Y.append(minIdx)
        return Y

        

dataX = np.array(data["X"])
dataY = np.array(data["Y"])

model = CustomModel()
model.fit(dataX,dataY)

with open(os.getcwd()+"/model.pickle","wb") as f:
    pickle.dump(model,f)

'''
# for visualizing data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# define the update function
cnt = 0 
def update(num):
    # clear the current axes
    global cnt
    ax.clear()
    plotHand(model.meanData[cnt],ax)
    cnt+=1
    if cnt >= dataY[-1] + 1:
        cnt = 0

# create the animation using the update function and a frame rate of 30 fps
animation = FuncAnimation(fig, update, frames=range(10), repeat=True, interval=1000 / 10)

# show the plot
plt.show()


'''