import mediapipe
import cv2
import time
import pickle 
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
import importlib

from hand_crafted_model import CustomModel

WHITE = (255,255,255)
RED = (255,0,0)

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands
capture = cv2.VideoCapture(0)

symbols=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

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

# load model
model = None
with open(os.getcwd() + "\model.pickle","rb") as f:
    model = pickle.load(f)

assert model != None

# dectect hands loop 
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,max_num_hands=2) as hands:
    while (True):
        ret, frame = capture.read()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prediction = ""
        if results.multi_hand_landmarks:
            # predict result 
            dataX = [] # contains all normalized points in an array
            for pos in results.multi_hand_landmarks[0].landmark:
                dataX.append(pos.x)
                dataX.append(pos.y)
                dataX.append(pos.z)
            dataX = np.array([dataX])
            prediction = symbols[model.predict(dataX)[0]]
            # draw the landmarks
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
        
        textColor = WHITE
        cv2.putText(frame,prediction,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,textColor,2,cv2.LINE_AA)
        cv2.imshow("Main window", frame)
        if cv2.waitKey(1) == 27:
            break 

cv2.destroyAllWindows()
capture.release() 
