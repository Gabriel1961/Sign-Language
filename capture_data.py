import mediapipe
import cv2
import time
import json
def writeDataToFile(data):
    with open("data.json","w") as f:
        json.dump(data,f)

WHITE = (255,255,255)
RED = (255,0,0)
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

capture = cv2.VideoCapture(0)
startTime = time.time()

NR_OF_SAMPLES_PER_SYMBOL = 40
NR_OF_SYMBOLS = 25
symbols=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
SAMPLE_INTERVAL_PER_LETTER = 10 # in seconds
SAMPLE_INTERVAL = SAMPLE_INTERVAL_PER_LETTER / NR_OF_SAMPLES_PER_SYMBOL
csymIdx = 0
csampleIdx = 0
lastSampleTime = startTime
triggerKey = 0
data = []

def getLables():
    lables = []
    for i in range(NR_OF_SYMBOLS):
        for j in range(NR_OF_SAMPLES_PER_SYMBOL):
            lables.append(i)
    return lables

with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,max_num_hands=2) as hands:
    while (True):
        ret, frame = capture.read()
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            
            if  len(results.multi_hand_landmarks) > 0 and time.time() > lastSampleTime + SAMPLE_INTERVAL and triggerKey == 1:

                lastSampleTime = time.time()
                sample = [] # contains all normalized points in an array
                for pos in results.multi_hand_landmarks[0].landmark:
                    sample.append(pos.x)
                    sample.append(pos.y)
                    sample.append(pos.z)
                #print("recorded sample #" + str(csampleIdx) + "for letter " + symbols[csymIdx])
                data.append(sample)
                csampleIdx+=1
                if csampleIdx == NR_OF_SAMPLES_PER_SYMBOL:
                    triggerKey = 0
                    csampleIdx = 0 
                    csymIdx += 1
                    if csymIdx == NR_OF_SYMBOLS:
                        objData = {
                            "X":data,
                            "Y":getLables()
                        }
                        writeDataToFile(objData)
                        exit()
            
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
        
        textColor = WHITE if triggerKey == 0 else RED 

        cv2.putText(frame,symbols[csymIdx],(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,textColor,2,cv2.LINE_AA)
        cv2.imshow("Main window", frame)
        pressedKey = cv2.waitKey(1)
        if pressedKey == 27:
            break 
        elif pressedKey == 32:
            triggerKey = 1
        

cv2.destroyAllWindows()
capture.release() 
