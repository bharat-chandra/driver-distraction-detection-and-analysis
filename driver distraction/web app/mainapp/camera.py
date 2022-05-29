import datetime
import os,urllib.request
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import playsound
import imutils
import time
import dlib
import cv2
from django.conf import settings
import numpy as np


ts=[]
audio = os.path.join(
			settings.BASE_DIR,'alarm.wav')
def sound_alarm(path):
    playsound.playsound(path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    C = dist.euclidean(eye[0],eye[3])
    ear = (A+B)/(2.0 * C)
    return ear
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 15
COUNTER = 0
ALARM_ON = False
print("[===>] loading facial landmark detector") 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lstart,lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart,rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

class VideoCamera1(object):
    def __init__(self):
        self.video = VideoStream(0).start()
        time.sleep(1.0)
        self.COUNTER = COUNTER
        self.ALARM_ON = ALARM_ON
    def __del__(self):
        self.video.stop()
    def get_frame(self):
        frame = self.video.read()
        #frame = cv2.resize(frame,(450,450))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray,0)
        for rect in rects:
            shape = predictor(gray,rect)
            shape = face_utils.shape_to_np(shape)
            lefteye = shape[lstart:lend]
            righteye = shape[rstart:rend]
            leftEar = eye_aspect_ratio(lefteye)
            rightEar = eye_aspect_ratio(righteye)
            ear  = (leftEar+rightEar) / 2.0
        
            leftEyeHull = cv2.convexHull(lefteye)
            rightEyeHull = cv2.convexHull(righteye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            if ear < EYE_AR_THRESH:
                self.COUNTER += 1

                if self.COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not self.ALARM_ON:
                        self.ALARM_ON = True

                        if audio != "":
                            t = Thread(target=sound_alarm,
                                args=(audio,))
                            t.deamon = True
                            t.start()

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # now = datetime.datetime.now()
                    # y1=str(now.day)+"-"+str(now.month)+"-"+str(now.year)
                    # x1=str(now.hour)+"."+str(now.minute)
                    # df=pd.read_csv("insights.csv",index_col=0)
                    # if y1 in df['dates'].values:
                    #     df['timestamps'][df.loc[df['dates']==y1].index.values[0]].append(x1)
                    #     df.to_csv("insights.csv",index=False)
                    # else:
                    #     df['dates'].append(y1)
                    #     df['timestamps'].append([x1])
                    #     df.to_csv("insights.csv",index=False)

            else:
                self.COUNTER = 0
                self.ALARM_ON = False

            cv2.putText(frame, "eye aspect ratio : {:.2f}".format(ear), (380, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret , jpeg = cv2.imencode('.jpg',frame)
        return jpeg.tobytes()