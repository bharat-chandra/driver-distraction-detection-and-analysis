import cv2,time

from cv2 import imread
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

model = load_model('modelv2.hdf5')
val = ['normal driving',
      ' texting - right',
      'talking on the phone - right',
      'texting - left',
      'talking on the phone - left',
      'operating the radio',
      'drinking',
      'reaching behind',
      'hair and makeup',
      'talking to passenger'
      ]

cap = cv2.VideoCapture("op.mp4")

while True:
    ret, frame = cap.read()
    if ret:
        #output = cv2.imwrite('frame.jpg', frame)
        # frame = load_img("frame.jpg",target_size=(64,64))
        # frame = img_to_array(frame)
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (64,64)).astype("float32")
        predict = model.predict((np.expand_dims(frame, axis=0)/255-0.5))[0]>0.5
        predictedLabel,l = val[ predict.tolist().index(True) ],len(val[ predict.tolist().index(True) ])
        #frame = imread('frame.jpg')
        cv2.rectangle(output, (300,5), (400+l*20,40), (93, 0, 255),1)
        cv2.putText(output, predictedLabel, (300,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (93, 0, 255), 2)
        cv2.imshow('frame',output)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    else:
        break

out.release()
cap.release()
cv2.destroyAllWindows()