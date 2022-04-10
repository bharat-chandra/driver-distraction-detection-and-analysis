import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
from django.http.response import StreamingHttpResponse
import pandas as pd
from firstApp.camera import VideoCamera1
from ast import literal_eval
import tensorflow as tf
from django.core.files.storage import FileSystemStorage
from .models import Performance
import cv2,datetime
from tensorflow import Graph, Session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
import base64
import matplotlib
matplotlib.use('Agg')

model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model=tf.keras.models.load_model('./models/modelv2.hdf5')

def index(request):
    context={'a':1}
    return render(request,'index.html',context)

val = ['normal driving',
      ' texting - right',
      'talking on the phone - right',
      'texting - left',
      'talking on the phone - left',
      'operating the radio',
      'drinking',
      'reaching behind',
      'hair and makeup',
      'talking to passenger']

def predictImage(request):
    s=0
    fs = FileSystemStorage()
    request_file=request.FILES['filePath']
    if request_file.name.endswith('.mp4') or request_file.name.endswith('.mkv'):
        s=1
        #fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
        file = fs.save(request_file.name, request_file)
        fileurl = fs.url(file)
        cap = cv2.VideoCapture("."+fileurl)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('frame.jpg', frame)
                output = frame.copy()
                frame = tf.keras.preprocessing.image.load_img("frame.jpg",target_size=(64,64))
                frame = tf.keras.preprocessing.image.img_to_array(frame)
                with model_graph.as_default():
                    with tf_session.as_default():
                        predict = model.predict((np.expand_dims(frame, axis=0)/255-0.5))[0]>0.5
                predictedLabel,l = val[ predict.tolist().index(True) ],len(val[ predict.tolist().index(True) ])
                #frame = cv2.imread('frame.jpg')
                cv2.rectangle(output, (500,5), (700+l*10,40), (93, 0, 255),1)
                cv2.putText(output, predictedLabel, (500,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (93, 0, 255), 3)
                writer.write(output)
            else:
                break
        writer.release()
        cap.release()
        cv2.destroyAllWindows()
    if s==1:
        context={'filePathName':"opvideo.mp4",'predictedLabel':predictedLabel}
        return  render(request,'video.html',context) 

    # file_name = "pic.jpg"
    # file_name_2 = default_storage.save(file_name, fileObj)
    # file_url = default_storage.url(file_name_2)
    if request_file:
        file = fs.save(request_file.name, request_file)
        fileurl = fs.url(file)

    img = tf.keras.preprocessing.image.load_img("."+fileurl, target_size=(64, 64))
    x = tf.keras.preprocessing.image.img_to_array(img)
    
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict((np.expand_dims(x,axis=0)/255 - 0.5))[0] > 0.5
    print(predi)
    predictedLabel= val[ predi.tolist().index(True) ]

    now = datetime.datetime.now()
    y1=str(now.day)+"-"+str(now.month)+"-"+str(now.year)
    x1=str(now.hour)+"."+str(now.minute)
    print("############################################",y1,x1)
    df=pd.read_csv("insights.csv")
    if y1 in df['dates'].values:
        df.timestamps = df.timestamps.apply(literal_eval) 
        df['timestamps'][df.loc[df['dates']==y1].index.values[0]].append(x1)
        df.to_csv("insights.csv",index=False)
    else:
        df=df.append({'dates':y1,'timestamps':[x1]},ignore_index=True)
        df.to_csv("insights.csv",index=False)
    context={'filePathName':"."+fileurl,'predictedLabel':predictedLabel}
    return render(request,'index.html',context) 

def gene(frame):
	while True:
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video1(req):
    return StreamingHttpResponse(gen(VideoCamera1()),content_type='multipart/x-mixed-replace;boundary=frame')

xlabels = np.arange(1.00,25.00) 

def display(request):
        c=-1
        df=pd.read_csv('insights.csv')
        df.timestamps = df.timestamps.apply(literal_eval) 
        for i in df['dates'].tolist():
                for _ in range(1):
                        c+=1
                        for ij in df['timestamps'][c]:
                                plt.scatter(i,ij)
                                plt.stem(i,ij)
                                if float(ij)>12:
                                        plt.text(i,ij,"-".join(i.split('-')[:2])+","+str(ij)+"PM",size=5,bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
                                else:
                                        plt.text(i,ij,"-".join(i.split('-')[:2])+","+str(ij)+"AM",size=5,bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))

        img="plot.jpg"
        plt.savefig(img,dpi=150)

        img = cv2.imread(img)
        image_content1 = cv2.imencode('.jpg', img)[1].tostring()
        encoded_image1 = base64.encodestring(image_content1)
        to_send1 = 'data:image/jpg;base64, ' + str(encoded_image1, 'utf-8')
        return HttpResponse("""
		<img class='img-responsive' src=\'{}\'>
        <br><h3 class='text-primary'>performance log : </h3><br>
        <div>{}</div>
        """.format(to_send1,df.to_html()))

def insert(request):
    now = datetime.datetime.now()
    y1=str(now.day)+"-"+str(now.month)+"-"+str(now.year)
    x1=str(now.hour)+"."+str(now.minute)
    try:
        result = Performance.objects.get(date=y1)
        l=eval(result.timestamp)
        l.append(x1)
        result.timestamp=str(l)
        result.save()
    except:
        result = Performance(driver_id=1,date=y1,timestamp=[x1])
        result.save()
    return HttpResponse("done")