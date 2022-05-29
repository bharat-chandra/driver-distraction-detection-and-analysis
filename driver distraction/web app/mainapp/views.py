from unittest import result
import warnings

from requests import Response
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
from django.http.response import StreamingHttpResponse
import pandas as pd
from mainapp.camera import VideoCamera1
from ast import Bytes, literal_eval
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
from io import BytesIO
import matplotlib
matplotlib.use('Agg')

model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model=tf.keras.models.load_model('./trained_models/modelv2.hdf5')

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
    fs = FileSystemStorage()
    request_file=request.FILES['filePath']
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
    try:
        result = Performance.objects.get(date=y1)
        l1,l2=eval(result.timestamp),eval(result.action)
        l1.append(x1)
        l2.append(predictedLabel)
        result.timestamp=str(l1)
        result.action=str(l2)
        result.save()
    except:
        result = Performance(driver_id=1,date=y1,timestamp=[x1],action=[predictedLabel])
        result.save()

    context={'filePathName':"."+fileurl,'predictedLabel':predictedLabel}
    return render(request,'index.html',context) 

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video1(req):
    return StreamingHttpResponse(gen(VideoCamera1()),content_type='multipart/x-mixed-replace;boundary=frame')

def display(request):
        df=Performance.objects.all()
        for i in range(len(df)):
            c=-1
            for j in eval(df[i].timestamp):
                    c+=1
                    plt.scatter(df[i].date,j)
                    plt.stem(df[i].date,j)
                    if float(j)>12:
                            plt.text(df[i].date,j,"-".join(df[i].date.split('-')[:2])+","+str(j)+"PM\n"+eval(df[i].action)[c],size=8,bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
                    else:
                            plt.text(df[i].date,j,"-".join(df[i].date.split('-')[:2])+","+str(j)+"AM\n"+eval(df[i].action)[c],size=8,bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))

        img = BytesIO()
        plt.gcf().autofmt_xdate()
        plt.savefig(img,format='png',bbox_inches='tight')
        img.seek(0)
        encoded_image1 = base64.b64encode(img.getvalue()).decode('utf8')
        to_send1 = 'data:image/png;base64, ' + str(encoded_image1)

        dates = [list(i.values())[0] for i in Performance.objects.values('date')]

        return render(request,'graph.html',context={'graph':to_send1,'dates':dates})

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

from django.views.decorators.csrf import csrf_exempt
@csrf_exempt
def query(request):
    response = HttpResponse()
    d={'date':[],'times':[],'actions':[]}
    date = request.POST['date']
    result = Performance.objects.all() if date=="all dates" else Performance.objects.filter(date=date)
    for k in range(len(result)):
        if date=='all dates':
            for h,i,j in zip(result[k].date,eval(result[k].timestamp),eval(result[k].action)):
                d['date'].append(h)
                d['times'].append(i)
                d['actions'].append(j)
    else:
        del d['date']
        for i,j in zip(eval(result[0].timestamp),eval(result[0].action)):
            d['times'].append(i)
            d['actions'].append(j)
    df = pd.DataFrame(d)
    
    freq=dict()
    for i in range(len(result)):
        for j in eval(result[i].action):
            if j in freq:
                freq[j]+=1
            else:
                freq[j]=1
    response.write(df.to_html())
    for i,j in freq.items():
        response.write("""
        <div style='background-color:black;border-radius:13px;padding:3px;'><span style='color:white'>{0}</span>
        <div style='background-color:orange;width:{1}0%;height:20px;border-radius:10px;'>
        </div>
        </div><br>""".format(i,j))
    return response