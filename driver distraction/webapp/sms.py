# # importing twilio
# from twilio.rest import Client

# # Your Account Sid and Auth Token from twilio.com / console
# account_sid = 'AC79ef10c2485d5844607de505a0f2d586'
# auth_token = 'abe221b22291164f086a4cfc0f90b54e'

# client = Client(account_sid, auth_token)

# ''' Change the value of 'from' with the number
# received from Twilio and the value of 'to'
# with the number in which you want to send message.'''
# message = client.messages.create(
# 							from_='+18624659426',
# 							body ='body',
# 							to ='+919676085525'
# 						)
# for i in range(10):
# ...     now = datetime.now()
# ...     cr=now.strftime("%H:%M:%S")
# ...     d=datetime.strptime(cr[:-4], "%H:%M") 
# ...     print(d.strftime("%I:%M %p") )
# ...     time.sleep(2)
# c=-1
# # >>> c=-1                   
# >>> for i in dates:
# ...     for j in range(1):
# ...             c+=1
# ...             for ij in ts[c]:
# ...                     plt.scatter(i,(ij))
# ...                     if float(ij)>12:
# ...                             plt.text(i,ij,i+str(ij)+"PM")
# ...                     else:
# ...                             plt.text(i,ij,i+str(ij)+"AM") 
# print(message.sid)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time 
ts=[]

df = pd.read_csv('insights.csv')
d={"date":[],"timestamp":[]}
