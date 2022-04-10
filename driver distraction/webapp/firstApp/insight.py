# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from ast import literal_eval
# import base64
# from io import BytesIO

# xlabels = np.arange(1.00,25.00,1.00)

# def display():
#         c=-1
#         df=pd.read_csv('insights.csv')
#         df.timestamps = df.timestamps.apply(literal_eval) 
#         for i in df['dates'].tolist():
#                 for _ in range(1):
#                         c+=1
#                         for ij in df['timestamps'][c]:
#                                 plt.scatter(i,ij)
#                                 if float(ij)>12:
#                                         plt.text(i,ij,i+str(ij)+"PM")
#                                 else:
#                                         plt.text(i,ij,i+str(ij)+"AM")

#         plt.xticks(xlabels)
#         plt.show()


