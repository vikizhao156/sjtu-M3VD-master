import pandas as pd     
import numpy as np
import keras
import os
import os.path
from matplotlib import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from keras.models import load_model
import pandas as pd
import sklearn


x_path='./train/nodule'
x_file=os.listdir(x_path)
x_test_path='./test'
model=load_model("weights.final.h5")

def get_testdataset():
    x_return=[]
    x_name=pd.read_csv("submit.csv") ['Id']
    for i in range(117):
        x_file_temp=os.path.join(x_test_path,x_name[i]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask*0.84+x_voxel*0.16
        x_return.append(x_temp[34:66,34:66,34:66])
    return x_return



x_test=np.array(get_testdataset())
x_test=x_test.reshape(x_test.shape[0],32,32,32,1)
x_test=x_test.astype('float32')/255
print(np.shape(x_test))
score=model.predict(x_test,batch_size=1)
csv = pd.read_csv("./sampleSubmission.csv")
csv.iloc[:, 1] = score[:, 1]
print(score[:,1])
csv.columns = ['Id', 'Predicted']
csv.to_csv("./submission.csv",index=None)



