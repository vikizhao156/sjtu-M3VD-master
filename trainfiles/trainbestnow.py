# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 09:33:23 2019

@author: 99628
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:36:59 2019

@author: 99628
"""
import pandas as pd     
import numpy as np
import keras
import os
import os.path
os.environ['KERAS_BACKEND']='tensorflow'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D,MaxPooling2D,BatchNormalization,Activation
from keras.optimizers import Adam
from matplotlib import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from mylib.models import densesharp, metrics, losses,DenseNet
from mylib.models.DenseNet import createDenseNet
from keras.optimizers import SGD
import pandas as pd
from keras.losses import categorical_crossentropy

from keras.callbacks import EarlyStopping

#########数据总数465 分出训练集的样本数为4/5 1/5为test集  留出法
x_path='./train/nodule'
x_file=os.listdir(x_path)
x_filef_train=x_file[0:400]
x_filef_test=x_file[400:465]
x_test_path='./test'

def get_dataset():
    x_return_train=[]
    x_return_test=[]
    x_name=pd.read_csv("train_val.csv") ['name']
    for i in range(len(x_filef_train)):
        x_file_temp=os.path.join(x_path,x_name[i]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask+x_voxel*0.1
        x_return_train.append(x_temp[34:66,34:66,34:66])
    for i in range(len(x_filef_test)):
        x_file_temp=os.path.join(x_path,x_name[i+400]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask+x_voxel*0.1
        x_return_test.append(x_temp[34:66,34:66,34:66])
    return  x_return_train,x_return_test

def get_label():
    x_label=pd.read_csv("train_val.csv") ['lable']
    x_train_label=keras.utils.to_categorical(x_label,2)[0:400]
    x_test_label=keras.utils.to_categorical(x_label,2)[400:465]
    return x_train_label,x_test_label
def get_testdataset():
    x_return=[]
    x_name=pd.read_csv("submit.csv") ['Id']
    for i in range(117):
        x_file_temp=os.path.join(x_test_path,x_name[i]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask+x_voxel*0.1
        x_return.append(x_temp[34:66,34:66,34:66])        
    return x_return

densenet_depth = 40
densenet_growth_rate = 20

x_train,x_test=get_dataset()
x_train=np.array(x_train)
x_test=np.array(x_test)
x_train_label,x_test_label=get_label()
x_train=x_train.reshape(x_train.shape[0],32,32,32,1)
x_train=x_train.astype('float32')/255
x_test=x_test.reshape(x_test.shape[0],32,32,32,1)
x_test=x_test.astype('float32')/255

x_predict=np.array(get_testdataset())
x_predict=x_predict.reshape(x_predict.shape[0],32,32,32,1)
x_predict=x_predict.astype('float32')/255
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
nb_classes = 2
batch_size = 32
model = createDenseNet(nb_classes=nb_classes,img_dim=[32,32,32,1],depth=densenet_depth,growth_rate = densenet_growth_rate)
model.compile(loss=categorical_crossentropy,optimizer=Adam(), metrics=['accuracy'])
model.summary()  # print the model

model.fit(x_train, x_train_label,batch_size=12, epochs=150,validation_split=0.20, verbose=2, shuffle=False, callbacks=[early_stopping])
loss,accuracy = model.evaluate(x_train,x_train_label)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (loss,accuracy))
loss,accuracy = model.evaluate(x_test,x_test_label)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (loss,accuracy))
####最终估计的地方
#print(model.predict(x_predict))
model.save("model_without_mask.h5")

score1=model.predict(x_predict)
print(score1)
score=score1[:,1].T
file_name = 'test1.csv'
save = pd.DataFrame(score)
save.to_csv(file_name)
