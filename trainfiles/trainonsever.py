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
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D
from keras.optimizers import Adam
from matplotlib import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from mylib.model.densesharp import get_compiled
#########数据总数465 分出训练集的样本数为4/5 1/5为test集  留出法
x_path='./train/nodule'
x_file=os.listdir(x_path)
x_filef_train=x_file[0:372]
x_filef_test=x_file[373:464]
x_test_path='./test'
x_test_file=os.listdir(x_test_path)
def get_dataset():
    x_return_train=[]
    x_return_test=[]
    for i in range(len(x_filef_train)):
        x_file_temp=os.path.join(x_path,x_filef_train[i])
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask
        x_return_train.append(x_temp[32:68,32:68,32:68])
    for i in range(len(x_filef_test)):
        x_file_temp=os.path.join(x_path,x_filef_test[i])
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask
        x_return_test.append(x_temp[32:68,32:68,32:68])
    return  x_return_train,x_return_test

def get_label():
    x_label=pd.read_csv("train_val.csv") ['lable']
    x_train_label=keras.utils.to_categorical(x_label,2)[0:372]
    x_test_label=keras.utils.to_categorical(x_label,2)[373:464]
    return x_train_label,x_test_label

def get_testdataset():
    x_return=[]
    for i in range(117):
        x_file_temp=os.path.join(x_test_path,x_test_file[i])
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask
        x_return.append(x_temp[32:68,32:68,32:68])        
    return x_return

x_train,x_test=get_dataset()
x_train=np.array(x_train)
x_test=np.array(x_test)
x_train_label,x_test_label=get_label()
x_train=x_train.reshape(x_train.shape[0],36,36,36,1)
x_train=x_train.astype('float32')/255
x_test=x_test.reshape(x_test.shape[0],36,36,36,1)
x_test=x_test.astype('float32')/255

x_predict=np.array(get_testdataset())
x_predict=x_predict.reshape(x_predict.shape[0],36,36,36,1)
x_predict=x_predict.astype('float32')/255

# define the model
model = Sequential()
model.add(Conv3D(
    input_shape = (36,36,36,1),
    filters = 32,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    activation = 'relu'))#第一层卷积层
model.add(MaxPooling3D(
    pool_size = 2,
    strides = 2,
    padding = 'same'))#第一层池化层
model.add(Conv3D(64,5,strides = 1,padding='same',activation = 'relu'))#第二层卷积层
model.add(MaxPooling3D(2,2,'same'))#第二层池化层
model.add(Flatten())#拉成1维为全连接做准备
model.add(Dense(1500,activation = 'relu'))
model.add(Dropout(0.5))#Dropout层，用来防止过拟合，但有人说不好用
model.add(Dense(2,activation='softmax'))
adam = Adam(lr=1e-4)#ADAM优化器
# define the object function, optimizer and metrics
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

model.fit(x_train,x_train_label,batch_size=64,epochs=50)
loss,accuracy = model.evaluate(x_train,x_train_label)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (loss,accuracy))
loss,accuracy = model.evaluate(x_test,x_test_label)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (loss,accuracy))
####最终估计的地方
#print(model.predict(x_predict))
model.save("model_file_path.h5")
s = model.predict(x_predict,batch_size = 10)
with open('upload3.csv','r') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(["Id","Predicted"])
    for i in range(117):
           writer = writerow(s[i])
