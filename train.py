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
from keras.losses import categorical_crossentropy,binary_crossentropy
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,ModelCheckpoint
from mylib.dataloader import transform
#########��������465 �ֳ�ѵ������������Ϊ4/5 1/5Ϊtest��  ������
x_path='./train/nodule'
x_file=os.listdir(x_path)
x_test_path='./test'

def get_dataset():
    x_return_train=[]
    x_return_test=[]
    x_name=pd.read_csv("train_val.csv") ['name']
    for i in range(len(x_name)):
        x_file_temp=os.path.join(x_path,x_name[i]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask*0.8+x_voxel*0.2
        x_temp=transform.rotation(x_temp,1)
        x_return_train.append(x_temp[34:66,34:66,34:66])
    return  x_return_train,x_return_test

def get_label():
    x_label=pd.read_csv("train_val.csv") ['lable']
    x_train_label=keras.utils.to_categorical(x_label,2)
    return x_train_label

def get_testdataset():
    x_return=[]
    x_name=pd.read_csv("submit.csv") ['Id']
    for i in range(117):
        x_file_temp=os.path.join(x_test_path,x_name[i]+'.npz')
        x_voxel=np.array(np.load(x_file_temp)['voxel'])
        x_mask=np.array(np.load(x_file_temp)['seg'])
        x_temp=x_voxel*x_mask*0.8+x_voxel*0.2
        x_temp=transform.rotation(x_temp,1)
        x_return.append(x_temp[34:66,34:66,34:66])        
    return x_return
    
def mixup_data(x1, y1,alpha,n):
    x2=np.zeros(np.shape(x1))
    y2=np.zeros(np.shape(y1),'float')
    x3=np.zeros(n)
    y3=np.zeros(n,'float')
    l=len(x1)
    indexs = np.random.randint(0, l, n)
    indexs2 = np.random.randint(0, l, n)
    for i in range(n):
        x2[i] = x1[indexs2[i]]*alpha+(1-alpha)*x1[indexs[i]]
        y2[i] = y1[indexs2[i]]*alpha+(1-alpha)*y1[indexs[i]]
        
    x3 = x2[:n]
    y3 = y2[:n]
    return x3, y3

densenet_depth =25
densenet_growth_rate = 30

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
early_stopping = EarlyStopping(monitor='val_loss', patience=50)
Checkpointer = ModelCheckpoint(filepath='result/%s/weights.{epoch:02d}.h5' % '20191225', verbose=1,
                               period=3, save_weights_only=False)
nb_classes = 2
batch_size = 32
model = createDenseNet(nb_classes=nb_classes,img_dim=[32,32,32,1],depth=densenet_depth,growth_rate = densenet_growth_rate)
model.compile(loss=binary_crossentropy,optimizer=SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True), metrics=['accuracy'])
model.summary()  # print the model

model.fit(x_train, x_train_label,batch_size=16, epochs=150,validation_split=0.20, verbose=2, shuffle=False, callbacks=[early_stopping,Checkpointer])
loss,accuracy = model.evaluate(x_train,x_train_label)
print('Training loss: %.4f, Training accuracy: %.2f%%' % (loss,accuracy))
loss,accuracy = model.evaluate(x_test,x_test_label)
print('Testing loss: %.4f, Testing accuracy: %.2f%%' % (loss,accuracy))
####���չ��Ƶĵط�
#print(model.predict(x_predict))
model.save("weight.final.h5")

score1=model.predict(x_predict)
print(score1)
score=score1[:,1].T
file_name = 'test1221.csv'
save = pd.DataFrame(score)
save.to_csv(file_name)
