from __future__ import print_function

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
import warnings


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

data=scio.loadmat('D:/数据集/text/de_LDS3.mat')

del data['__header__']
del data['__version__']
del data['__globals__']

text_data=[]
for num in range(0,10):
    for k in data.keys():
        a_train=data[k][3:5,:150,2].tolist()[0]
        b_train=data[k][3:5,:150,2].tolist()[1]
        a_train.extend(b_train)
        text_data.extend(a_train)

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)

a=np.asanyarray(text_data)
a=a.reshape(1,-1)
x=a.reshape(150,300)
y=np.array([[1,0,0,0,0,1,0,0,1,1,0,0,0,1,0],[0,1,0,0,1,0,0,1,0,0,1,0,1,0,0],[0,0,1,1,0,0,1,0,0,0,0,1,0,0,1]])
y=y.transpose()
m=y
for num in range(1,10):
    y=np.vstack((y,m))

print(x.shape)
print(y.shape)
            
batch_size = 10
num_classes = 3
epochs = 100
x= x.astype('float32')
print(x.shape[0], 'train samples')

model = Sequential()
model.add(Dense(512, activation='sigmoid', input_shape=(300,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x,y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x, y))

score = model.evaluate(x, y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

