import numpy as np
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from keras.constraints import max_norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

xdata = np.load("raw_x.npy")
ydata = np.loadtxt("raw_y.dat")

xdata = xdata.reshape(100,256,256,1)

def create_model():
    model = Sequential()
    model.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(256,256,1)))
    #model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(Conv2D(32,kernel_size=3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model

kf = KFold(n_splits=5,shuffle=True)

shits = []
for train,test in kf.split(xdata):
    model = create_model()
    train_inputs = xdata[train]
    train_outputs = ydata[train]
    test_inputs = xdata[test]
    test_outputs = ydata[test]
    #X_scaler = StandardScaler()
    #x = X_scaler.fit_transform(train_inputs)
    #Y_scaler = StandardScaler()
    #y = Y_scaler.fit_transform(train_outputs)
    model.fit(train_inputs,train_outputs,epochs=5,batch_size=5,verbose=1)
    #x2 = X_scaler.transform(test_inputs)
    #y2 = Y_scaler.transform(test_outputs)
    score = model.evaluate(test_inputs,test_outputs,batch_size=1)
    shits.append(score)

print shits
