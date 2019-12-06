import keras
from keras.datasets import cifar10
from keras.layers import Dense,Flatten
from keras.models import Model
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.applications.resnet import ResNet50
from keras.callbacks import TensorBoard
import os
import datetime
import numpy as np


logDir = os.path.join("logs","fit","resnet",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
callback = TensorBoard(log_dir=logDir)

(trainX,trainY),(testX,testY) = cifar10.load_data()

labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
trainY = to_categorical(trainY)
testY = to_categorical(testY)
print(trainY.shape)
trainX = trainX/255
testX = testX/255

print(trainX.shape)

model = ResNet50(input_shape = (32,32,3,),include_top = False)

for layer in model.layers :
  layer.trainable = False
vggOutput =  model.layers[-2]
x = Flatten()(vggOutput.output)
x = Dense(256,activation = "relu",input_shape=(512,))(x)
x = Dense(10,activation = "sigmoid")(x)

model = Model(model.inputs,x)

model.summary()

model.compile(loss = "categorical_crossentropy",optimizer="adam",metrics = ["accuracy"])

model.fit(trainX,trainY,epochs = 20,validation_data=(testX,testY),batch_size=128,callbacks = [callback])
model.save_weights("weightsResnet.h5")

