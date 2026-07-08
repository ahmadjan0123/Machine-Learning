import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

(x_train,y_train), (x_test,y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)

#resize the input
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

#scale- normalize
x_train = x_train/250

x_test =  x_test/250

#change the laels to categorical values

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(20,activation='relu',input_shape = (784,)))  #first hidden layer
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(
    x_train,
    y_train,
    epochs = 3 0,
    batch_size = 20,
    validation_split=0.2

)

predict = model.predict(x_test)
loss, accuracy = model.evaluate(x_test,y_test)

print('LOSS: ',loss)
print('accuracy:',accuracy)

