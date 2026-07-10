import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , MaxPooling2D, Conv2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) =  mnist.load_data()

print('shape of x train',x_train.shape)
print('shape of y test',y_test.shape)

#reshape

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

#normalize the image

x_train = x_train/255.0
x_test = x_test/255.0

# to categories
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten()),
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='relu'))

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(
    x_train,
    y_train,
    epochs = 20,
    batch_size = 200,
    validation_split=0.2

)

loss,accuracy = model.evaluate(x_test,y_test)

prediction = model.predict(x_test)

prediction_image = np.argmax(prediction[0])
print("Predicted Digit:", prediction_image)





