import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(' x train shape: ',x_train.shape)
print(' y train shape: ',y_train.shape)


#normalize the data
x_train = x_train/255.0
x_test = x_test/255.0

# reshape the data

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

# to categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential(
    [

        
        Conv2D(filters = 32, kernel_size=(3,3),activation='relu',input_shape = (28,28,1)),
        MaxPool2D((2,2)),
        Conv2D(filters=64, kernel_size=(3,3),activation='relu'),
        MaxPool2D((2,2)),
        Flatten(),
        Dense(50,activation = 'relu'),
        Dense(10,activation = 'softmax')
    ]
)


model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(
    x_train,
    y_train,
    epochs = 10,
    batch_size = 200
)


loss ,  accuracy = model.evaluate(x_test,y_test)


prediction = model.predict(x_test)

predict_img = np.argmax(prediction[0])
print(predict_img)




