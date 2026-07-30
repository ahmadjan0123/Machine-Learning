from tensorflow.keras.layers import Dense, Conv2D , Flatten ,MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# load data
(x_train , y_train ), (x_test,y_test) = mnist.load_data()

#normalize the data
x_train = x_train/255.0
x_test = x_test/255.0

# to categorical 
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

# reshape
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)


# model
model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape= (28,28,1)))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))


model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy','f1_score']
)

model.fit(
    x_train,
    y_train,
    batch_size = 650,
    epochs = 5
)

loss , acc , f1 = model.evaluate(x_test,y_test)
print('LOSS: ',loss)
print('Accuracy: ',acc)
print('F1 Score: ',f1)


predict = model.predict(x_test)

print(predict[0:5])

model.summary()