from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Conv2D , MaxPooling2D , Flatten
from tensorflow.keras. datasets import mnist
from tensorflow.keras.utils import to_categorical
(x_train, y_train) , (x_test,y_test) = mnist.load_data()

print('X Train shape',x_train.shape)
print('X Test Shape',x_test.shape)



x_train= x_train/255.0
x_test = x_test/255.0



y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)





model = Sequential(
    [
        Conv2D(filters = 32, kernel_size=(3,3),activation='relu',input_shape = (28,28,1)),
        MaxPooling2D(3,3),
        Conv2D(filters=32, kernel_size=(3,3),activation='relu'),
        MaxPooling2D(3,3),
        Flatten(),
        Dense(32,activation='relu'),
        Dense(10,activation='softmax')
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
    batch_size = 500
)


prediction = model.predict(x_test)
print(prediction[0:5])
loss , accuracy = model.evaluate(x_test,y_test)

print('LOSS: ',loss)
print('Accuracy: ',accuracy)

