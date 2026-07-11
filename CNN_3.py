from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist


(x_train,y_train) , (x_test,y_test) = mnist.load_data()


# DATA PREPROCESSING

# normalize
x_train = x_train/255.0
x_test = x_test/255.0

# to categories
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# reshape

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)




model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape= (28,28,1))),
model.add(MaxPool2D(2,2)),
model.add(Conv2D(filters=32,kernel_size=(2,2),activation='relu')),
model.add(MaxPool2D(2,2)),
model.add(Flatten()),
model.add(Dense(30,activation='relu')),
model.add(Dense(15,activation='relu')),
model.add(Dense(10,activation='softmax')),

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(
    x_train,
    y_train,
    epochs = 10,
    batch_size = 650
)

loss, accuracy = model.evaluate(x_test,y_test)

print('LOSS: ',loss)
print('accuracy: ',accuracy)

prediction = model.predict(x_test)

print(prediction[0:5])

model.summary()
 