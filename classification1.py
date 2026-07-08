import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()



x = data.data
y = data.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train , y_test = train_test_split(
    x,
    y,
    random_state = 42,
    test_size = 0.2
)

scalar = StandardScaler()

x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

model = Sequential()

model.add(Dense(10,activation='relu',input_shape = (30,)))
model.add(Dense(10,activation='relu',))
model.add(Dense(1,activation='sigmoid'))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

model.fit(
    x_train,
    y_train,
    epochs = 20,
    batch_size = 10,
    validation_split=0.2

)

loss, accuracy = model.evaluate(x_test,y_test)

predict = model.predict(x_test)

print('accuracy',accuracy)
print('loss',loss)

print(predict[0:5])

predicited_class = (predict>0.5)