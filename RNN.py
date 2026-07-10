from tensorflow.keras.layers import Dense,SimpleRNN,Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences



(x_train , y_train)  , (x_test ,  y_test) = imdb.load_data(num_words = 10000)

print('X SHAPE',x_train.shape)
print('Y SHAPE',y_test.shape)

x_train = pad_sequences(x_train, maxlen = 100)
x_test = pad_sequences(x_test, maxlen = 100)

model = Sequential()
model.add(Embedding(input_dim=10000,output_dim=32,input_length=100)),
model.add(SimpleRNN(32)),
model.add(Dense(1,activation='sigmoid'))

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
model.fit(
    x_train,
    y_train,
    epochs = 20,
    batch_size = 400
)

loss , accuracy = model.evaluate(x_test,y_test)
print('LOSS: ',loss)
print('ACCURACY: ',accuracy)

predict = model.predict(x_test)
print(predict[0:10])

