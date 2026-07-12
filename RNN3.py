from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Embedding, SimpleRNN
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.datasets import imdb 

(x_train,y_train) , (x_test, y_test) = imdb.load_data(num_words = 10000)

print('X TRAIN SHAPE: ',x_train.shape)
print('Y TRAIN SHAPE',y_train.shape)


max_length = 100
x_train = pad_sequences(x_train,max_length)
x_test = pad_sequences(x_test,max_length)

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=32,  input_length=100)),
model.add(SimpleRNN(32)),
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics= ['mae']
)


model.fit(
    x_train,
    y_train,
    epochs = 10,
    batch_size = 500
)

prediction = model.predict(x_test)

loss, mae = model.evaluate(x_test,y_test)

print('loss: ',loss)
print('mae:',mae)