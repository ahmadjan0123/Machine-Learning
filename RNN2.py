from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense,Embedding,SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences



(x_train,y_train) , (x_test,y_test) = imdb.load_data(num_words = 10000)

print('X TRAIN SHAPE',x_train.shape)
print('Y TEST SHAPE',y_test.shape)



x_train = pad_sequences(x_train,maxlen=100)
x_test = pad_sequences(x_test,maxlen=100)


model = Sequential([
    Embedding(input_dim=10000, output_dim=32  ,input_length=100),
    SimpleRNN(32),
    Dense(1,activation='sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

model.fit(
    x_train,
    y_train,
    epochs = 15,
    batch_size = 500,
    validation_split = 0.2
)





loss ,  accuracy = model.evaluate(x_test,y_test)

print('LOSS: ',loss)
print('ACCURACY: ',accuracy)

prediction = model.predic(x_test)

print(prediction[0:3])