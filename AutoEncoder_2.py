from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

(x_train,_),(x_test,_) = mnist.load_data()

print('X TRAIN SHAPE: ', x_train.shape)
print('X TEST SHAPE: ',x_test.shape)

#normalize the data

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#reshape

x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)

# encoder 

input_layer = Input((784,))
encoder = Dense(128,activation='relu')(input_layer)
encoder = Dense(64, activation='relu')(encoder)
encoder = Dense(32,activation='relu')(encoder)

#decoder

decoder = Dense(64,activation='relu')(encoder)
decoder = Dense(128,activation='relu')(decoder)
output_layer = Dense(784,activation='sigmoid')(decoder)

autoencoder = Model(input_layer,output_layer)

autoencoder.compile(
    optimizer='adam',
    loss = 'binary_crossentropy',
    metrics= ['mse']
)

autoencoder.fit(
    x_train,
    x_train,
    epochs = 10,
    batch_size=500
)

prediction = autoencoder.predict(x_test)

loss, accuracy = autoencoder.evaluate(x_test,x_test)


# TO PLOT DATA
plt.figure(figsize=(10,4))

for i in range(5):

    # Original Image
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap="gray")
    plt.axis("off")

    # Reconstructed Image
    plt.subplot(2,5,i+6)
    plt.imshow(prediction[i].reshape(28,28), cmap="gray")
    plt.axis("off")

plt.show()


