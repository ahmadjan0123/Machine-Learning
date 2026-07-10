from tensorflow.keras.layers import Input , Dense 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

(x_train,_), (x_test,_) = mnist.load_data()

print('X Train Shape',x_train.shape)
print('Y Train Shape',x_test.shape)

# NORMALIZE

x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0

#reshape

x_train = x_train.reshape(-1,784)
x_test = x_test.reshape(-1,784)

# Encoder
input_layer = Input(784, )
encoded = Dense(128,activation='relu')(input_layer)
encoded = Dense(64,activation='relu')(encoded)
encoded = Dense(32,activation='relu')(encoded)

# Decoder

decoded = Dense(64,activation='relu')(encoded)
decoded = Dense(128,activation='relu')(decoded)
output_layer = Dense(784,activation='sigmoid')(decoded)

autoencoder = Model(input_layer,output_layer)

autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

autoencoder.fit(
    x_train,
    x_test,
    epochs = 20,
    batch_size=500,
    validation_data=(x_train,x_test)
)



prediction = autoencoder.predict(x_test)
print(prediction[0:5])

plt.figure(figsize=(10,4))

for i in range(5):

    # Original Image
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap="gray")
    plt.axis("off")

    # Reconstructed Image
    plt.subplot(2,5,i+6)
    plt.imshow(decoded_images[i].reshape(28,28), cmap="gray")
    plt.axis("off")

plt.show()


autoencoder.summary()





