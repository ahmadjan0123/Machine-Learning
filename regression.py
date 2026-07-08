from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('concrete_data.csv')


print(df.info())
df.info()

x = df.drop('Strength',axis = 1)
y = df['Strength']

x = (x-x.mean())/x.std()

x_train, x_test ,  y_train , y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state =42
)

print(x.shape)
print(y.shape)

model = Sequential()
model.add(Dense(10, activation='relu',input_shape = (8,)))
model.add(Dense(1))

model.compile( optimizer = 'adam',loss = 'mean_squared_error')

model.fit(
    x_train,
    y_train,
    epochs = 10,
    batch_size = 10,
    verbose =1
)

loss = model.evaluate(
    x_test, y_test
)

print('LOSS: ',loss)

prediction = model.predict(x_test)
print('PREDICTION: ',prediction[:5])


comparission = pd.DataFrame({
    'Actual':y_test.values,
    'Predicted':prediction.flatten()
})
print(comparission)

model.summary()

model.save('concrete_dataset.csv')
