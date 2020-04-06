import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
import pandas as pd

n = 10000
X = np.random.normal(-5, 10, n)
e = np.random.normal(0, 0.3, n)
data = np.array([-X**3, np.log(np.abs(X)),
        np.exp(X), X+4, -X+np.sqrt(np.abs(X)), X])

data += e
data = np.transpose(data)
data = data - data.mean(axis=1, keepdims=True)
data = data / data.std(axis=1, keepdims=True)
print(X[0], data[0], sep='\n')
target = np.sin(X*3) + e
print(target.mean())
print(target.std())
target_test = target[:n//5]
target_train = target[n//5:]

x_test = data[:n//5]
print(x_test[0])
x_train = data[n//5:]

encoding_dim = 3

input_layer = Input(shape=(6,))

dense = Dense(100, activation='relu')(input_layer)

encoded = Dense(encoding_dim, activation='tanh')(dense)

hidden = Dense(100, activation='relu')(encoded)

decoded = Dense(6, activation='linear', name='decoder')(hidden)

k = 100

hidden = Dense(k, activation='relu')(encoded)

for i in range(20):
    hidden = Dense(k, activation='tanh')(hidden)
    
hidden1 = Dense(100, activation='relu')(encoded)

hidden1 = Dense(100, activation='relu')(hidden1)

hidden1 = Dense(100, activation='relu')(hidden1)

regr = Dense(1, activation='tanh')(hidden)

regr = Dense(1, activation='linear', name='regr')(concatenate([regr, hidden1]))

autoencoder = Model(input_layer, decoded)

encoder = Model(input_layer, encoded)

encoded_input = Input(shape=(encoding_dim,))

decoder_layer = autoencoder.layers[-1]


regr_model = Model(input_layer, (regr, decoded))
regr_model.compile(optimizer='adam', loss='mse')

n_epochs = 500 
history = regr_model.fit(x_train, [target_train, x_train],
                epochs=n_epochs,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, [target_test, x_test]))

x = range(1, n_epochs+1)
plt.plot(x, history.history['decoder_loss'])
plt.plot(x, history.history['val_decoder_loss'])
plt.title('Model decoder_loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(x[0], x[-1])
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(x, history.history['regr_loss'])
plt.plot(x, history.history['val_regr_loss'])
plt.title('Model regr_loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.xlim(x[0], x[-1])
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

df = pd.DataFrame(data)

df.to_csv('dataset.csv', index = False, header=False)

rm = Model(input_layer, regr)

rm.save('regr.h5')

y = rm.predict(data)

encoder = Model(input_layer, encoded)

encoder.save('encoder.h5')

encoded = encoder.predict(data)

df = pd.DataFrame(encoded)

df.to_csv('encoded.csv', index = False, header=False)

decoder = Model(input_layer, decoded)

decoder.save('decoder.h5')

decoded = decoder.predict(data)

df = pd.DataFrame(decoded)

df.to_csv('decoded.csv', index = False, header=False)
          
regression = Model(input_layer, regr)

target = target.reshape((target.shape[0], 1))
regr_res = regression.predict(data).reshape(target.shape)

df = pd.DataFrame(np.concatenate([regr_res, target], axis=1))

df.to_csv('regr.csv', index = False, header=['predicted', 'target'])
