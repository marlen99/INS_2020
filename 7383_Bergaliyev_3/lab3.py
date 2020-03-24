import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt
import pylab

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def _test(epochs):
    loss = []
    mae = []
    v_loss = []
    v_mae = []
    model = build_model()
    history = model.fit(train_data, train_targets, epochs=epochs,
                        validation_data=(test_data, test_targets), batch_size=1)
    loss = history.history['loss']
    mae = history.history['mae']
    v_loss = history.history['val_loss']
    v_mae = history.history['val_mae']
    x = range(1, epochs+1)

    plt.plot(x, loss)
    plt.plot(x, v_loss)
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('number of epochs')
    plt.xlim(x[19], x[-1])
    plt.ylim(min(loss[19:]), max(v_loss[19:]))
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    plt.plot(x, mae)
    plt.plot(x, v_mae)
    plt.title('Model mean absolute error')
    plt.ylabel('mean absolute error')
    plt.xlabel('number of epochs')
    plt.xlim(x[19], x[-1])
    plt.ylim(min(mae[19:]), max(v_mae[19:]))
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    

def cross_check(k_list):
    k = 4
    all_scores = []
    for k in k_list:
        num_val_samples = len(train_data) // k
        all_scores.append([])
        for i in range(0, k):
            print('processing fold #', i)
            val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
            partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                                axis=0)
            partial_train_targets = np.concatenate(
                [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
            model = best_model()
            model.fit(partial_train_data, partial_train_targets, epochs=100, batch_size=1, verbose=0)
            val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
            all_scores[-1].append(val_mae)            
        print(np.mean(all_scores[-1]))

    plt.plot(k_list, [np.mean(i) for i in all_scores])
    plt.title('Model mean absolute error')
    plt.ylabel('mean absolute error')
    plt.xlabel('K')
    plt.xlim(k_list[0], k_list[-1])
    pylab.xticks(k_list)
    plt.show()

def main():
    _test(400)
    cross_check(range(2, 7, 1))
    best_model()
    
if __name__ == '__main__':
    main()
