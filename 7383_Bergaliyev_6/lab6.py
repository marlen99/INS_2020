import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

def vectorize(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def loadData(dimension):
    (training_data, training_targets),(testing_data, testing_targets) = imdb.load_data(num_words=dimension)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)

    data = vectorize(data, dimension)
    targets = np.array(targets).astype("float32")

    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]
    return (train_x, train_y, test_x, test_y)

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(50, activation = "relu"))
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation = "relu"))
    model.add(layers.Dense(1, activation = "sigmoid"))
    model.compile(optimizer = "adam", loss = "binary_crossentropy",
                  metrics = ["accuracy"])
    return model

def plot_bars(res_train, res_test, dimensions):
    plt.bar(np.arange(len(dimensions)) * 3, res_train, width=1)
    plt.bar(np.arange(len(dimensions)) * 3 + 1, res_test, width=1)
    plt.xticks([3*i + 0.5 for i in range(0,len(dimensions))],
               labels=[dimension for dimension in dimensions])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('dimensions')
    plt.legend(['Train', 'Test'])
    plt.savefig('bars.png')
    plt.clf()
    
def test_model(dimension):
    train_x, train_y, test_x, test_y = loadData(dimension)
    model = build_model()
    results = model.fit(train_x, train_y, epochs=2, batch_size=500,
                        validation_data=(test_x, test_y))
    return (results.history['accuracy'][-1], results.history['val_accuracy'][-1])

def test_dimensions(dimensions):
    res_train = []
    res_test = []
    for dimension in dimensions:
        train, test = test_model(dimension)
        res_train.append(train)
        res_test.append(test)
    plot_bars(res_train, res_test, dimensions)

def testOwnText(text):
    train_x, train_y, test_x, test_y = loadData(10000)
    model = build_model()
    results = model.fit(train_x, train_y, epochs=2, batch_size=500,
                        validation_data=(test_x, test_y))
    text = vectorize([text])
    res = model.predict(text)
    print(res)

def loadText(filename):
    punctuation = ['.',',',':',';','!','?','(',')']
    text = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            text += [s.strip(''.join(punctuation)).lower() for s in line.strip().split()]
    print(text)
    indexes = imdb.get_word_index()
    encoded = []
    for w in text:
        if w in indexes and indexes[w] < 10000:
            encoded.append(indexes[w])
    return np.array(encoded)

if __name__ == '__main__':
    filename = 'text.txt'
    text = loadText(filename)
    testOwnText(text)
