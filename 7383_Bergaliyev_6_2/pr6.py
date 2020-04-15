from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import numpy as np
from var2 import gen_data
import matplotlib.pyplot as plt

def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
def main():
    size = 4000
    itest = size//5
    ivalid = 9*size//25
    itrain = size
    data, labels = gen_data(size)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labels = encoder.transform(labels)
    li = [(data[i], labels[i]) for i in range(0, size)]
    np.random.shuffle(li)
    data = np.array(list(map(lambda x: x[0], li)))
    labels = np.array(list(map(lambda x: x[1], li)))
    data = data.reshape(*data.shape, 1)
    test_data = data[:itest]
    test_labels = labels[:itest]
    valid_data = data[itest:ivalid]
    valid_labels = labels[itest:ivalid]
    train_data = data[ivalid:itrain]
    train_labels = labels[ivalid:itrain]
    model = build_model()
    model.fit(train_data, train_labels, epochs=10, batch_size=10,
              validation_data=(valid_data, valid_labels))
    model.evaluate(test_data, test_labels)


if __name__ == '__main__':
    main()
