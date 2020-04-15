from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
num_train, width, height, depth = X_train.shape
num_test = X_test.shape[0]
num_classes = np.unique(y_train).shape[0]
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
Y_train = to_categorical(y_train, num_classes)
Y_test = to_categorical(y_test, num_classes)

batch_size = 100
num_epochs = 25
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 64
conv_depth_3 = 128
drop_prob_1 = 0.2
drop_prob_2 = 0.5
hidden_size = 512

def build_model(kernel_size=3, with_dropout=True):
    inp = Input(shape=(width, height, depth))
    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size),
                           padding='same', strides=(1,1), activation='relu')(inp)
    conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size),
                           padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    if with_dropout:
        drop_1 = Dropout(drop_prob_1)(pool_1)
    else:
        drop_1 = pool_1
    conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size),
                           padding='same', strides=(1,1), activation='relu')(drop_1)
    conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size),
                           padding='same', strides=(1,1), activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
    if with_dropout:
        drop_2 = Dropout(drop_prob_1)(pool_2)
    else:
        drop_2 = pool_2
    conv_5 = Convolution2D(conv_depth_3, (kernel_size, kernel_size),
                           padding='same', strides=(1,1), activation='relu')(drop_2)
    conv_6 = Convolution2D(conv_depth_3, (kernel_size, kernel_size),
                           padding='same', strides=(1,1), activation='relu')(conv_5)
    pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_6)
    if with_dropout:
        drop_3 = Dropout(drop_prob_1)(pool_3)
    else:
        drop_3 = pool_3
    flat = Flatten()(drop_3)
    hidden = Dense(hidden_size, activation='relu')(flat)
    if with_dropout:
        drop_4 = Dropout(drop_prob_2)(hidden)
    else:
        drop_4 = hidden
    out = Dense(num_classes, activation='softmax')(drop_4)
    model = Model(inp, out)
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(0.1),
                  metrics=['accuracy'])
    return model

def plot_res(history, label):
    x = range(1, num_epochs+1)
    plt.plot(x, history.history['loss'])
    plt.plot(x, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim(x[0], x[-1])
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(label+'_loss.png')
    plt.clf()
    plt.plot(x, history.history['accuracy'])
    plt.plot(x, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xlim(x[0], x[-1])
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(label+'_acc.png')
    plt.clf()

def test_model():
    model = build_model()
    history = model.fit(X_train, Y_train,
              batch_size=batch_size, epochs=num_epochs,
              verbose=1, validation_split=0.1)
    res = model.evaluate(X_test, Y_test, verbose=0)
    print(res)
    plot_res(history, 'best')

def test_without_dropout():
    model = build_model(with_dropout=False)
    history = model.fit(X_train, Y_train,
              batch_size=batch_size, epochs=num_epochs,
              verbose=1, validation_split=0.1)
    res = model.evaluate(X_test, Y_test, verbose=1)
    print(res)
    plot_res(history, 'without_dropout')

def test_kernel_size():
    for k in [2, 5, 7]:
        model = build_model(k)
        history = model.fit(X_train, Y_train,
              batch_size=batch_size, epochs=num_epochs,
              verbose=1, validation_split=0.1)
        res = model.evaluate(X_test, Y_test, verbose=1)
        print(res)
        plot_res(history, str(k))

def main():
    test_model()
    test_without_dropout()
    test_kernel_size()
    
if __name__ == '__main__':
    main()

