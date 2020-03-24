import numpy as np
from math import exp 
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import boston_housing

def element_wise_predict(input_data, weights):
    relu = lambda x: max(x, 0)
    sigmoid = lambda x: 1/(1+exp(-x))
    act = [relu for _ in weights]
    act[-1] = sigmoid
    inp = input_data.copy()
    for d in range(0, len(weights)): 
        res = np.zeros((inp.shape[0], weights[d][0].shape[1]))
        for i in range(0, inp.shape[0]):
            for j in range(0, weights[d][0].shape[1]):
                s = 0
                for k in range(0, inp.shape[1]):
                    s += inp[i][k] * weights[d][0][k][j]
                res[i][j] = act[d](s + weights[d][1][j])
        inp = res
    return res

def tensor_predict(input_data, weights):
    relu = lambda x: np.maximum(x, 0)
    sigmoid = lambda x: 1/(1+np.exp(-x))
    act = [relu for _ in weights]
    act[-1] = sigmoid
    res = input_data.copy()
    for d in range(0, len(weights)): 
        res = act[d](np.dot(res, weights[d][0]) + weights[d][1])
    return res

def print_predicts(model, dataset):
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())
    element_wise_res = element_wise_predict(dataset, weights)
    tensor_res = tensor_predict(dataset, weights)
    model_res = model.predict(dataset)
    assert np.isclose(element_wise_res, model_res).all()
    assert np.isclose(tensor_res, model_res).all()
    print("Результат поэлементного вычисления:")
    print(element_wise_res)
    print("Результат тензорного вычисления:")
    print(tensor_res)
    print("Результат прогона через обученную модель:")
    print(model_res)
    
def logic_func(a, b, c):
    return (a or b) != (not(b and c))

def main():
    train_data = np.array([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 0],
                           [0, 1, 1],
                           [1, 0, 0],
                           [1, 0, 1],
                           [1, 1, 0],
                           [1, 1, 1]])
    train_target = np.array([int(logic_func(*x)) for x in train_data])
    model = Sequential()
    model.add(Dense(5, activation='relu', input_shape=(3,)))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

    print_predicts(model, train_data)
    
    model.fit(train_data, train_target, epochs=150, batch_size=1)

    print_predicts(model, train_data)
    

if __name__ == '__main__':
    main()
