import copy
import json
from tensorflow import keras
import numpy as np
import random
import time
from Tumbler import Tumbler

# https://www.geeksforgeeks.org/implementing-neural-networks-using-tensorflow/
# lists are not type sensitive. numpy arrays are type sensitive. 
# print(model.predict(np.array([state.scrub(table[4])])))
# print(model.predict(state.scrub_all()))
# print(model.predict(X[0:1]))

def Synthesize():
    print("Synthesizing Data")

    counter = [table[1], table[0], table[3], table[2], table[4]]
    history = [Tumbler()] * (data_size + 1)
    history[0] = Tumbler([[1, 2, 3, 4, 5], 
                          [1, 2, 3, 4, 5]], 
                         [[1, 2, 3, 4, 5], 
                          [1, 2, 3, 4, 5]], 
                         [[0,    0,    0]])
    i = 0

    for elem in history:
        for action in range(5):
            temp = copy.deepcopy(elem)
            temp.move(counter[action])

            if temp.reward != 100:
                temp.reward += discount * elem.reward

                if not temp in history:
                    X[i] = temp.scrub(table[action])
                    Y[i] = temp.reward
                    i += 1
                    history[i] = temp

                    if not i % 100:
                        print(Y[i - 100:i])
                        if i == data_size:
                            Save('data.json')
                            return

def Load(fstream):
    with open('debugger.txt', 'a') as debugger:
        debugger.write("Loading Data\n")
    global epoch, i, lim, model

    data = json.load(open(fstream, 'r'))
    for _i, (key, val) in enumerate(data.items()):
        X[_i] = np.array(json.loads(key))
        Y[_i] = val
    for it, _i in enumerate(range(_i + 1, data_size)):
        X[_i] = X[it]
        Y[_i] = Y[it]

    if fstream == 'data.json':
        model.compile(optimizer = 'adam', loss = 'mse')
        model.fit(X, Y, batch_size = 64, epochs = 300, verbose = 0)
        return

    f = open('log.txt', 'r')
    log = f.read().split()
    f.close()

    # loads the weights. automatically compiles the model. 
    model = keras.models.load_model('model.h5')
    index = -log[::-1].index("epoch:")
    epoch = int(log[index])
    i = lim = (epoch - 1) * cluster_size % data_size

    f = open('log.txt', 'w')
    f.write(''.join([elem + ('\n' if elem.find(':') == -1 else ' ') for elem in log[:index - 1]]))
    f.close()

def Save(fstream):
    with open('debugger.txt', 'a') as debugger:
        debugger.write("Saving Data\n")
    
        JSON = dict(zip([repr(elem.tolist()) for elem in X], Y))
        json.dump(JSON, open(fstream, 'w'), indent = 4)

        if fstream == 'data.json':
            return

        # save() saves the weights, model architecture, training configuration, and optimizer to a HDF5. 
        # save_weights() only saves the weights to a HDF5. weights can be applied to another model architecture. 
        model.save('model.h5')
        text = f"epoch: {epoch} time: {time.time() - Time}\n"
        open('log.txt', 'a').write(text)
        debugger.write(text)

def Clear():
    open('log.txt', 'w').close()
    open('buffer.json', 'w').close()

open('debugger.txt', 'w').close()
Time = time.time()
epoch = 1
i = lim = 0
table = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]

discount = 0.9
data_size = 50_000
cluster_size = 1_000
shape = Tumbler().scrub_all().shape[1]
X = np.zeros([data_size, shape], dtype = np.int8)
Y = np.zeros([data_size], dtype = np.float64)

model = keras.Sequential([
        keras.layers.Dense(125, activation = 'relu',
                            input_shape = [shape]),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(125, activation = 'relu'),
        keras.layers.Dense(25, activation = 'relu'),
        keras.layers.Dense(5, activation = 'relu'),
        keras.layers.Dense(1)])
model.summary()

#Synthesize()
#Load('data.json')
#Clear()
Load('buffer.json')

with open('debugger.txt', 'a') as debugger:
    debugger.write("start program\n")
for epoch in range(epoch, 1_000):
    Save('buffer.json')

    if i == data_size:
        i = lim = 0
    lim += cluster_size
    accuracy = 0
    while i < lim:
        # simulate environment
        state = Tumbler([[1, 2, 3, 4, 5], 
                         [1, 2, 3, 4, 5]], 
                        [[1, 2, 3, 4, 5], 
                         [1, 2, 3, 4, 5]], 
                        [[0,    0,    0]])
        while state.reward == 100:
            for temp in range(min(epoch, 50)):
                state.move(table[random.randrange(0, 5)])

        # replay buffer
        value = model.predict(state.scrub_all(), verbose = 0)
        for temp in range(min(epoch, 50, data_size - i)):
            reward = state.reward
            action = table[value.argmax() if random.randrange(0, 100) < 95 else random.randrange(0, 5)]
            X[i] = state.scrub(action)
            state.move(action)

            if state.reward == 100:
                accuracy += 1
                Y[i] = reward + discount * state.reward
            else:
                value = model.predict(state.scrub_all(), verbose = 0)
                Y[i] = reward + discount * value.max()

            # train model
            i += 1
            if not i % 100:
                Qnew = keras.models.clone_model(model)
                Qnew.compile(optimizer = 'adam', loss = 'mse')
                loss = Qnew.fit(X, Y, batch_size = 64, epochs = 300, verbose = 0).history['loss'][-1]
                model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))
                
                text = f"loss: {loss}\n"
                open('log.txt', 'a').write(text)
                with open('debugger.txt', 'a') as debugger:
                    debugger.write(text)

            if state.reward == 100:
                break

    with open('debugger.txt', 'a') as debugger:
        debugger.write(f"accuracy: {accuracy * 100 / cluster_size}")