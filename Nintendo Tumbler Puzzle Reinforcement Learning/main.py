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
    print("Data Synthesis")

    counter = [table[1], table[0], table[3], table[2], table[4]]
    history = [Tumbler()] * (data_size + 1)
    history[0] = state

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
                            break

def Import(fstream):
    print("Importing Data")

    global epoch, model
    epoch = 0
    if fstream == 'buffer.json':
        log = open('log.txt', 'r').read().split()
        epoch = int(log[-log[::-1].index("epoch:")])
        # loads the weights. automatically compiles the model. 
        model = keras.models.load_model('model.h5')
    
    data = json.load(open(fstream, 'r'))
    for i, (key, val) in enumerate(data.items()):
        X[i] = np.array(json.loads(key))
        Y[i] = val

def Export(fstream):
    print("Exporting Data")
    
    if fstream == 'buffer.json':
        text = f"epoch: {epoch}\n"
        print(text)
        open('log.txt', 'a').write(text)
        # save() saves the weights, model architecture, training configuration, and optimizer to a HDF5. 
        # save_weights() only saves the weights to a HDF5. weights can be applied to another model architecture. 
        model.save('model.h5')

    JSON = dict()
    for i in range(data_size):
        JSON[np.array2string(X[i], separator = ", ", max_line_width = 1_000)] = float(Y[i])
    json.dump(JSON, open(fstream, 'w'), indent = 4)

Time = time.time()
table = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
state = Tumbler([[1, 2, 3, 4, 5], 
                 [1, 2, 3, 4, 5]], 
                [[1, 2, 3, 4, 5], 
                 [1, 2, 3, 4, 5]], 
                [[0,    0,    0]])

discount = 0.99
data_size = 10_000
shape = state.scrub_all().shape[1]
X = np.zeros([data_size, shape], dtype = np.int8)
Y = np.zeros([data_size], dtype = np.float32)

#Import('buffer.json')
#Synthesize()
#Export('data.json')
Import('data.json')

model = keras.Sequential([
        keras.layers.Dense(81, activation = 'relu',
                            input_shape = [shape]),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(49, activation = 'relu'),
        keras.layers.Dense(36, activation = 'relu'),
        keras.layers.Dense(25, activation = 'relu'),
        keras.layers.Dense(16, activation = 'relu'),
        keras.layers.Dense(9, activation = 'relu'),
        keras.layers.Dense(4, activation = 'relu'),
        keras.layers.Dense(1)])
model.compile(optimizer = 'adam', loss = 'mse')
model.summary()
model.fit(X, Y, batch_size = 64, epochs = 100, verbose = 0)

state.move(table[4])
value = np.array([0, 0, 0, 0, 1])
print("start program")
for epoch in range(epoch, 1_000):
    Export('buffer.json')

    accuracy = 0
    for i in range(100, data_size):
        # simulate environment
        reward = state.reward
        action = table[value.argmax() if random.randrange(0, 100) < min(99, epoch * 10) else random.randrange(0, 5)]
        X[i] = state.scrub(action)
        state.move(action)

        # replay buffer
        if state.reward == 1_000:
            Y[i] = reward + discount * 1_000
            for temp in range(50):
                state.move(table[random.randrange(0, 5)])

            value = model.predict(state.scrub_all(), verbose = 0)
            accuracy += 1
        else:
            value = model.predict(state.scrub_all(), verbose = 0)
            Y[i] = reward + discount * np.amax(value)

        # train model
        if not (i + 1) % 100:
            Qnew = keras.models.clone_model(model)
            Qnew.compile(optimizer = 'adam', loss = 'mse')
            loss = Qnew.fit(X, Y, batch_size = 64, epochs = 100, verbose = 0).history['loss'][-1]
            open('log.txt', 'a').write(f"reward: {reward} value {Y[i]} loss: {loss}\n")
            print(f"loss: {'x' * min(100, int(loss // 4))}")
            model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))
    print("accuracy: ", accuracy * 5)