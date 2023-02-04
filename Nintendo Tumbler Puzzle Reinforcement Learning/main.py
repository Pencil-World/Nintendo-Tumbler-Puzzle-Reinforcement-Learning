import copy
from tensorflow import keras
import numpy as np
import random
from Tumbler import Tumbler

# https://www.geeksforgeeks.org/implementing-neural-networks-using-tensorflow/

# def Import():
#     f = open("text.txt", "r")
#     i = f.read()
#     mode = keras.load_model('model.h5')
    
#     data = open("data.txt", "r")
#     for index in range(data_size):
#         X[index] = offspring.scrub(table[temp])
#         y[index] = offspring.reward

def create_data():
    counter = [table[1], table[0], table[3], table[2], table[4]]
    history = np.full([data_size + 1], Tumbler(), dtype = Tumbler)
    history[0] = copy.deepcopy(state)
    i = 0

    for elem in history:
        print(elem)
        for action in range(5):
            temp = copy.deepcopy(elem)
            temp.move(counter[action])
            if not temp in history:
                X[i] = temp.scrub(table[action])
                print(temp.reward)
                temp.reward += discount * elem.reward
                Y[i] = temp.reward
                history[i + 1] = temp
                i += 1
                if not i % 100:
                    print(Y[i - 100:i])
                    if i == data_size:
                        return

table = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
state = Tumbler([[1, 2, 3, 4, 5], 
                 [1, 2, 3, 4, 5]], 
                [[1, 2, 3, 4, 5], 
                 [1, 2, 3, 4, 5]], 
                [[0,    0,    0]])

discount = 0.99
data_size = 10_000
shape = np.shape(state.scrub_all())[1]
X = np.empty([data_size * 2, shape], dtype = np.int8)
Y = np.empty([data_size * 2], dtype = np.float16)

print("prefabricating data")
create_data()
state.move(table[4])
value = [0, 0, 0, 0, 1]

sys.exit()
np.set_printoptions(threshold=sys.maxsize)

# model = keras.Sequential([
#         keras.layers.Dense(81, activation = 'relu',
#                             input_shape = [shape]),
#         keras.layers.Dense(64, activation = 'relu'),
#         keras.layers.Dense(49, activation = 'relu'),
#         keras.layers.Dense(36, activation = 'relu'),
#         keras.layers.Dense(25, activation = 'relu'),
#         keras.layers.Dense(16, activation = 'relu'),
#         keras.layers.Dense(9, activation = 'relu'),
#         keras.layers.Dense(4, activation = 'relu'),
#         keras.layers.Dense(1)])
model = keras.Sequential([
        keras.layers.Dense(10, activation = 'relu',
                            input_shape = [shape]),
        keras.layers.Dense(1)])
model.compile(optimizer = keras.optimizers.Adam(learning_rate=1e-3), loss = 'mse')
model.summary()
print(model.predict(np.zeros([1, 144])))
print(model.predict(np.array([state.scrub(table[4])])))
print(model.predict(state.scrub_all()))
print(model.predict(np.array([X[0]])))

sys.exit()
model.fit(X, Y, batch_size = 64, epochs = 10)
i = 0
print("start program")
while(True):
    print("epoch: ", i)
    #   model.save('model.h5')
    #   f = open("text.txt", "w")
    #   f.write(str(i))
    #   f.close()

    # generate data
    accuracy = 0
    for index in range(data_size):
        reward = state.reward
        action = table[value.argmax() if random.randrange(0, 100) < 95 else random.randrange(0, 5)]
        X[index] = state.scrub(action)
    
        state.move(action)
        if state.reward == 1_000:
            for temp in range(50):
                state.move(table[random.randrange(0, 5)])
            accuracy += 1
            value = model.predict(state.scrub_all(), verbose = 0)
            Y[index] = reward + discount * state.reward
        else:
            value = model.predict(state.scrub_all(), verbose = 0)
            Y[index] = reward + discount * np.amax(value)

        # train model
        Qnew = keras.models.clone_model(model)
        Qnew.compile(optimizer = 'adam', loss = 'mse')
        print("loss: ", Qnew.fit(X, Y, batch_size = 64, epochs = 200, verbose = 0).history['loss'][-1])
        model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))
    print("accuracy: ", accuracy * 5)