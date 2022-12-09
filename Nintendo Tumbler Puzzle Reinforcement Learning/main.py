import numpy as np
from tensorflow import keras
import random
import copy
from Tumbler import Tumbler

# https://www.geeksforgeeks.org/implementing-neural-networks-using-tensorflow/

def expand_horizon(terminal, start, stop):
  counter = [table[1], table[0], table[3], table[2], table[4]]
  history = np.full([stop - start + 1], Tumbler(), dtype = Tumbler)
  history[0] = terminal

  for backup in range(stop - start):
    for temp in range(5):
      if start == stop:
        return
      offspring = copy.deepcopy(history[backup])
      offspring.move(counter[temp])
      if not offspring in history:
        X[start] = offspring.scrub(table[temp])
        offspring.reward += discount * history[backup].reward
        y[start] = offspring.reward
        history[backup + temp + 1] = offspring
        start += 1
        if not start % 100:
          print("history: ", start)

table = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
state = Tumbler([[-2, 1, 0, 1, 2], [-2, 1, 0, 1, 2]], [[-2, 1, 0, 1, 2], [-2, 1, 0, 1, 2]], [0, 0, 0]) # make tumbler one-hot encoded values

discount = 0.99
data_size = 10_000
shape = np.shape(state.scrub_all())[1]
X = np.empty([data_size, shape], dtype = np.int8)
y = np.empty([data_size], dtype = np.float16)

print("prefabricating data")
expand_horizon(state, 0, data_size)
state.move(table[4])
value = np.array(table[4])

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
model.fit(X, y, batch_size = 64, epochs = 200, verbose = 0)

i = 0
# f = open("text.txt", "r")
# i = f.read()
# f.close()
# mode = keras.load_model('model.h5')
while(True):
  print("epoch: ", i)
  if not i % 10:
    model.save('model.h5')
    f = open("text.txt", "w")
    f.write(str(i))
    f.close()

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
      y[index] = reward + discount * state.reward
    else:
      value = model.predict(state.scrub_all(), verbose = 0)
      y[index] = reward + discount * np.amax(value)

    # train model
    Qnew = keras.models.clone_model(model)
    Qnew.compile(optimizer = 'adam', loss = 'mse')
    print("loss: ", Qnew.fit(X, y, batch_size = 256, epochs = 400, verbose = 0).history['loss'][-1])
    model.set_weights(0.9 * np.array(model.get_weights(), dtype = object) + 0.1 * np.array(Qnew.get_weights(), dtype = object))
  print("accuracy: ", accuracy * 5)