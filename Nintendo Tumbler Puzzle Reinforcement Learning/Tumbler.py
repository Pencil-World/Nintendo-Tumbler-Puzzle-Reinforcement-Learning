import numpy as np

class Tumbler:
  def __init__(self, upper = None, lower = None, hidden = None, isUp = False):
    self.upper = np.array(upper)
    self.lower = np.array(lower)
    self.hidden = np.array(hidden)
    self.isUp = isUp # state of hidden. can be above or below the main 2 barrels. 
    if hidden != None: 
        self.evaluate()
  
  def __str__(self):
    hidden = "   ".join([str(i) for i in self.hidden])
    upper = "\n".join([str(i) for i in self.upper])
    lower = "\n".join([str(i) for i in self.lower])
    return f"[{hidden}]\n{upper}\n{lower}" if self.isUp else f"{upper}\n{lower}\n[{hidden}]"

  def __eq__(self, other):
    return np.all(self.upper == other.upper) and np.all(self.lower == other.lower) and np.all(self.hidden == other.hidden) and self.isUp == self.isUp

  def move(self, action):
    if action[0]: # rotate upper to the left
      self.upper = np.roll(self.upper, -1, 1)
    elif action[1]: # rotate upper to the right
      self.upper = np.roll(self.upper, 1, 1)
    elif action[2]: # rotate lower to the left
      self.lower = np.roll(self.lower, -1, 1)
    elif action[3]: # rotate lower to the right
      self.lower = np.roll(self.lower, 1, 1)
    elif action[4]: # switch the position of the towers
      self.isUp = not self.isUp
      for i in range(3):
        temp = self.hidden[i]
        iter = 2 * i
        if self.isUp:
          self.hidden[i] = self.upper[0][iter]
          self.upper[0][iter] = self.upper[1][iter]
          self.upper[1][iter] = self.lower[0][iter]
          self.lower[0][iter] = self.lower[1][iter]
          self.lower[1][iter] = temp
        else:
          self.hidden[i] = self.lower[1][iter]
          self.lower[1][iter] = self.lower[0][iter]
          self.lower[0][iter] = self.upper[1][iter]
          self.upper[1][iter] = self.upper[0][iter]
          self.upper[0][iter] = temp
    else:
      print("Incorrect Move")
    
    self.evaluate()

  def evaluate(self):
    comboUpper = comboLower = value = -20
    for i in range(5):
      equivalent = 0
      if self.upper[0][i] == self.upper[1][i]:
        comboUpper += 1
        equivalent += 1
      if self.lower[0][i] == self.lower[1][i]:
        comboLower += 1
        equivalent += 1
      if self.upper[0][i] == self.lower[0][i]:
        value += 1
    self.reward = 1_000 if value == -15 else value + 1.5 * comboUpper + 1.5 * comboLower

  def scrub(self, action):
    return np.concatenate([self.upper.flatten(), self.lower.flatten(), self.hidden, [self.isUp], action])
  
  def scrub_all(self):
    return np.array([self.scrub([1, 0, 0, 0, 0]), self.scrub([0, 1, 0, 0, 0]), self.scrub([0, 0, 1, 0, 0]), self.scrub([0, 0, 0, 1, 0]), self.scrub([0, 0, 0, 0, 1])])