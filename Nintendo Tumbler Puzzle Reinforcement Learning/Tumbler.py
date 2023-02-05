import copy
#import json
import numpy as np

class Tumbler():
    def __init__(self, upper = None, lower = None, hidden = None, isUp = False, dict = None):
        if dict:
            for key, val in dict.items():
                if type(val) is list:
                    val = np.array(val)
                setattr(self, key, val)
            return

        if not hidden:
            self.reward = 0
            return

        self.upper = np.zeros([2, 5, 6])
        self.lower = np.zeros([2, 5, 6])
        self.hidden = np.zeros([1, 3, 6])
        for temp, (mat, index) in enumerate(zip([self.upper, self.lower, self.hidden], [upper, lower, hidden])):
            for row, arr in enumerate(mat):
                for col, elem in enumerate(arr):
                    elem[index[row][col]] = 1

        self.isUp = isUp # state of hidden. can be above or below the main 2 barrels. 
        self.evaluate()
    
    def __str__(self):
        hidden = "   ".join([str(i.argmax()) for i in self.hidden[0]])
        upper = np.array([[elem.argmax() for elem in arr] for arr in self.upper])
        lower = np.array([[elem.argmax() for elem in arr] for arr in self.lower])
        return f"[[{hidden}]]\n{upper}\n{lower}" if self.isUp else f"{upper}\n{lower}\n[[{hidden}]]"

    def __eq__(self, other):
        return self.reward and other.reward and self.isUp == self.isUp and np.all(self.upper == other.upper) and np.all(self.lower == other.lower) and np.all(self.hidden == other.hidden)

    #@staticmethod
    #def json_dumps(object):
    #    return json.dumps(object, default = lambda o: (o.tolist() if type(o) is np.ndarray else o.__dict__), 
    #        sort_keys = True, indent = 4)

    #@staticmethod
    #def json_loads(object):
    #    return Tumbler(dict = json.loads(object))
    
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
                temp = copy.deepcopy(self.hidden[0][i])
                iter = 2 * i
                if self.isUp:
                    self.hidden[0][i] = self.upper[0][iter]
                    self.upper[0][iter] = self.upper[1][iter]
                    self.upper[1][iter] = self.lower[0][iter]
                    self.lower[0][iter] = self.lower[1][iter]
                    self.lower[1][iter] = temp
                else:
                    self.hidden[0][i] = self.lower[1][iter]
                    self.lower[1][iter] = self.lower[0][iter]
                    self.lower[0][iter] = self.upper[1][iter]
                    self.upper[1][iter] = self.upper[0][iter]
                    self.upper[0][iter] = temp
        else:
            print("Incorrect Move")
        
        self.evaluate()

    def evaluate(self):#switch reward and value
        comboUpper = comboLower = 0
        value = -20
        for i in range(5):
            comboUpper += self.upper[0][i].argmax() == self.upper[1][i].argmax()
            comboLower += self.lower[0][i].argmax() == self.lower[1][i].argmax()
            value += self.upper[0][i].argmax() == self.upper[1][i].argmax() == self.lower[0][i].argmax() == self.lower[1][i].argmax()
        self.reward = 1_000 if value == -15 else value + 1.5 * comboUpper + 1.5 * comboLower

    def scrub(self, action):
        return np.concatenate([self.upper.flatten(), self.lower.flatten(), self.hidden.flatten(), [self.isUp], action])
    
    def scrub_all(self):
        return np.array([self.scrub([1, 0, 0, 0, 0]), self.scrub([0, 1, 0, 0, 0]), self.scrub([0, 0, 1, 0, 0]), self.scrub([0, 0, 0, 1, 0]), self.scrub([0, 0, 0, 0, 1])])