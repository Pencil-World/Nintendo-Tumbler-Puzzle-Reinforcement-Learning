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

        self.isUp = isUp
        self.upper = np.zeros([2, 5, 6])
        self.lower = np.zeros([2, 5, 6])
        self.hidden = np.zeros([1, 3, 6])
        if not hidden:
            self.reward = 0
            return

        for (mat, index) in zip([self.upper, self.lower, self.hidden], [upper, lower, hidden]):
            for row, arr in enumerate(index):
                for col, elem in enumerate(arr):
                    mat[row][col][elem] = 1

        # state of hidden. either above or below the main 2 barrels. 
        self.evaluate()
    
    def __str__(self):
        hidden = "   ".join([str(i.argmax()) for i in self.hidden[0]])
        upper = np.array([[elem.argmax() for elem in arr] for arr in self.upper])
        lower = np.array([[elem.argmax() for elem in arr] for arr in self.lower])
        return f"[[{hidden}]]\n{upper}\n{lower}" if self.isUp else f"{upper}\n{lower}\n[[{hidden}]]"

    def __eq__(self, other):
        return self.reward and other.reward and self.isUp == self.isUp and (self.upper == other.upper).all() and (self.lower == other.lower).all() and (self.hidden == other.hidden).all()

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
            for i in range(0, 6, 2):
                temp = copy.deepcopy(self.hidden[0][i // 2])
                if self.isUp:
                    self.hidden[0][i // 2] = self.upper[0][i]
                    self.upper[0][i] = self.upper[1][i]
                    self.upper[1][i] = self.lower[0][i]
                    self.lower[0][i] = self.lower[1][i]
                    self.lower[1][i] = temp
                else:
                    self.hidden[0][i // 2] = self.lower[1][i]
                    self.lower[1][i] = self.lower[0][i]
                    self.lower[0][i] = self.upper[1][i]
                    self.upper[1][i] = self.upper[0][i]
                    self.upper[0][i] = temp
        else:
            print("Incorrect Move")
        
        self.evaluate()

    def evaluate(self):
        terminal = 0
        for i in range(5):
            terminal += 1 == self.upper[0][i][i + 1] == self.upper[1][i][i + 1] == self.lower[0][i][i + 1] == self.lower[1][i][i + 1]
        self.reward = 100 if terminal == 5 else 0

    def scrub(self, action):
        return np.concatenate([self.upper.flatten(), self.lower.flatten(), self.hidden.flatten(), [self.isUp], action])
    
    def scrub_all(self):
        return np.array([self.scrub([1, 0, 0, 0, 0]), self.scrub([0, 1, 0, 0, 0]), self.scrub([0, 0, 1, 0, 0]), self.scrub([0, 0, 0, 1, 0]), self.scrub([0, 0, 0, 0, 1])])