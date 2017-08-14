import random


class NN:
    def __init__(self, threshold, size):
        self.threshold = threshold
        self.size = size
        self.init_weight()

    def init_weight(self):
        self.weights = [[random.randint(1, 10) for x in range(self.size)] for j in range(self.size)]

    def check_sample(self, sample):
        vsum = 0
        for i in range(self.size):
            for j in range(self.size):
                vsum += self.weights[i][j] * sample[i][j]
        if vsum > self.threshold:
            return True
        else:
            return False

    def teach(self, sample):
        for i in range(self.size):
            for j in range(self.size):
                self.weights[i][j] += sample[i][j]

nn = NN(20, 6)

tsample1 = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

tsample2 = [
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

tsample3 = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
]

tsample4 = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0],
]

wsample1 = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

wsample2 = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

wsample3 = [
    [0, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

print('After learning:')
for row in nn.weights:
    print(row)

nn.teach(tsample1)
nn.teach(tsample2)
nn.teach(tsample3)
nn.teach(tsample4)

print(u"чайка" if nn.check_sample(wsample1) else u"НЛО")
print(u"чайка" if nn.check_sample(wsample2) else u"НЛО")
print(u"чайка" if nn.check_sample(wsample3) else u"НЛО")

print('Before learning:')
for row in nn.weights:
    print(row)