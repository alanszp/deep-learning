import numpy as np
import collections
from src import *

Input = collections.namedtuple('Input', 'X Y')

def process_input(acc, input):
	acc.X.append(input[:-1])
	acc.Y.append(input[-1])
	return acc

m = np.genfromtxt('train.csv', delimiter=',')
m = reduce(process_input, m, Input(X=[], Y=[]))


X = np.matrix(m.X).T
Y = np.matrix(m.Y)

layer = Layer((2, 1))

layer1 = Layer((2, 1))
nn1 = NeuralNetwork([layer1])

layer1 = Layer((2, 3))
layer2 = Layer((3, 1))
nn2 = NeuralNetwork([layer1,layer2])

layer1 = Layer((2, 3), activation = SigmoidActivation())
layer2 = Layer((3, 4), activation = SigmoidActivation())
layer3 = Layer((4, 1))
nn3 = NeuralNetwork([layer1,layer2, layer3])

nn4 = NeuralNetwork([
	Layer((2, 10), activation = TanhActivation()),
	Layer((10, 20), activation = TanhActivation()),
	Layer((20, 50), activation = TanhActivation()),
	Layer((50, 20), activation = TanhActivation()),
	Layer((20, 1), activation = SigmoidActivation())
])

polimorfic = nn4

print 'X', X.shape
print 'Y', Y.shape

print ''

for steps in range(10000):
	cost = polimorfic.learn(X, Y)
	if steps % 100 == 0:
            print(cost)

J = polimorfic.cost(polimorfic.activate(X), Y)

print 'J: ', J
print ''
print ''
print ''
print ''
print ''

t = np.genfromtxt('test.csv', delimiter=',')
t = reduce(process_input, t, Input(X=[], Y=[]))

test_X = np.matrix(t.X).T
test_Y = np.matrix(t.Y)
 
print 'X', X.shape
print 'Y', Y.shape

A = polimorfic.activate(test_X)
print 'J', polimorfic.cost(A, test_Y)

print ''
print ''

np.testing.assert_allclose(np.round(A.tolist(),2), test_Y.tolist())

print ''
print ''
print ''
print ''

print 'Everything ok!'