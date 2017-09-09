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

layer1 = Layer((2, 4))
layer2 = Layer((4, 10))
layer3 = Layer((10, 1))
nn = NeuralNetwork([layer1,layer2, layer3])

polimorfic = layer

print 'X', X.shape
print 'Y', Y.shape

print ''

for steps in range(1000):
	polimorfic.learn(X, Y)

W = layer.W
B = layer.B
J = polimorfic.cost(polimorfic.activate(X), Y)

print 'J: ', J
print 'w: ', W.T, ' | b: ', B
print ''
print ''
print ''
print ''
print ''

t = np.genfromtxt('test.csv', delimiter=',')
t = reduce(process_input, t, Input(X=[], Y=[]))

X = np.matrix(t.X).T
Y = np.matrix(t.Y)
 
print 'X', X.shape
print 'Y', Y.shape

A = polimorfic.activate(X)
print 'J', polimorfic.cost(A, Y)

print ''
print ''
print ''
print ''
print ''
print ''
np.testing.assert_allclose(np.round(A.tolist()), Y.tolist())

print 'Everything ok!'