import numpy as np
import collections

Input = collections.namedtuple('Input', 'X Y')

class BinaryLinearNeuron:

	def __init__(self, nx):
		self.W = np.zeros((nx, 1))
		self.b = 0
		self.alpha = 1
		self.inputs = np.array([])
		self.outputs = np.array([])

	def set_input(self, inputs):
		self.inputs = np.array(inputs)

	def set_output(self, outputs):
		self.outputs = np.array(outputs)

	def __activate(self, X):
		Z = np.dot(self.W.T, X) + self.b
		return 1/(1 + np.exp(-Z))

	def activate(self, X):
		A = __activate(X)
		for o in self.outputs:
			o.propagate_forward(A)

	def correct(self, X, dZ):
		mx = len(X)
		dW = X * dZ.T / mx
		db = np.sum(dZ) / mx
		self.W -= self.alpha * dW
		self.b -= self.alpha * db

		for i in self.inputs:
			i.propagate_backwards(dZ)

class Connector:

	def __init__(self, inputs, outputs):
		self.inputs = inputs
		self.outputs = outputs
		self.X = np.zeros((len(inputs), 1)

		for i in inputs:
			i.set_output(self)

		for o in outputs:
			o.set_input(self)


	def propagate_forward(self, elem, A):
		for o in self.outputs:
			return

		
	def propagate_backwards(self, elem, dZ):
		return

m = np.genfromtxt('train.csv', delimiter=',')
mx = len(m)

def process_input(acc, input):
	acc.X.append(input[:-1])
	acc.Y.append(input[-1])
	return acc

init = Input(X=[], Y=[])
m = reduce(process_input, m, init)
nx = len(m.X[0])

X = np.matrix(m.X).T
Y = np.matrix(m.Y)


neuron1 = BinaryLinearNeuron(nx)
neuron = BinaryLinearNeuron(1)


print 'X', X.shape
print 'Y', Y.shape
print 'W', neuron.W.shape

print ''
print ''

alpha = 1

for steps in range(5000):
	A = neuron.activate(X)
	
	# print ''
	# print '----------'
	# print 'A: ', A, 'Z: ', Z, '|||  1-A: ', 1-A, ' ||| np.log(1 - A)', np.log(1 - A)
	# print '------------'
	
	dZ = A - Y

	L = -(np.dot(Y, np.log(A.T)) + np.dot((1-Y), np.log(1 - A.T)))

	J = L / mx

	neuron.correct(X, dZ)

print ''
print ''

W = neuron.W
b = neuron.b

print 'J: ', J
print 'w: ', W.T, ' | b: ', b
print ''
print ''
print ''
print ''
print ''

t = np.genfromtxt('test.csv', delimiter=',')

init = Input(X=[], Y=[])
t = reduce(process_input, t, init)

X = np.matrix(t.X).T
Y = np.matrix(t.Y)

print 'X', np.round(X,2)
print 'Y', Y
print 'W', W.shape


A = neuron.activate(X)

L = -(np.dot(Y, np.log(A.T))
	+ np.dot((1-Y), np.log(1 - A.T)))

A = np.round(A)


print ''
print ''
print ''
print Y
print A
