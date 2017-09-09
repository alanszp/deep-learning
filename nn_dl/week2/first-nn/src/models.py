import numpy as np

class LogisticActivation:

	def compute(self, Z):
		return  1 / (1 + np.exp(-Z))

	def derivate(self, Z):
		A = np.nan_to_num(self.compute(Z))
		return np.multiply(A, -A + 1)

class NeuralNetwork:

	def __init__(self, layers):
		self.validate(layers)

		self.layers = layers
		self.size = self.calculate_size()

	def validate(self, layers):
		last = None
		for l in layers:
			if last is None:
				last = l
				continue

			if last.shape[1] != l.shape[0]:
				raise ValueError("shapes ("+ str(last.shape[0]) +", "+ str(last.shape[1]) +") and ("+ str(l.shape[0]) +", "+ str(l.shape[1]) +") not aligned: "+ str(last.shape[1]) +" (dim 1) != "+ str(l.shape[0]) +" (dim 0)")

			last = l

	def calculate_size(self):
		first = self.layers[0]
		last = self.layers[-1]

		return (first.shape[0], last.shape[1])

	def activate(self, X, memory = False):
		def stack_activation(stack, layer):
			stack.append(layer.activate(stack[-1]))
			return stack

		stack = reduce(stack_activation, self.layers, [X])

		if memory:
			return stack
		else:
			return stack[-1]

	def learn(self, X, Y):
		staked_A = self.activate(X, memory = True)

		l = zip(self.layers, staked_A)

		last_A = staked_A[-1]
		dZ = np.nan_to_num((- last_A + Y) / (np.multiply(last_A, last_A - 1)))

		for l in reversed(l):
			layer, A = l
			dZ = layer.correct(A, dZ)

		return self.loss(staked_A[-1], Y)

	def loss(self, A, Y):
		last = self.layers[-1]
		return last.loss(A, Y)

	def cost(self, A, Y):
		last = self.layers[-1]
		return last.cost(A, Y)


class Layer:

	def __init__(self, shape, activation = None, alpha = 0.5):
		self.shape = shape
		self.W = np.zeros(shape)
		self.B = np.zeros((self.shape[1], 1))
		self.alpha = alpha

		if activation is None:
			self.activation = LogisticActivation()
		else:
			self.activation = activation

	def linear(self, X):
		return np.dot(self.W.T, X) + self.B

	def activate(self, X):
		Z = self.linear(X)
		return self.activation.compute(Z)

	def correct(self, A, dO):
		mx = A.shape[1]
		Z = self.linear(A)
		dZ = self.activation.derivate(Z)
		
		dOZ = np.multiply(dZ, dO)

		dW = A * dOZ.T / mx
		dB = np.sum(dOZ) / mx

		self.W = self.W - self.alpha * dW
		self.B -= self.alpha * dB

		return dOZ

	def learn(self, X, Y):
		A = self.activate(X)
		dL = np.nan_to_num((- A + Y) / (np.multiply(A, A - 1)))

		self.correct(X, dL)
		return self.loss(A, Y)

	def loss(self, A, Y):
		L = -(np.multiply(Y, np.log(A)) + np.multiply((1-Y), np.log(1 - A)))
		return np.nan_to_num(L)

	def cost(self, A, Y):
		mx = Y.shape[1]
		L = self.loss(A, Y)
		return np.sum(L) / mx