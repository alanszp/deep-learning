import numpy as np
import collections

Input = collections.namedtuple('Input', 'X Y')

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


W = np.zeros((nx, 1))
b = 0

print 'X', X.shape
print 'Y', Y.shape
print 'W', W.shape

print ''
print ''

alpha = 1

for steps in range(5000):
	Z = np.dot(W.T, X) + b
	A = 1/(1 + np.exp(-Z))
	
	# print ''
	# print '----------'
	# print 'A: ', A, 'Z: ', Z, '|||  1-A: ', 1-A, ' ||| np.log(1 - A)', np.log(1 - A)
	# print '------------'
	
	dZ = A - Y

	L = -(np.dot(Y, np.log(A.T)) + np.dot((1-Y), np.log(1 - A.T)))

	J = L / mx
	dW = X * dZ.T / mx
	db = np.sum(dZ) / mx

	#print 'J: ', J, ' | dw: ', dW.T, ' | db: ', db

	W -= alpha * dW
	b -= alpha * db

	#print 'w: ', W, ' | b: ', b
print ''
print ''

print 'J: ', J, ' | dw: ', dW.T, ' | db: ', db
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


Z = W.T * X + b
A = 1/(1 + np.exp(-Z))

L = -(np.dot(Y, np.log(A.T))
	+ np.dot((1-Y), np.log(1 - A.T)))

A = np.round(A)

print Z
print Y
print A
