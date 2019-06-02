import numpy as np
from matplotlib import pyplot as plt

def featureNormalize(X):
	"""
	Calculates and returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.

	Args:
		X: It a ndarray which contains features. Each of its row is a training example and
		   each column has an attribute of training examples.
    
    Returns:
    	X: The normalized version of X where
           the mean value of each feature is 0 and the standard deviation
           is 1.
        mu: Contains mean of every column in X.
        sigma: Contains standard deviation of every column in X

	"""
	mu = np.mean(X, axis=0)
	sigma = np.std(X, axis=0)
	X_norm = (X - mu) / sigma

	return X_norm, mu, sigma

def computeCost(X, y, theta):
	"""
	Computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
	
	Args:
		X: Input feature ndarray.
		y: Output array
		theta: Current parameters for linear regression.
	
	Returns:
		J: Computed cost of using theta as parameters for linear regression
		to fit the data points in X and y.
	"""
	hypothesis = np.dot(X, theta)
	loss = hypothesis - y
	J = np.sum(loss ** 2) / (2 * m)
	return J

def gradientDescent(X, y, theta=None, alpha=0.01, num_iters=100):
	"""
	Performs gradient descent to learn theta.

	Args:
		X: Input feature ndarray.
		y: Output array
		theta: Initial parameters for linear regression.
		alpha: The learning rate.
		num_iters: Number of iterations of gradient descent to be performed.

	Returns:
		theta: Updated parameters for linear regression.
		J_history: An array that contains costs for every iteration.
	"""
	m = X.shape[0]
	n = X.shape[1]
	if theta is None:
		theta = np.zeros((n, 1))

	J_history = np.zeros((num_iters, 1))

	for it in range(num_iters):
		hypothesis = np.dot(X, theta)
		loss = hypothesis - y
		theta = theta - (alpha / m) * np.dot(X.T, loss)
		J_history[it] = computeCost(X, y, theta)

	return theta, J_history

def normalEqn(X, y):
	"""
	Computes the closed-form solution to linear regression using
	normal equations.

	Args:
		X: Input feature ndarray.
		y: Output array

	Returns:
		theta: Parameters for linear regression calculated using normal 
		       equations.
	"""
	theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
	return theta

print('Loading data ...\n')

# Load data
data = np.genfromtxt('data.txt', delimiter=',')
m = data.shape[0]
n = data.shape[1]-1
X = data[:, 0:-1].reshape((m, n))
y = data[:, -1].reshape((m, 1))

# Print out some data points
print('First 10 examples from the dataset:')
print('x = ', X[0:10, :], "\ny = ", y[0:10])

print('\nProgram paused. Press enter to continue.')

input()

# Scale features
print('\nNormalizing Features ...')

X, mu, sigma = featureNormalize(X)

# Add intercept term to X
ones_col = np.ones((m, 1))
X = np.hstack((ones_col, X))

# Running gradient descent
print('\nRunning gradient descent ...')

# Choose some alpha value
alpha = 0.01
num_iters = 100

# Init theta and run gradient descent
theta = np.zeros((n+1, 1))
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(range(len(J_history)), J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.savefig('Cost at each iteration.png')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent:')
print(theta)
print()

# Estimate the price of  a 1650 sq-ft, 3 br house
x = np.array([1650, 3])
x_norm = (x - sigma) / mu
x_norm = np.hstack((1, x_norm)).reshape((1, n+1))
price = np.dot(x_norm, theta)

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):', price, '\n')

print('\nProgram paused. Press enter to continue.')

input()

print('\nSolving with normal equations ...\n')

# Load data
data = np.genfromtxt('data.txt', delimiter=',')
m = data.shape[0]
X = data[:, 0:-1]
y = data[:, -1].reshape((m, 1))

# Add intercept term to X
ones_col = np.ones((m, 1))
X = np.hstack((ones_col, X))

theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from normal equations:')
print(theta)
print()

# Estimate the price of a 1650 sq-ft, 3 br house
x = np.array([1, 1650, 3])
price = np.dot(x, theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):', price, '\n')