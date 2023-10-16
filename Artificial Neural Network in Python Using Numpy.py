An Artificial Neural Network (ANN) is an information processing paradigm that is inspired the brain.

from joblib.numpy_pickle_utils import xrange
from numpy import *

class Neural_Network(object):
	def __init__(self):
		#Generate Random Numbers
		random.seed(4)
	
	#Assign random weights to a matrix
	self.weights = 2 * random.random((4, 2)) - 1

	#Sigmoid Function
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))

	#derivative of sigmoid function
	#This is the gradient of the Sigmoid Curve
	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	#Training the Neural Network by adjusting the weights each time 
	def train(self, inputs, outputs, training_iterators):
		for iterations in xrange(training_iterators):
			#Passing the training set.
			output = self.learn(inputs)
			
			#Calculating the error
			error = outputs - output
			
			#Adjusting the weights by a small factor 
			factor = dot(inputs.T, error * self.__sigmoid_derivative(output))
			self.synaptic_weights += factor

	#Making the Neural Network Think.
	def learn(self, inputs):
		return self.__sigmoid(dot(inouts, self.synaptic_weights))

if __name__ == "__main__":
	#Initializing
	NeuralNetwork = NeuralNet()
	
	#Training Set.
	inputs = array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
    outputs = array([[1, 0, 1]]).T

	#Training the NN
	NeuralNetwork.train(inputs, outputs, 100000)

	#Testing the NN with a test case.
	print(NeuralNetwork.learn(array([1,0,1)))
	
