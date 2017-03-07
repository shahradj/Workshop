import lasagne
from lasagne import layers

from sklearn.datasets import load_iris
import theano
import theano.tensor as T
import os
import numpy as np
import random

class NN():
	"""
		Neural network
	"""
	def __init__(self, n_in=4, n_hidden=50, n_out=3,classification=True):
		self.hidden=n_hidden
		if classification:
			self.makeClassificationNetwork(n_in,n_hidden,n_out)
		else:
			self.makeRegressionNetwork(n_in,n_hidden,n_out)

	def makeClassificationNetwork(self,n_in,n_hidden,n_out,learning_rate=0.001):
		"""
			build a feedforward neural network with classification output
		"""
		#network input
		input_ = T.matrix('input_')  # matrix of shape batch size times number of input variables
		target_ = T.ivector('target_')# integer vector of batchsize length

		#network
		l_input=layers.InputLayer((None,n_in))
		l_hid=layers.DenseLayer(l_input,num_units=n_hidden)
		self.l_out=layers.DenseLayer(l_hid,num_units=n_out,nonlinearity=lasagne.nonlinearities.softmax)

		#network output
		l_outvalue = layers.get_output(self.l_out, input_)
		self.predict=theano.function([input_],l_outvalue,allow_input_downcast=True)

		#loss/cost function
		loss = T.mean(lasagne.objectives.categorical_crossentropy(l_outvalue, target_))

		#calculate the updates
		params = layers.get_all_params(self.l_out)
		updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=0.9)
	
		#update the weights from a given input and target
		self.train_function = theano.function([input_, target_],loss, updates=updates,allow_input_downcast=True)

	def makeRegressionNetwork(self,n_in,n_hidden,n_out,learning_rate=0.001):
		"""
			build a feedforward neural network with regression output
		"""
		#network input
		input_ = T.matrix('input_')  # matrix of shape batch size times number of input variables
		target_ = T.matrix('target_')# matrix of shape batch size times number of output variables
		
		#network
		l_input=layers.InputLayer((None,n_in))
		l_hid=layers.DenseLayer(l_input,num_units=n_hidden)
		self.l_out=layers.DenseLayer(l_hid,num_units=n_out,nonlinearity=None)
			
		#network output
		l_outvalue = layers.get_output(self.l_out, input_)
		self.predict=theano.function([input_],l_outvalue,allow_input_downcast=True)
		
		#loss/cost function
		loss = T.mean(lasagne.objectives.squared_error(l_outvalue, target_))
		
		#calculate the updates
		params = layers.get_all_params(self.l_out)
		updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=0.9)

		#update the weights from a given input and target
		self.train_function = theano.function([input_, target_],loss, updates=updates,allow_input_downcast=True)

	def testNetwork(self,input_,target):
		"""
			returns percentage accuracy of the network on a set of known targets (only for classification networks)
		"""
		predictions=self.predict(input_).argmax(axis=1)
		correctPredictions=[p==t for p,t in zip(predictions,target)]
		percentage_accuracy=sum(correctPredictions)*1.0/len(target)
		return percentage_accuracy
		
	def saveNetwork(self,filename='Weights.pkl'):
		"""
			save the network weights that has been trained
		"""
		thefile=open(filename,'wb')
		cPickle.dump(layers.get_all_param_values(self.l_out),thefile)

	def loadNetwork(self,filename='Weights.pkl'):
		"""
			load the network weights and build the corresponding network
		"""
		thefile=open(filename,'r')
		oldvals=cPickle.load(thefile)
		layers.set_all_param_values(self.l_out,oldvals)

if __name__=='__main__':
	net=NN()
	iris=load_iris()
	#separate the data into train and test samples
	trainIndices=random.sample(range(150),125)
	testIndices=set(range(150)).difference(trainIndices)
	traininput_=[iris.data[k] for k in trainIndices]
	traintarget_=[iris.target[k] for k in trainIndices]
	testinput_=[iris.data[k] for k in testIndices]
	testtarget_=[iris.target[k] for k in testIndices]
	
	batchsize=5
	n_epochs=10
	totalcosts=[]
	for epoch in range(n_epochs):
		#train the network
		for j in range(len(traininput_)/batchsize):
			input_=traininput_[j*batchsize:(j+1)*batchsize]
			target_=traintarget_[j*batchsize:(j+1)*batchsize]
			cost=net.train_function(input_,target_)
			totalcosts.append(cost)
			print 'Mean cost:',np.mean(totalcosts)
		
		#test the network
		perc_acc=net.testNetwork(testinput_,testtarget_)
		print 'Percentage accuracy of test set:%f'%perc_acc
