import lasagne
from lasagne import layers

import theano
import theano.tensor as T
import os
import numpy as np

class NN():
	"""
		The reinforcement learning language neural network
	"""
	def __init__(self, n_in=100, n_hidden=500, n_out=5):
		self.hidden=n_hidden
		self.makeNetwork(n_in,n_hidden,n_out)

	def makeNetwork(self,n_in,n_hidden,n_out,learning_rate=0.00002):
		x = T.tensor3s('x')  # shp: num_batch x num_features
		y = T.ivector('y')# shp: num_batch
		l_input=layers.InputLayer((30,None,n_in))
		# bs,_=x.shape
		# l_rinput=layers.ReshapeLayer(l_input,((bs,None,100)))
		l_forward_1 = lasagne.layers.LSTMLayer(l_input, 500, nonlinearity=lasagne.nonlinearities.tanh)
		l_forward_slice = lasagne.layers.SliceLayer(l_forward_1, -1, 1)
		l_hid=layers.DenseLayer(l_forward_slice,num_units=n_hidden)
		self.l_out=layers.DenseLayer(l_hid,num_units=n_out,nonlinearity=lasagne.nonlinearities.softmax)
		
		l_outvalue = layers.get_output(self.l_out, x)
		params = layers.get_all_params(self.l_out)
		loss = T.mean(lasagne.objectives.categorical_crossentropy(l_outvalue, y))
		updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=0.9)
		self.train_function = theano.function([x, y],loss, updates=updates,allow_input_downcast=True)
		self.pred_function=theano.function([x],l_outvalue,allow_input_downcast=True)

	def testNetwork(self,train,target):
		predictions=self.pred_function(train)
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
