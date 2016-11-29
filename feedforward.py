import lasagne
from lasagne import layers

import theano
import theano.tensor as T
import os
import numpy as np

class LSTM():
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

net=LSTM()
for i,generator_filename in enumerate(os.listdir('*.json')):
	referenceData=json.loads(''.join(open(generator_filename).readlines()))
	modulationTypes=['Noise','FM','GMSK']
	target=np.zeros(30)
	for bin_ in range(30):
		for signal in referenceData['signals']:
			if bin_ in signal['occupied_bins']:
				target[bin_]=modulationTypes.index(signal['modulation_type'])
				break
	waterfallData=makeDataFromFilename(generator_filename)
	assert waterfallData.shape[1]==3000

	train=[waterfallData[:,i*100:(i+1)*100] for i in range(30)]
	print 'Loss of training:',net.train_function(train,target)

"""
input=T.matrix('input')
output=T.ivector('output')

l_input=layers.InputLayer((None,1000))
l_lstm=layers.LSTMLayer(l_input,num_units=1000,nonlinearity=lasagne.nonlinearities.tanh)
l_slice=layers.SliceLayer(l_lstm,-1)
l_class=layers.DenseLayer(l_slice,num_units=5,nonlinearity=lasage.nonlinearities.softmax)

l_outvalue=layers.get_output(l_class,input)
loss=layers.objectives.categorical_crossentropy(l_outvalue,output)
updates=lasagne.updates.nesterov_momentum(l_class,loss,learning_rate=0.0001)

train_function=theano.function([input,output],loss,updates=updates)
pred_function=theano.function([input],l_outvalue)


counter=0
for bindata,binclass in datagenerator():
	loss=train_function(bindata, binclass)
	counter+=1
	if counter%100==0:
		testdata,testclass=datagenerator()
		testprediction=pred_function(testdata)
		print sum([testp==testa for testp,testa in zip(testprediction,testclass)])/len(testclass)
	if loss<0.5:
		break


pred_function(unknownBinData)
"""