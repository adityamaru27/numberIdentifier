import gzip
import numpy as np
import timeit
import siz.moves.cPickle as pickle
import theano
import theano.tensor as T

#weight matrix
self.W = theano.shared(
	value=np.zeros((num_in, num_out), dtype=theano.config.floatX),
	name='W', 
	borrow=True
	)

#bias vector
self.b = theano.shared(
	value=np.zeros((num_out,), dtype=theano.config.floatX),
	name='b',
	borrow=T
	)

#function to find probability of class label
self.p_y_x = T.nnet.softmax(T.dot(x, self.W) + self.b)

#mechanism for calculating the actual prediction
self.pred_y = T.argmax(self.p_y_x, axis=1)

#negative log likelihood function

def neagtaive_log_likelihood(self, y):
	return -T.mean(T.log(self.p_y_x)[T.arrange(y.size[0], y)])

''' arrange makes a vector containing integers from 0...N-1
	log function makes a N * 10 matrix of log values of the probabilites
	of class y given feature vector x
	this whole line combined gives us a vector N*1 of the log likelihoods of 
	each training example/class digit pair, and then we take the mean'''

def error(self, y):
	#compare y to y_pred
	if y.ndim != self.y_pred.ndim:
		raise TypeError("y should have same shape as y_pred",
			("y", y.type, "y_pred", y_pred.type))
	if y.dtype.startswith('int'):
		T.mean(T.neq(self.y_pred, y))

	# neq returns a vector 1's and 0's where 1 represents error of classification

# sequentially copying the dataset will be to expensive so let us make shared Theano variable
# to be able to copy to the GPU

def shared_variable(self, data_xy):
	data_x, data_y = data_xy
	shared_x = theano.shared(
		np.asarray(data_x, dtype=theano.config.floatX),
		borrow=borrow
		)

	shared_y = theano.shared(
		np.asarray(data_y, dtype=theano.config.floatX),
		borrow=borrow
	)		
	return shared_x, T.cast(shared_y, 'int32')

# now we must unzip and load the MNSET database
# divide it into training, validation and testing sets

def load_mnist_data(filename):
	with gzip.open(filename, 'rb') as gzf:
		try:
			train_set, validation_set, test_set = pickle.load(
				gzf, encoding='latin1'
				)
		except:
			train_set, validation_set, test_set = pickle.load(gzf)

	#copy to gpu

	test_set_x, test_set_y = shared_variable(test_set)
	train_set_x, train_set_y = shared_variable(train_set)
	valid_set_x, valid_set_y = shared_variable(validation_set)

	#list of tuples containing feature-response pairs

	mylist = [(test_set_x, test_set_y),
	(train_set_x, train_set_y),
	(valid_set_x, valid_set_y)]


#stochastic gradient descent
def stoch_grad_desc_training(filename, gamma=0.13, epochs=1000, B=600):
	datasets = load_mnist_data(filename)

	#loading the correct data
	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	#calculating the batches of each of these datasets we will use
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] // B
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // B
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] // B

	# now building the actual model
	print("Building the model...")

	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')

	#creating an instance of the logreg model
	logreg = LogisticRegression(x=x, num_in=28**2, num_out=10)
	cost = logreg.neagtaive_log_likelihood(y)
	# the above cost is what we have to minimisde using SGD

	test_model = theano.function(
		inputs=[index],
		outputs=logreg.errors(y),
		givens={
		x: test_set_x[index*B: (index+1) * B],
		y: test_set_y[index*B: (index+1) * B]
		})
	valid_model=theano.function(
		inputs=[index],
		outputs=logreg.errors(y),
		givens={
		x: valid_set_x[index*B: (index+1) * B],
		y: valid_set_y[index*B: (index+1) * B]
		})

	#gradient calculation
	grad_W = T.grad(cost=cost, wrt=logreg.W)
	grad_b = T.grad(cost=cost, wrt=logreg.b)

	#gradient updates
	updates = [
	(logreg.W, logreg.W - gamma*grad_W),
	(logreg.b, logreg.b - gamma*grad_b)
	]

	train_model = theano.function(
		inputs=[index],
		outputs=cost,
		updates=updates,
		givens={
		x: train_set_x[index*B: (index+1) * B],
		y: train_set_y[index*B: (index+1) * B]
		})

	patience = 5000
	#minimum number of examples to look at in the minibatch
	patience_increae = 2
	improvement_threshold = 0.995

	#determine how often to assess the performance on the validation set
	validation_frequency = min(n_train_batches, patience // 2)
	best_validation_loss = np.inf
	test_score = 0

	#start the timer
	start_time = timeit.default_timer()
	




		





















