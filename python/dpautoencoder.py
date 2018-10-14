'''
Differentially Private Auto-Encoder
author: Hai Phan
'''
import timeit
import sys
import numpy as np
import tensorflow as tf
# import input_data
import math
import os
from util import *
from logisticRegression import LogisticRegression
from logisticRegression import dpLogisticRegression
from dpLayers import HiddenLayer
from dpLayers import Autoencoder
from dpLayers import ConvFlat

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def next_batch( X, y, start, batch_size, num_examples):
    """Return the next `batch_size` examples from this data set."""
    start = batch_size*start # i 
    index_in_epoch = batch_size*(start+1)
    # self._i ndex_in_epoch += batch_size
    if index_in_epoch > num_examples:
      # Start next epoch
      start = 0
      index_in_epoch = batch_size*(start+1)
      assert batch_size <= num_examples
    end = index_in_epoch
    return X[start:end], y[start:end]
    # return X[start:end,:], y[start:end,:]


class dpAutoEncoder(object):
    '''
    An implement of differentially private auto-encoder
    '''
    def __init__(self, n_in=16000, n_out=31, hidden_layers_sizes=[248,1024,248], epsilon = 0.25, _batch_size = 128, finetuneLR = 0.01):
        '''
        :param n_in: int, the dimension of input
        :param n_out: int, the dimension of output
        :param hidden_layers_sizes: list or tuple, the number of hidden neurons, the last item will be the number of hidden neurons in the last hidden layer
        :param epsilon: privacy budget epsilon
        :param _batch_size: the batch size
        :param finetuneLR: fine tunning learning rate
        '''
        # Number of layers
        assert len(hidden_layers_sizes) > 0
        self.n_layers = len(hidden_layers_sizes)
        self.layers = []    # hidden layers
        self.params = []       # keep track of params for training
        self.last_n_in = hidden_layers_sizes[-1] # the number of hidden neurons in the last hidden layer
        self.pretrain_ops = []; # list of pretrain objective functions for hidden layers
        self.epsilon = epsilon; # privacy budget epsilon epsilon
        self.batch_size = _batch_size; # batch size

        # Define the input, output, Laplace noise for the output layer
        self.x = tf.placeholder(tf.float32, shape=[None, n_in], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None, n_out])
        self.LaplaceNoise = tf.placeholder(tf.float32, self.last_n_in)
        ######
        
        #############################
        ##Construct the Model########
        #############################
        # Create the 1st auto-encoder layer
        Auto_Layer1 = Autoencoder(inpt=self.x, n_in = 16000, n_out = hidden_layers_sizes[0], activation=tf.nn.sigmoid)
        self.layers.append(Auto_Layer1)
        # append allow one element in 
        self.params.extend(Auto_Layer1.params)
        # get the pretrain objective function
        self.pretrain_ops.append(Auto_Layer1.get_dp_train_ops(epsilon = self.epsilon, data_size = 50000, learning_rate= 0.01))
        ###
        
        # Create the 2nd cauto-encoder layer
        Auto_Layer2 = Autoencoder(inpt=self.layers[-1].output, n_in = self.layers[-1].n_out, n_out = hidden_layers_sizes[1], activation=tf.nn.sigmoid)
        self.layers.append(Auto_Layer2)
        self.params.extend(Auto_Layer2.params)
        # get the pretrain objective function
        self.pretrain_ops.append(Auto_Layer2.get_dp_train_ops(epsilon = self.epsilon, data_size = 50000, learning_rate= 0.01))
        ###
        
        # Create the flat connected hidden layer
        flat1 = HiddenLayer(inpt=self.layers[-1].output, n_in = self.layers[-1].n_out, n_out = self.last_n_in, activation=tf.nn.relu)
        self.layers.append(flat1)
        self.params.extend(flat1.params)
        ###
        
        # Create the output layer
        # We use the differentially private Logistic Regression (dpLogisticRegression) layer as the objective function
        self.output_layer = dpLogisticRegression(inpt=self.layers[-1].output, n_in = self.last_n_in, n_out= n_out, LaplaceNoise = self.LaplaceNoise)
        # We can also use the non-differentially private layer: LogisticRegression(inpt=self.layers[-1].output, n_in=hidden_layers_sizes[-1], n_out=n_out)
        self.params.extend(self.output_layer.params)
        ###

        #######################################
        ##Define Fine Tune Cost and Optimizer##
        #######################################
        # The finetuning cost
        self.cost = self.output_layer.cost(self.y)
        # train_op for finetuning with AdamOptimizer
        global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(finetuneLR, global_step, 700, 0.96, staircase=True); # learning rate decay can be carefully used
        # Fine tune with AdamOptimizer. Note that we do not fine tune the pre-trained parameters at the auto-encoder layers
        self.train_op = tf.train.AdamOptimizer(finetuneLR).minimize(self.cost, var_list=[flat1.params, self.output_layer.params], global_step = global_step)
        # The accuracy
        self.accuracy = self.output_layer.accuarcy(self.y)
        ###
    
    def generateNoise(n_in, epsilon, data_size, test = False):
        Delta = 0.0;
        if test == True: # do not inject noise in the test phase
            Delta = 0.0;
        else:
            Delta = 10*(n_in + 1/4 * n_in**2); # global sensitivity for the output layer, note that 10 is the number of classes of the output layer
        # Generate the Laplace noise
        perturbFM = np.random.laplace(0.0, Delta/(epsilon*data_size), n_in)
        perturbFM = np.reshape(perturbFM, [n_in])
        return perturbFM

    def pretrain(self, sess, X_train, y, batch_size=128, pretraining_epochs=10, lr=0.01,
                    display_step=1):

        '''
        Pretrain the layers (just train the auto-encoder layers)
        :param sess: tf.Session
        :param X_train: the input of the train set (You might modify this function if you do not use the designed mnist)
        :param batch_size: int
        :param lr: float
        :param pretraining_epoch: int
        :param display_step: int
        '''
        print('Starting pretraining...\n')
        num_examples = X_train.shape[0] # X.shape[0]
        start_time = timeit.default_timer()
        batch_num = int(math.ceil(num_examples / batch_size))  # The number of batch per epoch
        # batch_num = int(math.ceil(X_train.train.num_examples / batch_size)) # The number of batch per epoch
        # Pretrain layer by layer
        for i in range(self.n_layers-1):
            # Get the cost of the current auto-encoder layer
            cost = tf.reduce_mean(self.layers[i].cost);
            # Get the objective function of the current auto-encoder layer
            train_ops = self.pretrain_ops[i]
            for epoch in range(pretraining_epochs):
                avg_cost = 0.0
                for j in range(batch_num):
                    # x_batch, _ = X_train.train.next_batch(batch_size)
                    x_batch, _ = next_batch(X_train, y, j, batch_size, num_examples)
                    x = [[0.0 for colum in range(x_batch.shape[1])] for row in range(x_batch.shape[0])]
                    # y = []
                    for row in range(x_batch.shape[0]):
                        for colum in range(x_batch.shape[1]):
                            x[row][colum] = x_batch[row][colum][0]
                    # training
                    sess.run(train_ops, feed_dict={self.x: x})
                    # cost
                    avg_cost += sess.run(cost, feed_dict={self.x: x}) / batch_num
                # print out the average cost every display_step
                if epoch % display_step == 0:
                    print("\tPretraing layer {0} Epoch {1} cost: {2}".format(i, epoch, avg_cost))

        end_time = timeit.default_timer()
        print("\nThe pretraining process ran for {0} minutes".format((end_time - start_time) / 60))
    
    def finetuning(self, sess, X, y, X_valid, y_valid, training_epochs=2400, _epsilon = 0.25, _batch_size = 280, display_step=5):
        '''
        Finetuing the network
        '''

        num_examples = X.shape[0] # X.shape[0] 
        print("\nStart finetuning...\n")
        start_time = timeit.default_timer()
        LapNoise = dpAutoEncoder.generateNoise(n_in = 248, epsilon = _epsilon, data_size = 26583, test = False); #Add Laplace noise in training
        for epoch in range(training_epochs):
            #avg_cost = 0.0
            # change X.shape to trainSet 
            batch_num = int(math.ceil( num_examples/ _batch_size)) # The number of batch per epoch

            for i in range(batch_num):

                x_batch, y_batch = next_batch(X, y, i, _batch_size, num_examples)
                x = [[0.0 for colum in range(x_batch.shape[1])] for row in range(x_batch.shape[0])]
                # y = []
                for row in range(x_batch.shape[0]):
                    for colum in range(x_batch.shape[1]):
                        x[row][colum]= x_batch[row][colum][0]
                # for row in range(y_batch.shape[0]):
                #     for colum in range(y_batch.shape[1]):
                #         x[row][colum]= y_batch[row][colum][0]


                # training
                sess.run(self.train_op, feed_dict={self.x: x, self.y: y_batch, self.LaplaceNoise: LapNoise})
                # print("\tEpoch {0} \tbatch {1} \t test accuacy: \t {2}".format(epoch, i, test_acc))


                # print out the average cost
            # if epoch % display_step == 0:
                LapNoise_test = dpAutoEncoder.generateNoise(n_in = 248, epsilon = _epsilon, data_size = 26583, test = True); #Do not add noise when testing
                x2 = [[0.0 for colum in range(X_valid.shape[1])] for row in range(X_valid.shape[0])]
                acc = [0.0 for j in range(batch_num)]
                for row in range(X_valid.shape[0]):
                    for colum in range(X_valid.shape[1]):
                        x2[row][colum] = X_valid[row][colum][0]

                val_acc = sess.run(self.accuracy, feed_dict={self.x: x2,self.y: y_valid, self.LaplaceNoise: LapNoise_test})
                acc[i] = val_acc
                ave= [0.0 for i in range(training_epochs)]
                print("\tEpoch {0} \tbatch {1} \t validation accuacy: \t {2}".format(epoch, i, val_acc))
                if i == batch_num - 1:
                    ave[epoch] = sum(acc)/len(acc)
                    print("\tAverage Epoch {0} \t test accuacy: \t {2}".format(epoch, ave[epoch]))
                    print (ave)
                print(val_acc)

        end_time = timeit.default_timer()
        print("\nThe finetuning process ran for {0} minutes".format((end_time - start_time) / 60))

def main(dataset, batch_size):
    # mnist examples
    # sound 16000 output 31 hidden_layer_sizes 
    # _batch_size = 280 sound 128
    df, test_df, X, y, X_test = prepare_data(dataset=dataset)
    input_shape = X[0].shape
    print("========== input shape is : {} ===========".format(input_shape))
    folds = get_folds()
    num_folds = len(folds)

    for i, (train_indices, valid_indices) in enumerate(folds):
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]
        
        dpn = dpAutoEncoder(n_in=16000, n_out=31, hidden_layers_sizes=[248, 1024, 248], epsilon = 8.0, _batch_size = 128, finetuneLR = 1e-3)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        # set random_seed
        tf.set_random_seed(seed=1111)
        # dpn.pretrain(sess, X_train, y_train)
        # dpn.finetuning(sess, _epsilon = dpn.epsilon, _batch_size = dpn.batch_size, trainSet=sound)
        # dpn.pretrain(sess, X_train=mnist)
        dpn.finetuning(sess, X_train, y_train, X_valid, y_valid, _epsilon = dpn.epsilon, _batch_size = dpn.batch_size)


if __name__ == '__main__':
    """
    usage:
        python dpautoencoder.py rawwav 128
    """
    # assert len(sys.argv) == 2, "need specify dataset and batch size"
    dataset, batch_size = sys.argv[1], sys.argv[2]
    main(dataset, batch_size)
