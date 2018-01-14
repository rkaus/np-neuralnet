from __future__ import division
import numpy as np
import sys

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes=2, output_nodes=1, learning_rate=0.1,iterations=100):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                       (self.input_nodes, self.hidden_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))

        # Iteration storage
        self.iterations = 100
        self.losses = {'train':[], 'test':[]}

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.
       
            Args
            ----
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        '''

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        ''' 
            Args
            ----
            X: features batch
        '''
        hidden_inputs = np.dot(X,self.weights_input_to_hidden) 
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output)
        final_outputs = final_inputs # signals no further activation necessary
       
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        '''
            Args
            ----
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        output_error_term = error * 1.0
       
        hidden_error = np.dot(self.weights_hidden_to_output,error)       
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
      
        # Weight step (hidden to output)
        delta_weights_i_h += hidden_error_term * X[:,None]
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights using gradient descent
        
            Args
            ----
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features
       
            Args
            ----
            features: 1D array of feature values
        '''       
        hidden_inputs = np.dot(features,self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs) # 1x2

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs

    def iterate(self, X_train, X_test, y_train, y_test):
        ''' Train neural network and store loss values

            Args
            ----
            X_train: Training batch of feature values
            X_test:  Test batch of feature values
            y_train: Training batch of target values
            y_test:  Test batch of feature values
        '''
        MSE = lambda x, y: np.mean((x-y)**2)

        for i in range(self.iterations):
            sample = np.random.choice(X_train.index, size=128)
            X_sample = X_train.loc[sample].values               # grab random sample
            y_sample = y_train.loc[sample]['cnt'].values        # of features/targets

            self.train(X_sample,y_sample)

            # Printing out the training progress
            train_loss = MSE(self.run(X_train).T, y_train['cnt'].values)
            test_loss = MSE(self.run(X_test).T, y_test['cnt'].values)
            sys.stdout.write("\rProgress: {:2.1f}".format(100 * i/float(self.iterations)) \
                             + "% ... Training loss: " + str(train_loss)[:5] \
                             + " ... Test loss: " + str(test_loss)[:5])
            sys.stdout.flush()
            
            #Store losses for plotting 
            self.losses['train'].append(train_loss)
            self.losses['test'].append(test_loss)          
  
        sys.stdout.write('\n')