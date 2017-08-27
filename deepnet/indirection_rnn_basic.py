import numpy as np
import tensorflow as tf
from sentenceSets import sentenceSets

## Source: https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767

##################################################################################################
## I'm going to try this first by not encoding the role at the time of word presentation
## The NN is shown three fillers, then prompted with a role on the final timestep
## We are looking for it to match the filler from the timestep that matches the role
##################################################################################################

## Initialize settings
num_epochs = 500
state_size = 3
nroles = 3
nfillers = 10
train_set_size = 200
test_set_size = 100

## Set random seed
np.random.seed(1)

## Create placeholders for the sentence training sets
batchX_placeholder = tf.placeholder(tf.float32, [train_set_size, nroles+1])
batchY_placeholder = tf.placeholder(tf.int32, [train_set_size, nroles+1])

## Create starting state placeholder
init_state = tf.placeholder(tf.float32, [train_set_size, state_size])

## Create variables for the weights and biases for cell state and NN output
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, nfillers),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,nfillers)), dtype=tf.float32)

## Split the batch data into adjacent time-steps 
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)
labels = labels_series[-1]

## Forward pass
## Here we are building the model; it is not actually run at this point
## What we actually want to do is calculate the sum of two affine transforms: 
##   current_input * Wa + current_state * Wb
## By concatenating the two tensors we only use one matrix multiplication. 
## The addition of the bias b is broadcasted on all samples in the batch
current_state = init_state
for current_input in inputs_series:
    ## Prepend input column to state matrix
    current_input = tf.reshape(current_input, [train_set_size, 1]) ## Trainspose input
    input_and_state_concatenated = tf.concat([current_input, current_state], 1)

    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)
    current_state = next_state
state = current_state

## Here we get the softmax predicted class for each sentence
logits = tf.matmul(state, W2) + b2
predictions = tf.nn.softmax(logits)

## The mean loss is used for gradient descent
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []

    for epoch_idx in range(num_epochs):

        ## Choose test
        train_set, test_set = sentenceSets.standardGeneralization(3,10,200,100)
        #train_set, test_set = sentenceSets.spuriousAnticorrelation(3,10,200,100)
        #train_set, test_set = sentenceSets.fullCombinatorial(3,10,200,100)

        ## debug
        #print(train_set[0:9,])

        _current_state = np.zeros((train_set_size, state_size))

        ## Choose random testing roles from training set
        queries = np.random.randint(nroles,size=train_set_size)
        answers = [train_set[i,queries[i]] for i in range(train_set_size)]

        x = y = train_set
        x = np.vstack((x.T,np.asarray(queries))).T
        y = np.vstack((y.T,np.asarray(answers))).T

        batchX = x
        batchY = y 

        _total_loss, _train_step, _current_state, _predictions = sess.run(
            [total_loss, train_step, current_state, predictions],
            feed_dict={
                batchX_placeholder:batchX,
                batchY_placeholder:batchY,
                init_state:_current_state
            })

        loss_list.append(_total_loss)

        if epoch_idx % 10 == 0:
            print("Epoch", epoch_idx, "Loss", _total_loss)
            #print(train_set[0,])
            #print(x[0,])
            #print(_predictions[0,])
