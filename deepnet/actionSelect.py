import numpy as np
import tensorflow as tf

## This program is just a mock-up for training the action selection layer

seed = 1
nfillers = 10

## Set seed for repeatability
tf.set_random_seed(seed)
np.random.seed(seed)

## The bias is constant, not variable
## Variable bias doesn't seem to help
#bias_a = tf.Variable(tf.random_normal([nfillers,1])) 
#bias_a = tf.Variable(tf.ones([nfillers,1],dtype=tf.float32))
bias_a = tf.ones([nfillers,1],dtype=tf.float32)
lrate_a = 0.1

## Create placeholder vectors for input and answers
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

## Create weight vectors, 10x10
## In the real program we'll also have input units for the three roles, so it'll be 13x10
## TODO: this is not unitary; I'm not sure how to make it so
W_a = tf.Variable(tf.random_normal([nfillers,nfillers],mean=0.0,stddev=np.mean(1/nfillers)))

## Create training set, which is just fillers 0-9
## These are both the input and output; we're trying to get a one-to-one mapping for this test
train_set = np.identity(nfillers)

## Build the model
## 'x' will be the state, modeled as a binary feature vector
## We are using gradient descent to minimize squared error
ac_model = tf.nn.softmax(tf.matmul(W_a,x) + bias_a,dim=0)
loss = tf.reduce_sum(tf.square(ac_model-y))
optimizer = tf.train.GradientDescentOptimizer(lrate_a)
train = optimizer.minimize(loss)
num_trials = 1000

## Run the model
with tf.Session() as sess:

    ## Initialize all variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    ## Run as many times as we want to train
    for t in range(num_trials):
        sess.run(train, {x:train_set, y:train_set})

    ## Evaluate training accuracy
    curr_W, curr_loss = sess.run([W_a,loss], {x:train_set, y:train_set})
    print("loss: %s"%(curr_loss))                           ## This shorthand print formatting is sweet

    ## Run a sample trial on the trained model
    f = np.repeat(0.0,nfillers).reshape(nfillers,1)
    f[0] = 1.0
    print(sess.run(ac_model,{x:f}))

