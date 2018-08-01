import tensorflow as tf
import numpy as np

#
# XOR example
#
# 0,0 => 0
# 0,1 => 1
# 1,0 => 1
# 1,1 => 0

XOR_X = np.array([[0,0],[0,1],[1,0],[1,1]])
XOR_Y = np.array([[0],  [1],  [1],  [0]])

#
# Build model graph
#
n_input_features = 2
n_hidden_nodes = 2
n_output_features = 1

# input/output
x = tf.placeholder(tf.float32, shape=(None,n_input_features), name='input')
y = tf.placeholder(tf.float32, shape=(None,n_output_features), name='correct_output')

# hidden layer
w_hidden = tf.Variable(tf.random_uniform([n_input_features,n_hidden_nodes], -1, 1), name="weights_hidden")
b_hidden = tf.Variable(tf.random_uniform([n_hidden_nodes], -1, 1), name="bias_hidden")
layer1 = tf.sigmoid(tf.add(tf.matmul(x,w_hidden), b_hidden))

#  output layer
w = tf.Variable(tf.random_uniform([n_hidden_nodes,n_output_features], -1, 1), name="weights")
b = tf.Variable(tf.random_uniform([n_output_features], -1, 1), name="bias")
output = tf.sigmoid(tf.add(tf.matmul(layer1, w), b))


#
# Train model
#

# Inefficient, but simplest cost function
cost = tf.reduce_mean(tf.square(y - output))
# Try better cost function: Average Cross Entropy
#cost = - tf.reduce_mean( (y * tf.log(output)) + (1 - y) * tf.log(1.0 - output)  )

# Gradient descent with learning rate=0.1
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

# Run training
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(50000):
    sess.run(optimizer, feed_dict={x: XOR_X, y: XOR_Y})

# Evaluate model
out = sess.run(output, feed_dict={x: XOR_X, y: XOR_Y})
print(out)

cost = sess.run(cost, {x: XOR_X, y: XOR_Y})
print("Final cost: " + str(cost))
