import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Training Data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

n_input_features = 1
n_output_features = 1
n_samples = train_X.shape[0]

# Build graph model
x = tf.placeholder(tf.float32, name='input')
y = tf.placeholder(tf.float32, name='correct_output')

w = tf.Variable(tf.random_uniform([n_input_features]), name="weights")
b = tf.Variable(tf.random_uniform([n_output_features]), name="bias")

output = tf.add(tf.multiply(x, w), b)

# cost function
cost = tf.losses.mean_squared_error(y, output)

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# run training session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
cost_history = []
for epoch in range(1000):
    sess.run(optimizer, feed_dict={x:train_X, y:train_Y})
    cost_history.append( sess.run(cost, feed_dict={x:train_X, y:train_Y}) )

print("W=", sess.run(w), "b=", sess.run(b))

# Learning curve
plt.plot(cost_history, label='Learning curve')
plt.legend()
plt.show()

# Plot values & line
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, sess.run(output, feed_dict={x: train_X}), label='Fitted line')
plt.legend()
plt.show()
