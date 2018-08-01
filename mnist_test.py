import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import imput_data

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

#Path to comp graphs
LOGDIR = './graphs'

#Start session
session = tf.Session()

#Hyper parameters
LEARNING_RATE = 0.01
BATCH_SIZE = 1000
EPOCH = 10

# Hidden Layers --- no. of output nodes
HL_1 = 1000
HL_2 = 500

# Other paramters
INPUT_SIZE = 28 * 28
N_CLASSES = 10

with tf.name_scope('input'):
	images = tf.placeholder(tf.float32, [None, INPUT_SIZE], name="images")
	labels = tf.placeholder(tf.float32, [None, N_CLASSES], name="labels")
	
def fc_layer(x, layer, size_out, activation = None): # input to layer, name of the layer, no. of output nodes, activation function used
	with tf.name_scope('layer'):
		size_in = int(x.shape[1])
		W = tf.Variable(tf.ranom_normal([size_in, size_out]), name="weights")
		b = tf.Variable(tf.constant(-1, dtype=tf.float32, shape=[size_out]), name="biases")
		
		wx_plus_b = tf.add(tf.matmul(x, W), b)
		if activvation:
			return activation(wx_plus_b)
		return wx_plus_b
		
		
fc_1 = fc_layer(images, 'fc_1', HL_1, tf.nn.relu)
fc_2 = fc_layer(fc_1, 'fc_2', HL_2, tf.nn.relu)

# to prevent overfitting
dropped = tf.nn.dropped(fc_2, keep_prob=0.9)

# output layer
y = fc_layer(dropped, 'output', N_CLASSES)


	
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels))
	tf.summary.scalar('loss', loss) # used for displaying changes on tensorboard
	
with tf.name_scope('optimizer'):
	train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
	
with tf.name_scope('evalutation'): # for gauging the accuracy of the network
	correct = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
	tf.summary.scalar('accuracy', accuracy) # again, needed for displaying on tensorboard
	
training

