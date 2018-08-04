import tensorflow as tf

X = tf.placeholder(dtype=tf.float32, shape=(4, 2))
Y = tf.placeholder(dtype=tf.float32, shape=(4, 1))

INPUT_XOR = [[0,0],[0,1],[1,0],[1,1]]
OUTPUT_XOR = [[0],[1],[1],[0]]


learning_rate = 0.01
epochs = 10000

# Hidden Layer
with tf.variable_scope('hidden_layer'):
    h_w = tf.Variable(tf.truncated_normal([2, 2]), name='Weights')
    h_b = tf.Variable(tf.truncated_normal([4, 2]))
    h = tf.nn.relu(tf.matmul(X, h_w) + h_b)

# Output Layer
with tf.variable_scope('output'):
    o_w = tf.Variable(tf.truncated_normal([2, 1]))
    o_b = tf.Variable(tf.truncated_normal([4, 1]))
    Y_estimation = tf.nn.sigmoid(tf.matmul(h, o_w) + o_b)

# Loss function
with tf.variable_scope('cost'):
    cost = tf.reduce_mean(tf.squared_difference(Y_estimation, Y))

# Training Layer
with tf.variable_scope('train'):
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Tensorflow Session
with tf.Session() as session:

    # init session variables
    session.run(tf.global_variables_initializer())

    # log count
    log_count_frac = epochs/10

    for epoch in range(epochs):

        # Training network
        session.run(train, feed_dict={X: INPUT_XOR, Y:OUTPUT_XOR})

        # log training parameters
        if epoch % log_count_frac == 0:
            cost_results = session.run(cost, feed_dict={X: INPUT_XOR, Y:OUTPUT_XOR})
            print(cost_results)

    print("Training Completed !")
    Y_test = session.run(Y_estimation, feed_dict={X:INPUT_XOR})
    print(Y_test)