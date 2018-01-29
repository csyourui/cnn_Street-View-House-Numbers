import tensorflow as tf
from svhn import SVHN
import matplotlib.pyplot as plt
import numpy as np

# Parameters
learning_rate = 0.001
iterations = 50000
batch_size = 50
display_step = 1000

# Network Parameters
dropout = 0.8

normalization_offset = 0.0  # beta
normalization_scale = 1.0  # gamma
normalization_epsilon = 0.001  # epsilon


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(1.0, shape=shape))


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 读取数据集
svhn = SVHN(10, use_extra=False, gray=False)

# 建立模型
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])




# Batch normalization

mean, variance = tf.nn.moments(X, [1, 2, 3], keep_dims=True)
x = tf.nn.batch_normalization(X, mean, variance, normalization_offset, normalization_scale,normalization_epsilon)

#第一层卷积
W_conv1 = weight_variable([5, 5, 3, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1 )
h_pool1 = max_pool(h_conv1)


#第二层卷积
W_conv2 = weight_variable([5, 5, 16, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2 )
h_pool2 = max_pool(h_conv2)


#第三层卷积
W_conv3 = weight_variable([5, 5, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3 )
h_pool3 = max_pool(h_conv3)


# Fully Connected Layer
shape = h_pool3.get_shape().as_list()
h_pool2_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 64])

W_fc1 = weight_variable([32 // 8 * 32 // 8 * 64, 128])
b_fc1 = bias_variable([128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout Layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Build the graph for the deep net
W_fc2 = weight_variable([128, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = y_conv))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())

    # Variables useful for batch creation
    start = 0
    end = 0

    # The accuracy and loss for every iteration in train and test set
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    for i in range(iterations):
        # Construct the batch
        if start == svhn.train_examples:
            start = 0
        end = min(svhn.train_examples, start + batch_size)

        batch_x = svhn.train_data[start:end]
        batch_y = svhn.train_labels[start:end]

        start = end

        # Run the optimizer
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})

        if (i + 1) % display_step == 0 or i == 0:
            _accuracy, _cost = sess.run([accuracy, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
            print("Step: {0:6d}, Training Accuracy: {1:5f}, Batch Loss: {2:5f}".format(i + 1, _accuracy, _cost))
            train_accuracies.append(_accuracy)
            train_losses.append(_cost)

    # Test the model by measuring it's accuracy



    test_iterations = svhn.test_examples / batch_size + 1
    test_iterations = int(test_iterations)
    for i in range(test_iterations):
        batch_x, batch_y = (svhn.test_data[i * batch_size:(i + 1) * batch_size],
                            svhn.test_labels[i * batch_size:(i + 1) * batch_size])
        _accuracy, _cost = sess.run([accuracy, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
        test_accuracies.append(_accuracy)
        test_losses.append(_cost)
    print("Mean Test Accuracy: {0:5f}, Mean Test Loss: {1:5f}".format(np.mean(test_accuracies), np.mean(test_losses)))

    # Plot batch accuracy and loss for both train and test sets
    plt.style.use("ggplot")
    fig, ax = plt.subplots(2, 2)

    # Train Accuracy
    ax[0, 0].set_title("Train Accuracy per Batch")
    ax[0, 0].set_xlabel("Batch")
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 0].set_ylim([0, 1.05])
    ax[0, 0].plot(range(0, iterations + 1, display_step), train_accuracies, linewidth=1, color="darkgreen")

    # Train Loss
    ax[0, 1].set_title("Train Loss per Batch")
    ax[0, 1].set_xlabel("Batch")
    ax[0, 1].set_ylabel("Loss")
    ax[0, 1].set_ylim([0, max(train_losses)])
    ax[0, 1].plot(range(0, iterations + 1, display_step), train_losses, linewidth=1, color="darkred")

    # TestAccuracy
    ax[1, 0].set_title("Test Accuracy per Batch")
    ax[1, 0].set_xlabel("Batch")
    ax[1, 0].set_ylabel("Accuracy")
    ax[1, 0].set_ylim([0, 1.05])
    ax[1, 0].plot(range(0, test_iterations), test_accuracies, linewidth=1, color="darkgreen")

    # Test Loss
    ax[1, 1].set_title("Test Loss per Batch")
    ax[1, 1].set_xlabel("Batch")
    ax[1, 1].set_ylabel("Loss")
    ax[1, 1].set_ylim([0, max(test_losses)])
    ax[1, 1].plot(range(0, test_iterations), test_losses, linewidth=1, color="darkred")

    plt.show()
