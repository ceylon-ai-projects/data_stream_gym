import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

base_path = "/home/dewmal/WorkingProjects/ceylon_models/trading/trader/"


def read_dataset():
    df = pd.read_csv("{}Sonar.csv".format(base_path))
    print(len(df.columns))
    # Variables
    x = df[df.columns[0:60]].values
    y = df[df.columns[60]]

    # Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y)

    y = encoder.transform(y)
    y = one_hot_encoder(y)

    print(x.shape)
    return (x, y)


def one_hot_encoder(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))

    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# Read the dataset

X, Y = read_dataset()

# Shuffle the dataset to mix up the rows
X, Y = shuffle(X, Y, random_state=1)

# Split data set to train and test
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=415)

# Inspect The shape of train and testing
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)

# Define the important params and variables to work with tensor
learning_rate = 0.3
training_epochs = 100
cost_history = []
n_dim = X.shape[1]
print("n dim", n_dim)
n_class = 2
model_path = "{}model_saved".format(base_path)

# Define the number of hidden layers and number of neurons for each layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])


def multiplayer_perception(x, weights, biases):
    # Hidden layer with RELU Activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # Hidden layer with sigmoid activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # Hidden layer with RELU Activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class])),
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class])),
}

# Initialize all the variables

init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Call your model defined
y = multiplayer_perception(x, weights, biases)

# Define cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)

# Calculate the cost and the accuracy for each epoch

mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
    sess.run(training_steps, feed_dict={x: train_x, y_: train_y})
    cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
    cost_history.append(cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse = sess.run(mse)
    mse_history.append(mse)

    accuracy = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
    accuracy_history.append(accuracy)

    print("Epoch : ", epoch, ' - ', cost, ' cost:', "-MSE:", mse, "- Train Accuraccy:", accuracy)

save_path = saver.save(sess, model_path)
print("Model saved in file: %s" % save_path)

# Plot mse and acc
plt.plot(mse_history, label="MSE")
plt.show()
plt.plot(accuracy_history, label="Acc")
plt.show()
