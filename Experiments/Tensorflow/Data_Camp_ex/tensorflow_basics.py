import tensorflow as tf
a = tf.constant(2.0)
b = tf.constant(3.0)
c = a * b


sess = tf.Session()
sess.run(c)

#6.0

# Creating placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Assigning addition operation w.r.t. a and b to node add
add = a + b

# Create session object
sess = tf.Session()

# Executing add by passing the values [1, 3] [2, 4] for a and b respectively
output = sess.run(add, {a: [1,3], b: [2, 4]})
print('Adding a and b:', output)
print('Datatype:', output.dtype)

#Adding a and b: [3. 7.]
#Datatype: float32

#Variables are defined by providing their initial value and type
variable = tf.Variable([0.9,0.7], dtype = tf.float32)

#variable must be initialized before a graph is used for the first time.
init = tf.global_variables_initializer()
sess.run(init)

#Convolutional Neural Network (CNN) in TensorFlow

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
%matplotlib inline
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

data = input_data.read_data_sets('data/fashion',one_hot=True,\
                                 source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/')

# Shapes of training set
print("Training set (images) shape: {shape}".format(shape=data.train.images.shape))
print("Training set (labels) shape: {shape}".format(shape=data.train.labels.shape))

# Shapes of test set
print("Test set (images) shape: {shape}".format(shape=data.test.images.shape))
print("Test set (labels) shape: {shape}".format(shape=data.test.labels.shape))


#Training set (images) shape: (55000, 784)
#Training set (labels) shape: (55000, 10)
#Test set (images) shape: (10000, 784)
#Test set (labels) shape: (10000, 10)

# Create dictionary of target classes
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(data.train.images[0], (28,28))
curr_lbl = np.argmax(data.train.labels[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(data.test.images[0], (28,28))
curr_lbl = np.argmax(data.test.labels[0,:])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

#Text(0.5, 1.0, '(Label: Ankle boot)')

data.train.images[0][500:]

array([0.40784317, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.03921569, 0.9568628 ,
       0.8588236 , 0.9803922 , 0.80392164, 0.7803922 , 0.8196079 ,
       0.79215693, 0.8196079 , 0.82745105, 0.7411765 , 0.83921576,
       0.8078432 , 0.8235295 , 0.7843138 , 0.8313726 , 0.6039216 ,
       0.94117653, 0.81568635, 0.8588236 , 0.54901963, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.08235294, 1.        , 0.8705883 , 0.9333334 ,
       0.72156864, 0.8235295 , 0.75294125, 0.8078432 , 0.8196079 ,
       0.8235295 , 0.7411765 , 0.8352942 , 0.82745105, 0.8196079 ,
       0.75294125, 0.8941177 , 0.60784316, 0.8862746 , 0.9333334 ,
       0.9450981 , 0.6509804 , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.14509805,
       0.9607844 , 0.8862746 , 0.9450981 , 0.5882353 , 0.7725491 ,
       0.7411765 , 0.8000001 , 0.8196079 , 0.8235295 , 0.7176471 ,
       0.8352942 , 0.8352942 , 0.78823537, 0.72156864, 0.8431373 ,
       0.57254905, 0.8470589 , 0.92549026, 0.882353  , 0.6039216 ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.227451  , 0.93725497, 0.89019614,
       1.        , 0.61960787, 0.7568628 , 0.76470596, 0.8000001 ,
       0.8196079 , 0.8352942 , 0.7058824 , 0.8117648 , 0.85098046,
       0.7803922 , 0.7607844 , 0.82745105, 0.61960787, 0.8588236 ,
       0.92549026, 0.8470589 , 0.5921569 , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.26666668, 0.91372555, 0.8862746 , 0.95294124, 0.54509807,
       0.7843138 , 0.7568628 , 0.80392164, 0.8235295 , 0.81568635,
       0.7058824 , 0.80392164, 0.8313726 , 0.7960785 , 0.7686275 ,
       0.8470589 , 0.6156863 , 0.7019608 , 1.        , 0.8470589 ,
       0.60784316, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.31764707, 0.882353  ,
       0.87843144, 0.82745105, 0.5411765 , 0.8588236 , 0.7254902 ,
       0.78823537, 0.8352942 , 0.8117648 , 0.7725491 , 0.8862746 ,
       0.8313726 , 0.7843138 , 0.74509805, 0.8431373 , 0.7176471 ,
       0.3529412 , 1.        , 0.82745105, 0.5764706 , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.35686275, 0.8235295 , 0.90196085, 0.61960787,
       0.44705886, 0.80392164, 0.73333335, 0.81568635, 0.8196079 ,
       0.8078432 , 0.7568628 , 0.8235295 , 0.82745105, 0.8000001 ,
       0.76470596, 0.8000001 , 0.70980394, 0.09019608, 1.        ,
       0.8352942 , 0.61960787, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.34117648,
       0.80392164, 0.909804  , 0.427451  , 0.6431373 , 1.        ,
       0.83921576, 0.87843144, 0.8705883 , 0.8235295 , 0.7725491 ,
       0.83921576, 0.882353  , 0.8705883 , 0.82745105, 0.86274517,
       0.85098046, 0.        , 0.9176471 , 0.8470589 , 0.6627451 ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.36078432, 0.8352942 , 0.909804  ,
       0.57254905, 0.01960784, 0.5254902 , 0.5921569 , 0.63529414,
       0.6666667 , 0.7176471 , 0.7137255 , 0.6431373 , 0.6509804 ,
       0.69803923, 0.63529414, 0.6117647 , 0.38431376, 0.        ,
       0.94117653, 0.882353  , 0.8235295 , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.16862746, 0.6431373 , 0.8078432 , 0.5529412 , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.49803925, 0.4901961 ,
       0.29803923, 0.        , 0.        , 0.        ], dtype=float32)
       
np.max(data.train.images[0])

#1.0

np.min(data.train.images[0])

#0.0

# Reshape training and testing image
train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1,28,28,1)

train_X.shape, test_X.shape

#((55000, 28, 28, 1), (10000, 28, 28, 1))

train_y = data.train.labels
test_y = data.test.labels

train_y.shape, test_y.shape

#((55000, 10), (10000, 10))

training_iters = 10
learning_rate = 0.001
batch_size = 128


# MNIST data input (img shape: 28*28)
n_input = 28

# MNIST total classes (0-9 digits)
n_classes = 10


#both placeholders are of type float
x = tf.placeholder("float", [None, 28,28,1])
y = tf.placeholder("float", [None, n_classes])


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')
    
    
    weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(128,n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}


def conv_net(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
    
    
pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Here, you check whether the index of the maximum value of the predicted image is equal to the actual labeled image. And both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Initializing the variables
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_X)//batch_size):
            batch_x = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={x: batch_x,
                                                              y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()
    
    
plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
plt.plot(range(len(train_loss)), test_loss, 'r', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()

#<Figure size 432x288 with 0 Axes>

plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
plt.plot(range(len(train_loss)), test_accuracy, 'r', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.legend()
plt.figure()
plt.show()

#<Figure size 432x288 with 0 Axes>

#The first step is to load the train folder using Python's built-in glob module and then read the labels.csv using the Pandas library.

import glob
import pandas as pd

imgs = []
label = []

data = glob.glob('train/*')

len(data)

#50000

labels_main = pd.read_csv('trainLabels.csv')

labels_main.head(5)

labels = labels_main.iloc[:,1].tolist()


conversion = {'airplane':0,'automobile':1,'bird':2,'cat':3, 'deer':4, 'dog':5, 'frog':6,\
              'horse':7, 'ship':8, 'truck':9}

num_labels = []

num_labels.append([conversion[item] for item in labels])

num_labels = np.array(num_labels)

num_labels

#array([[6, 9, 9, ..., 9, 1, 1]])

from keras.utils import to_categorical

#Using TensorFlow backend.

label_one = to_categorical(num_labels)

label_one = label_one.reshape(-1,10)

label_one.shape

#(50000, 10)

import cv2

for i in data:
    img = cv2.imread(i)
    if img is not None:
        imgs.append(img)

train_imgs = np.array(imgs)
train_imgs.shape

#(50000, 32, 32, 3)

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(train_imgs[0], (32,32,3))
curr_lbl = labels_main.iloc[0,1]
plt.imshow(curr_img)
plt.title("(Label: " + str(curr_lbl) + ")")

# Display the second image in training data
plt.subplot(122)
curr_img = np.reshape(train_imgs[1], (32,32,3))
curr_lbl = labels_main.iloc[1,1]
plt.imshow(curr_img)
plt.title("(Label: " + str(curr_lbl) + ")")

Text(0.5, 1.0, '(Label: truck)')


train_images = train_imgs / np.max(train_imgs)

np.max(train_images), np.min(train_images)

#(1.0, 0.0)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_images, label_one, test_size=0.2, random_state=42)

train_X = X_train.reshape(-1, 32, 32, 3)
test_X = X_test.reshape(-1, 32, 32, 3)