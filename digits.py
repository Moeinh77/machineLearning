import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

learn = tf.contrib.learn
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = learn.datasets.load_dataset('mnist')
data = mnist.train.images
labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)

max_examples = 6000
data = data[:max_examples]  # only take first 'max_examples' items
labels = labels[:max_examples]

np.set_printoptions(threshold=np.nan)

print(data[0])


def display(i):
    img = test_data[i]
    plt.title('Example %d. Label: %d' % (i, test_labels[i]))
    plt.imshow(img.reshape((28, 28)), cmap=plt.cm.gray_r)


display(0)
print(len(data[0]))

#f = open("784.txt", 'w')
import codecs
with codecs.open("784.txt",'w',encoding='utf8') as f:
        f.write(data[0])
