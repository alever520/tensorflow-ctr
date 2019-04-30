import tensorflow as tf

feature1 = {
    "r": tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9]]),
}

tmp = tf.Print(feature1["r"], [feature1["r"]], summarize=6)

#feature1["l"] = tf.stack([feature1["r"][0]] * feature1["r"].shape[0], axis=0)
feature1["l"] = tf.tile([feature1["r"][0]], [tf.shape(feature1["r"])[0],1])

res = tf.Print(feature1["l"], [feature1["l"]])

sess = tf.Session()

sess.run([res, tmp, tf.shape(feature1["r"])[0]])


import tensorflow as tf
import numpy as np
x = tf.placeholder(tf.float32, shape=(None,))
y = tf.stack([x[0]] * tf.shape(x)[0], axis=0)

with tf.Session() as sess:
    rand_array = np.random.rand(10,)
    print(sess.run([x, y], feed_dict={x: rand_array}))





import tensorflow as tf
import numpy as np
x = tf.placeholder(tf.float32, shape=(None,3,3))
multiples = tf.concat([[tf.shape(x)[0]], tf.tile([1], [tf.rank(x)-1])], axis=0)
y = tf.tile([x[0]], multiples)

with tf.Session() as sess:
    rand_array = np.random.rand(5,3,3)
    print(sess.run([x, y], feed_dict={x: rand_array}))


