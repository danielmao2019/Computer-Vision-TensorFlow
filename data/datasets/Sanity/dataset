import tensorflow as tf

a=tf.random.uniform(shape=(1,)+model.layers[0].input_shape[0][1:])
b=tf.ones(shape=(1,)+model.layers[0].input_shape[0][1:], dtype=tf.float32)
c=tf.random.normal(shape=(1,)+model.layers[0].input_shape[0][1:])

data=tf.data.Dataset.from_tensor_slices(a,b,c)
