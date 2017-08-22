import tensorflow as tf
a=tf.constant([2.0,3.0,4.0],shape=[1,3],name='a')
b=tf.constant([11.0,12.0,13.0],shape=[1,3],name='b')

with tf.name_scope("mean_x"):      
        x=tf.reduce_mean(a)
        sess=tf.Session()
        print(sess.run(x))

with tf.name_scope("mean_y"):
        y=tf.reduce_mean(b)
        sess=tf.Session()
        print(sess.run(y))
        
with tf.name_scope("var"):
        d=tf.subtract(a,x)
        sess=tf.Session()
        print(sess.run(d ))

        e=tf.square(d)
        f=tf.reduce_sum(e)
        sess=tf.Session()
        print(sess.run(f))

with tf.name_scope("covariance"):
    g=tf.subtract(b,y)
    sess=tf.Session()
    g=tf.multiply(d,g)
    h=tf.reduce_sum(g)
    print(sess.run(h))

with tf.name_scope("value_of_c"):
    j=tf.divide(h,f)
    print(sess.run(j))


with tf.name_scope("value_m"):
    writer=tf.summary.FileWriter('/home/d2/vish_mllab',sess.graph)
    i=tf.multiply(j,x)
    k=tf.subtract(y,i)
    print(sess.run(j))