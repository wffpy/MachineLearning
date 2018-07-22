import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.preprocessing
import os
from Face_detection.Load_data import Train_Test_sets

height = 32
width = 32
channel = 1
save_path = "model/model"

'''
Name : generatebatch()
Parameters : X 

'''
def generatebatch(X,y,n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i * batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = y[start:end]
        yield batch_xs, batch_ys

def weight(shape):
    intial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(intial)

def bias(shape):
    intial = tf.constant(0.1, shape=shape)
    return tf.Variable(intial)



def build_net(sess):

    tf_X = tf.placeholder(dtype=tf.float32,shape = [None, height, width, channel],name="tf_X")
    tf_y = tf.placeholder(dtype=tf.float32,shape = [None,2],name="tf_y")
    keep_prob = tf.placeholder(tf.float32,name="keep_prob")

#     Convolution layer
#     w1 = tf.Variable(tf.random_normal([5,5,channel,4]))
#     b1 = tf.Variable(tf.random_normal([4]))
    w1 = weight([5,5,channel,4])
    b1 = bias([4])
    conv_out1 = tf.nn.relu(tf.nn.conv2d(tf_X,w1,[1,2,2,1],padding="VALID")+b1)

#     Convolution layer2
#     w2 = tf.Variable(tf.random_normal([3,3,4,16]))
#     b2 = tf.Variable(tf.random_normal([16]))
    w2 = weight([3,3,4,16])
    b2 = bias([16])
    conv_out2 = tf.nn.relu(tf.nn.conv2d(conv_out1,w2,[1,2,2,1],padding="VALID")+b2)


#     Convolution layer 3
#     w3 = tf.Variable(tf.random_normal([3,3,16,32]))
#     b3 = tf.Variable(tf.random_normal([32]))
    w3 = weight([3,3,16,32])
    b3 = bias([32])
    conv_out3 = tf.nn.relu(tf.nn.conv2d(conv_out2,w3,[1,1,1,1],padding="VALID")+b3)

    dim = 32*4*4


#     Full Convolution layer
    flat = tf.reshape(conv_out3,[-1,dim])
    # w4 = tf.Variable(tf.random_normal([dim,600]))
    # b4 = tf.Variable(tf.random_normal([600]))
    w4 = weight([dim,600])
    b4 = bias([600])
    fc_out4 = tf.nn.relu(tf.matmul(flat,w4)+b4)

    o4 = tf.nn.dropout(fc_out4,keep_prob)

#   Full Convolution Layer 5
#     w5 = tf.Variable(tf.random_normal([600,2]))
    # b5 = tf.Variable(tf.random_normal([2]))
    w5 = weight([600,2])
    b5 = bias([2])
    pred = tf.nn.softmax(tf.matmul(o4,w5)+b5)
    # pred  = fc5_out

    sess.run(tf.initialize_all_variables())
    return pred,tf_X,tf_y,keep_prob



#    Start a tensorflow session
def start_sess():
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    return sess




def train(sess,pred,tf_X,tf_y,X,y,keep_prob,learn_rate,epsilon,n_epoch,batch_size,save_path):
    # loss = -tf.reduce_mean(tf_y*tf.log(tf.clip_by_value(pred,1e-11,1.0)))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_y*tf.log(pred+1e-10), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate=learn_rate,epsilon=epsilon).minimize(cross_entropy)
    bool_pred =  tf.equal(tf.argmax(pred,1),tf.argmax(tf_y,1))
    accuracy = tf.reduce_mean(tf.cast(bool_pred,tf.float32))

    sess.run(tf.global_variables_initializer())
    # batchs = generatebatch(X,y,y.shape[0],batch_size)

    for epoch in range(n_epoch):
        for batchX,batchy in generatebatch(X,y,y.shape[0],batch_size):
            # train_step.run(session=sess,feed_dict={tf_x:batchX,tf_y:batchy})
            sess.run(train_step,feed_dict={tf_X:batchX,tf_y:batchy,keep_prob:0.5})

        if epoch%10 ==0:
            res = sess.run(accuracy,feed_dict = {tf_X:X,tf_y:y,keep_prob:1})
            print(epoch,res)

    # res_ypred = pred.eval(sess,feed_dict={tf_x:X,tf_y:y})

    if not save_path is None:
        saver = tf.train.Saver()
        tf.add_to_collection("pred",pred)
        # tf.add_to_collection("keep_prob",keep_prob)
        saver.save(sess=sess,save_path=save_path)
        # saver.restore(sess=sess,save_path=save_path)
        print("saved")




def Test(X,y,model_path):

    saver = tf.train.import_meta_graph("model/model.meta")
    graph = tf.get_default_graph()
    tf_X = graph.get_operation_by_name('tf_X').outputs[0]
    tf_y = graph.get_operation_by_name('tf_y').outputs[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
    pred = tf.get_collection('pred')[0]

    bool_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(tf_y, 1))
    accuracy = tf.reduce_mean(tf.cast(bool_pred, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver.restore(sess,tf.train.latest_checkpoint('model/'))
    res = sess.run(accuracy, feed_dict={tf_X: X, tf_y: y, keep_prob:1})
    print("accuracy",res)


def Train_net(save_path):
    Train_set,Test_set = Train_Test_sets()

    X= []
    y = []
    for data in Train_set:
        X.append(data[0])
        y.append(data[1])
        # print(data[1])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(-1,32,32,1)
    # root,dirs,files = os.walk(save_path)

    sess = start_sess()
    pred,tf_X,tf_y,keep_prob = build_net(sess)
    train(sess,pred,tf_X,tf_y,X,y,keep_prob,learn_rate=1e-4,epsilon=1e-8,n_epoch=20,batch_size=100,save_path=save_path)




# Train_net(save_path)

