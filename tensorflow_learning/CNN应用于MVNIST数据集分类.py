#用CNN进行手写数字识别
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data              #导入手写数字相关的工具包
#载入数据集（会自动下载）
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#每个批次的大小
batch_size = 100                                                        #训练不是一次一张图片，而是一次一批次图片
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size                        #"//"是整除的意思

#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)                     #生成一个截断的正态分布
    return tf.Variable(initial)

#初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x,W):
    #x input tensor of shape'[batch, in_height, in_width, in_channels]'
    #W filter(过滤器) / kernel(内核) tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #'strides[0] = strides[3] = 1'. strides[1]代表x方向的步长，strides[2]代表y方向的步长
    #padding: A 'string' from: '"SAME","VALID"'
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')           #1X1步长

#池化层
def max_pool_2x2(x):
    #ksize [1,x,y,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')   #2X2步长

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])                               #定义的批次是100，所以None就为100,784=28X28(一张图片是28X28的，这里换成一维向量)
y = tf.placeholder(tf.float32,[None,10])                                #数字是0~9，一共10个数字

#改变x的格式转化为4D的向量[batch, in_height, in_width, in_channels]
x_image = tf.reshape(x,[-1,28,28,1])                                    #-1表示不指定批次大小，自动识别程序里的（这里定义了100）、28*28图像、1表示黑白图像

#初始化第一个卷积层的权值和偏置
W_conv1 = weight_variable([5,5,1,32])                                   #5*5的采样窗口，32个卷积核从1个平面抽取特征
b_conv1 = bias_variable([32])                                           #每一个卷积核一个偏置值,32个卷积核就有32个

#把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)                                         #进行max-pooling，2X2步长

#初始化第二个卷积层的权值和偏置
W_conv2 = weight_variable([5,5,32,64])                                   #5*5的采样窗口，32是因为签名第一个卷积层初始化后为32个卷积核，64个卷积核从1个平面抽取特征
b_conv2 = bias_variable([64])                                           #每一个卷积核一个偏置值

#把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)                                         #进行max-pooling，2X2步长

#28*28的图片第一次卷积后还是28*28，第一次池化后变为14*14
#第二次卷积后为14*14，第二次池化后变为7*7
#经过上面操作后得到64张7*7的平面

#初始化第一个全连接层的权值
W_fc1 = weight_variable([7*7*64,1024])                                  #上一层有7*7*64个神经元，全连接层有1024个神经元
b_fc1 = bias_variable([1024])                                           #1024个节点

#把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
#求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#keep_prob用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#初始化第二个全连接层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

#计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果存放在一个布尔列表中
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))       #argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))       #tf.cast转化后true就为1.0，false就为0;再求平均值

#进行训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):                                             #迭代周期
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)      #每次获得batch_size张图片，图片数据保存在batch_xs，图片标签保存在batch_ys
                                                                        #mnist.train.next_batch就是下一次执行就会获得另外batch_size张图片，依次类推，运行n_batch次
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})        #测试准确率
                                                                                            #把测试集图片和标签喂进去
        print("Iter" + str(epoch) + ",Testing Accuracy " + str(acc))






