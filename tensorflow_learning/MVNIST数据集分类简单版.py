#检测手写数字识别的准确率

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data              #与手写数字相关的工具包
#载入数据集（会自动下载）
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)            #"MNIST_data"为下载数据的保存路径，这个是文件名字，默认和py程序路径一样

#每个批次的大小
batch_size = 100                                                        #训练不是一次一张图片，而是一次一批次图片
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size                        #"//"是整除的意思

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])                               #定义的批次是100，所以None就为100,784=28X28(一张图片是28X28的，这里换成一维向量)
y = tf.placeholder(tf.float32,[None,10])                                #数字是0~9，一共10个数字

#创建一个简单的神经网络
W = tf.Variable(tf.zeros([784,10]))                                     #权值
b = tf.Variable(tf.zeros([10]))                                         #偏置值
prediction = tf.nn.softmax(tf.matmul(x,W)+b)                            #得到概率组

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))      #就是求均方误差
#使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(3).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))   #argmax返回最大那个概率对应的标签
                                                                        #tf.equal就是判断真实样本的标签和最大概率的标签是否一致
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))       #tf.cast转化后true就为1.0，false就为0;再求平均值

#进行训练
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(30):                                             #迭代周期
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)      #每次获得batch_size张图片，图片数据保存在batch_xs，图片标签保存在batch_ys
                                                                        #mnist.train.next_batch就是下一次执行就会获得另外batch_size张图片，依次类推，运行n_batch次
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})        #测试准确率
                                                                                            #把测试集图片和标签喂进去
        print("Iter" + str(epoch) + ",Testing Accuracy " + str(acc))
