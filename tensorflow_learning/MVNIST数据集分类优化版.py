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
keep_prob=tf.placeholder(tf.float32)                                    #设置神经元被选中的概率
lr = tf.Variable(0.001, dtype=tf.float32)                               #learning rate，初始学习率为0.001

#创建神经网络

#第一个隐藏层
W1 = tf.Variable(tf.truncated_normal([784,500],stddev=0.1))             #权值
b1 = tf.Variable(tf.zeros([500])+0.1)                                   #偏置值
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)                                     #激活函数是双曲正切函数
L1_drop = tf.nn.dropout(L1,keep_prob)                                   #这里并没有使用drop

#第二个隐藏层
W2 = tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
b2 = tf.Variable(tf.zeros([300])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([300,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop,W3)+b3)                    #得到概率组

#交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))      #求代价值
#训练
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

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
        sess.run(tf.assign(lr,0.001*(0.95**epoch)))                     #0.95的epoch次方，改变学习率
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)      #每次获得batch_size张图片，图片数据保存在batch_xs，图片标签保存在batch_ys
                                                                        #mnist.train.next_batch就是下一次执行就会获得另外batch_size张图片，依次类推，运行n_batch次
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        learning_rate = sess.run(lr)
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})        #测试准确率
                                                                                            #把测试集图片和标签喂进去
        print("Iter" + str(epoch) + ",Testing Accuracy " + str(acc) + ",Learning Rate" + str(learning_rate))
