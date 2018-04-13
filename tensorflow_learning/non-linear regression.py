import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt                     #matplotlib 画图工具包

#使用numpy生成200个随机点
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]    #在-0.5到0.5内生成均匀分布的200个点,存在：那里
                                                    #np.newaxis相当于加了一个维度,矩阵(200,1)
noise = np.random.normal(0,0.02,x_data.shape)       #生成一些干扰项，形状和x_data的一样
y_data=np.square(x_data) + noise                    #得到大致和U一样的图形，但会有很多混乱点

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,1])             #tf.placeholder(dtype, shape=None, name=None)
y = tf.placeholder(tf.float32,[None,1])             #dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
                                                    #shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3]；
                                                    # [None, 1]表示列是1，行不定
                                                    # x为输入值，y为预测值
#定义神经网络中间层(隐藏层)
Weights_L1 = tf.Variable(tf.random_normal([1,10]))  #权重,输入层1个神经元，中间层10个神经元
biases_L1 = tf.Variable(tf.zeros([1,10]))           #偏置值，输入层1个神经元，中间层10个神经元
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1  #就是有钊说的那个y=wx+b  (w是权重，b是偏置值),信号总和
L1 = tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))  #权重,中间层10个神经元,输出层1个神经元
biases_L2 = tf.Variable(tf.zeros([1,1]))            #偏置值，输入层1个神经元，输出层1个神经元
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)               #输出结果，预测的结果

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:                          #绘画
    #变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):                           #训练2000次
        sess.run(train_step,feed_dict={x:x_data,y:y_data})      #x传入x_data,y传入y_data

    #获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)                      #打印样本点
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()                                      #显示图