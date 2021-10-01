import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

'''
线性函数为y=2x+1
'''

np.random.seed(5)       #设置随机数种子
train_epochs = 10       #迭代次数(训练轮数)
learning_rate = 0.01    #学习率
step = 0            #记录训练步数
loss_list = []      #用于保存loss值的列表
display_step = 10   #控制训练过程中数据显示的频率，不是超参数

#定义模型函数
def model(x,w,b):
    return tf.multiply(x, w)+b

#定义均方差损失函数
def loss(x,y,w,b):
    err = model(x, w, b) - y                  #计算模型预测值和标签值的差异
    squared_err = tf.square(err)            #求平方，得出方差
    return tf.reduce_mean(squared_err)      #求均值，得出均方差

'''
    采用均方差作为损失函数（第二种方法）
    pred = model(x,w,b)  pred是预测值，前向计算
    loss = tf.reduce_mean(tf.square(y-pred))
'''

#计算样本数据[x，y]在参数[w，b]点上的梯度
def grad(x,y,w,b):
    with tf.GradientTape() as tape:  #上下文管理器封装需要求导的计算步骤，
        loss_ = loss(x, y, w, b)
    return tape.gradient(loss_, [w, b])     #************

#对一个数进行预测
def pre_number(x_test):
    predict = model(x_test, w.numpy(), b.numpy())
    print("预测值：{}".format(predict))

    target = 2 * x_test + 1.0
    print("目标值：{}".format(target))

#用随机数设置x，y
x_data = np.linspace(-1,1,100)
y_data = 2*x_data+1.0+np.random.randn(*x_data.shape)*0.4  #用*把x.shape设置为实参

#画出随机生成数据的散点图
plt.scatter(x_data,y_data)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Figure:Training Data")

#画出我们想要学习到的线性函数 y=2x+1
#plt.plot(x_data, 2*x_data+1.0,color='red',linewidth=3)

#构建线性函数的斜率，变量w
w = tf.Variable(1.0)

#构建线性函数的截距，变量b
b = tf.Variable(0.0)

for epoch in range(train_epochs):
    for xs, ys in zip(x_data, y_data):
        loss_ = loss(xs, ys, w, b)     #计算损失
        loss_list.append(loss_)     #保存本次损失计算结果

        delta_w, delta_b = grad(xs, ys, w, b)   #计算当前[w，b]的梯度
        change_w = delta_w * learning_rate  #计算变量w需要调整的量
        change_b = delta_b * learning_rate  #计算变量b需要调整的量
        w.assign_sub(change_w)              #变量w更新后的量
        b.assign_sub(change_b)              #变量b更新后的量

        step = step + 1
        if step % display_step == 0:
            print("Trainging Epoch:{},Step:{},loss={}".format(epoch+1, step, loss_))
    plt.plot(x_data, w.numpy() * x_data + b.numpy())

print("w: ", w.numpy())
print("b: ", b.numpy())

pre_number(5.0)
plt.show()