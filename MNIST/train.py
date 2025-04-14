import numpy as np
import tensorflow as tf
from model.model import create_model

#利用load_data()函数从MINST中获取数据并返回两个元组训练图像和标签及测试图像和标签
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

#图像归一化处理，将0-255像素转化为0-1区间，便于快速收敛
x_train,x_test=x_train/255.0,x_test/255.0
#对标签值进行编码（向量值），在MNIST手写数字识别中需要识别的数字为0-9所以编码为10
y_train = tf.kears.utils.to.categorical(y_train,10)
y_test = tf.kears.utils.to.categorical(y_test,10)

#创建和训练模型
model = create_model()
history = model.fit(x_train , y_train,epochs = 5 , validation_data = (x_test,y_test))

#保存模型
model.save('minst_model.h5')
