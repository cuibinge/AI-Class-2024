from tensorflow.keras import layers,models

def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),#将28*28的图片展平为784*1的一维向量
        layers.Dense(128, activation='relu'),#全连接层，设置神经元个数为128个，并调用ReLU为激活函数
        layers.Dropout(0.2),#正则化
        layers.Dense(10, activation='softmax'),#输出层，用softmax为激活函数输出概率分布
    ])

    model.compile(optimizer='adam' , loss='categorical_crossentropy' , metrics=['accuracy'])
    return model