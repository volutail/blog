---
layout: default
title:  "Learning Keras (1)"
---
# Keras学习笔记（一）
[Keras](https://keras.io)是一个采用python语言编写的高级神经网络API，兼容TensorFlow，CNTK和Theano等深度学习框架。神经网络的搭建在Keras上十分简单，并且过程更加形象，因此在这里记录下Keras框架的学习过程。
## Keras安装
### Python安装
Keras支持的Python版本为2.7-3.6。Python的安装可以访问[Python官网](https://www.python.org/)，并根据自己的系统平台进行选择，推荐使用64位python3.6版本。
### Backend安装
Keras只提供神经网络相关功能的高级接口，底层计算通过“Backend”进行，这里推荐使用TensorFlow作为底层计算框架（由google开发，社区庞大，代码可读性高）。TensorFlow的安装可以通过pip包管理器进行，执行以下命令：  
`pip install tensorflow`  
如果你的电脑拥有支持Nvidia CUDA的GPU，那么可以安装GPU版本的TensorFlow来进行计算加速，执行以下命令  
`pip install tensorflow-gpu`  
### Keras库安装
Keras的安装同样可以使用pip包管理器，执行以下命令：  
`pip install keras`  
### 关于多版本Python的问题
如果同时安装了多个版本的python，那么pip包管理器很容易产生混淆（一般一个pip对应管理一个python解释器的库），出现第三方库没有安装到当前使用的python上的情况，通过拷贝文件夹使用python同样也会引起pip无法使用的问题。出现这种情况可以尝试使用以下的命令：  
`python -m pip install keras`  
采用以上命令可以直接调用python解释器对应的pip。

## 用Keras构建模型
在Keras中，一个核心的数据结构就是model，使用Keras构建模型可以通过两种方式，分别为`Suquential`和`Model`。
`Sequential`可以用来构建具有线性结构的模型，通过向`Sequencial`的构造函数传递一组层实体或通过`.add()`添加层：  
```
#导入Keras相关模块
from keras.models import Sequential  
from keras.layers import Dense,Activation  

#通过Sequential构造函数创建模型
model1 = Sequential([
    Dense(32,input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

#通过调用.add()函数构建模型
model2=Sequential()
model2.add(Dense(32,input_dim=784))
model2.add(Activation('relu'))
```
对于具有较为复杂结构的模型（如包含分支结构，具有多个输入、输出，共享层等），可以使用Keras functional API，即通过`Model()`构建模型：  
```
#导入Keras相关模块
from keras.layers import Input,Dense  
from keras.models import Model  

#定义模型输入
inputs = Input(shape=(784,))  

#添加层
x=Dense(64,activation='relu')(inputs)
x=Dense(64,activation='relu')(x)
predictions = Dense(10,activation='softmax')(x)

#构建模型
model=Model(inputs=inputs,outputs=predictions)
```
通过上面这段代码可以知道  
* layer实体可以在张量上进行调用，其返回值依然是一个张量
* 定义好输入和输出张量后就可以定义一个`Model`  
* 采用这种方式构建的模型和使用`Sequential`构建的模型功能相同

### 模型的编译
在开始模型训练之前，需要通过`compile`方法对模型进行编译，并设置学习的过程。`compile`方法接受三个参数：  
* optimizer  
    优化器，可以使用已有优化器的字符串名称（如`rmsprop`或`adagrad`）或者`Optimizer`类的实体。
* loss  
    损失函数，可以使用已有损失函数的字符串名称（如`categorical_crossentropy`或`mse`）或者`Loss`类的实体。
* metrics  
    评价函数，可以使用已有评价函数的字符串名称或者自定义的评分函数。  

```
#多分类问题
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#二分类问题
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#均方差回归问题
model.compile(optimizer='rmsprop',
              loss='mse')
              
#自定义评价
import keras.backend as K

def mean_pred(y_true,y_pred):
    return K.mean(y_pred)
    
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy',mean_pred])
```

### 模型的训练
Keras模型使用`fit`函数进行模型的训练。  
```
#构建模型
model = Sequential()
model.add(Dense(32,activation='relu',input_dim=100))  
model.add(Dense(1,activation='sigmoid'))  
model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

#构建数据
import numpy as np
data=np.random.random((1000,100))  
labels=np.random.randint(2,size=(1000,1))  

#拟合数据
model.fit(data,labels,epochs=10,batch_size=32)
```

## 小结
这篇文章主要记录自己学习Keras的过程，内容主要来自Keras官网文档及自身理解，欢迎对深度学习、图像处理有兴趣的朋友一起学习、讨论。