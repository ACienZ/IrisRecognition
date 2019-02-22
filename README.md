# IrisRecognition

## 简介
> 该脚本为[鸢尾花分类](https://www.kaggle.com/uciml/iris)的“深度”神经网络分类器`demo`

## 测试环境
> 操作系统：win10
> 
> python:3.6.8
>
> tensorflow:1.12.0 GPU version

## 操作步骤
> 1.加载数据
> 
> 使用pandas读取训练集及测试集所在CSV文件中的数据
> 
> 2.创建特征列并定义分类器网络结构
>
>此处隐含层结构为[10,20,10]
>
> 3.训练模型
>
>此处batch_size为120(一次性训练) steps为1000
>
> 4.测试模型

## 测试结果(随机抽取一次运行结果)
> 'accuracy': 0.96666664, 
> 
> 'average_loss': 0.08288006, 
> 
> 'loss': 2.4864018, 
> 
> 'global_step': 3000

## 数据来源
> http://download.tensorflow.org/data/iris_training.csv
> http://download.tensorflow.org/data/iris_test.csv

## 参考文献
> 1.[Tensorflow-iris-案例教程-零基础-机器学习](https://www.jianshu.com/p/b86c020747f9)
> 
> 2.[Estimator 的数据集](https://www.tensorflow.org/guide/datasets_for_estimators)