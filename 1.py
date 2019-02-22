#鸢尾花分类问题
#使用tf.estimator.DNNClassifier

#启动tensorflow虚拟环境：
#进入C:\users\Admin\>
#执行.\venv\Scripts\activate

import os
import pandas as pd
import tensorflow as tf

################加载数据####################

Features=['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
Species=['Setosa','Versicolor','Virginica']

#设定文件路径：
dir_path=os.path.dirname(os.path.realpath(__file__))  #获取当前脚本所在路径
#记录iris_training.csv路径
train_path=os.path.join(dir_path,'iris_training.csv')
#记录iris_test.csv路径
test_path=os.path.join(dir_path,'iris_test.csv')

#读取train csv
train=pd.read_csv(train_path,names=Features,header=0)
#print(train)
train_x,train_y=train,train.pop('Species') #pop返回对应列

#读取test csv
test=pd.read_csv(test_path,names=Features,header=0)
test_x,test_y=test,test.pop('Species')

# 创建特征列：
# *https://www.jianshu.com/p/fceb64c790f3
# *https://www.sohu.com/a/211388625_670669
feature_columns=[]
#train_x.keys为列索引 即列名
for key in train_x.keys():
    #print(key)
    feature_columns.append(tf.feature_column.numeric_column(key=key))
#print(feature_columns)

###############定义分类器################

#定义DNN：
#特征列为feature_columns 隐含层为10,20,10 输出层神经元个数为3
#https://www.tensorflow.org/api_docs/python/tf/contrib/estimator/DNNEstimator
classifier=tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10,20,10],
    n_classes=3,
    #注意：更改模型后 比如更改隐含层神经元个数后 需要清空该输出文件夹 否则报错
    model_dir='./output',    #to remove warning:Using temporary folder as model directory
)

##################训练模型#######################
#定义input function
#features：训练集数据train_x labels:训练集标签train_y  batch_size:每次gradient descent所处理的数据
def train_input_fun(features,labels,batch_size):
    #创建来自tensors的dataset
    #from_tensor_slices的作用是切分传入Tensor的第一个维度，生成相应的dataset
    # *https://blog.csdn.net/qi_1221/article/details/79460875
    #dict(features)将训练集转化为字典的形式 键值为4个特征 各个键对应的数值为长120的列表
    dataset=tf.data.Dataset.from_tensor_slices((dict(train_x),train_y))
    # print(dataset)
    # <TensorSliceDataset shapes: (
    #     {SepalLength: (), SepalWidth: (), PetalLength: (), PetalWidth: ()}, ()), 
    #     types: ({SepalLength: tf.float64, SepalWidth: tf.float64, PetalLength: tf.float64, PetalWidth: tf.float64}, tf.int64)
    #     >
    
    #设定dataset中shuffle(buffer_size大于Dataset中样本数量，确保数据完全随机处理)
    # *https://blog.csdn.net/qq_16234613/article/details/81703228
    # *https://www.tensorflow.org/guide/datasets_for_estimators
    dataset=dataset.shuffle(1000)
    #设定batch_size
    dataset=dataset.batch(batch_size)
    #设定epoch数(一次epoch即训练集中全部数据训练一次) 默认为空 非空可能造成最后一组batch大小不等于batch_size（这并没啥问题）
    dataset=dataset.repeat()

    #创建单次迭代器 返回tensor
    return dataset.make_one_shot_iterator().get_next()
#print(dict(train_x).get('PetalWidth'))

# 将 TensorFlow 日志信息输出到屏幕
# tf.logging.set_verbosity(tf.logging.INFO)

#定义batch_size
batch_size=120
#训练
classifier.train(input_fn=lambda:train_input_fun(train_x,train_y,batch_size),steps=1000)

####################评估模型###########################
#定义评估函数
def eval_input_fun(features,labels,batch_size):
    features=dict(test_x)
    inputs=(features,test_y)
    dataset=tf.data.Dataset.from_tensor_slices(inputs)
    dataset=dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()

#评估模型
result=classifier.evaluate(input_fn=lambda:eval_input_fun(test_x,test_y,batch_size))

#打印结果
print(result)