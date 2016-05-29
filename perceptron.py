import numpy as np
import matplotlib.pyplot as plt
from random import choice




# 从csv文件获取所有数据
def loadData():

    f = open('perceptron(2dim).csv', 'r')
    
    testData = []
    testAnswer = []
    
    for line in f:
        temp = line.strip('\n').split(',')
        testData.append([1, int(temp[0]), int(temp[1])])
        testAnswer.append(int(temp[2]))
        
    return testData, testAnswer

    

    
# 从csv文件分别获取男生女生的数据
def loadDataSeperate():

    f = open('perceptron(2dim).csv', 'r')
    
    boys = []
    girls = []
    
    for line in f:
        temp = line.strip('\n').split(',')
       
        if(temp[2] == '1'):
            boys.append([int(temp[0]), int(temp[1])])
        else:
            girls.append([int(temp[0]), int(temp[1])])
            
    return boys,girls    
    

    
    
# 用于感知器的sign函数
def sign(x):
    if(x>0):
        return 1
    else:
        return -1
    

    
    
def pocketAlgorithm(data, answer):
    tempWeight = np.array([-1,1,1]).transpose()
    bestWeight = tempWeight
    leastErrorMade = 1e10
    iteration = 1000
    alpha = 0.01
    
    
    # 设定pocket algorithm的迭代次数
    while(iteration >= 0):
        
        # 用来存放当前感知器分类错误的点
        errorList = []
        
        # 遍历所有数据得到分类存在问题的所有点
        for idx, val in enumerate(data):
            # print(idx, val)
            # print(val)
            # print(currentBestWeight)
            predictValue = sign(val.dot(tempWeight))
            
            if(predictValue != answer[idx]):
                errorList.append(idx)
         
        # 保存当前感知器错误分类的点个数,并判断是否要更新最优感知器
        errorMade = len(errorList)
        if(errorMade < leastErrorMade):
            bestWeight = tempWeight
            leastErrorMade = errorMade
        
        # print(errorMade)
        
        # 从所有错误的点中随机选一个用来纠正当前的感知器
        randomErrorIndex= choice(errorList)
        
        # 针对随机选的点对感知器进行修正
        tempWeight = tempWeight + alpha * answer[randomErrorIndex] * data[randomErrorIndex]
        
        # 确保相应的迭代次数过后退出循环
        iteration -= 1

    return bestWeight



    
# 观察男女生在肺活量和立定跳的两个维度上是否线性可分
# boys,girls = loadDataSeperate()

# boys = np.array(boys)
# girls = np.array(girls)

# plt.plot(boys[ : ,0], boys[ : ,1], 'ro')
# plt.plot(girls[ : ,0], girls[ : ,1], 'bo')
# plt.show()   

 
# 画出规范化之后的数据点分布图
# plt.plot(testData_normed[ : ,1], testData_normed[ : ,2], 'ro')
# plt.show()   


# 通过pocket algorithm求出感知器
testData, testAnswer = loadData()
testData = np.array(testData)
print(testData)
print(testAnswer)

testData_normed = testData / testData.max(axis=0)
print(testData_normed)

perceptron = pocketAlgorithm(testData_normed, testAnswer)
print(perceptron)


# 画出数据点与得到的感知器
# plt.plot(testData_normed[ : ,1], testData_normed[ : ,2], 'ro')
# plt.plot([0, perceptron[0]/-perceptron[2]],[perceptron[0]/-perceptron[1], 0])
# plt.show()  


# 画出标记真实类别的数据点和得到的感知器
boys, girls = loadDataSeperate()
boys = np.array(boys)
girls = np.array(girls)
boys_normed = boys / boys.max(axis=0)
girls_normed = girls / girls.max(axis=0)
plt.plot(boys_normed[ : ,0], boys_normed[ : ,1], 'ro')
plt.plot(girls_normed[ : ,0], girls_normed[ : ,1], 'bo')
plt.plot([0, perceptron[0]/-perceptron[2]],[perceptron[0]/-perceptron[1], 0])
plt.show() 


