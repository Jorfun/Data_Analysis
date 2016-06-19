"""
====================================
logistic regression - 通过肺活量和立定跳远成绩来预测男女
====================================
"""

# 导入
import numpy as np
import matplotlib.pyplot as plt
import math


# 从csv文件获取所有数据
def loadData():

    #获取测试和训练数据
    f = open('logistic(2dim).csv', 'r')
    
    trainInput = []
    trainOutput = []
    testInput = []
    testOutput = []
    
    for idx,line in enumerate(f):
        temp = line.strip('\n').split(',')
        
        if(idx <= 2000):
            trainInput.append([int(temp[0]), int(temp[1])])
            trainOutput.append(int(temp[2]))
        else:           
            testInput.append([int(temp[0]), int(temp[1])])
            testOutput.append(int(temp[2]))
     
    return trainInput, trainOutput, testInput, testOutput




# 对数据进行特征归一化处理
def normalize(x):
    xMin = x.min(axis=0)
    xMax = x.max(axis=0)
    range =  xMax - xMin
    # print("xMin \n", xMin)
    # print("xMax \n", xMax)
    # print("range \n", range, "\n\n")
    # range[0] = 1
    # xMin[0] = 0
    x_norm = (x - xMin) / range
    
    return x_norm




def sigmoid(x):
    if(type(x) is np.float64):
        return 1 / (1 + math.exp(-x))
    else:
        return 1 / (1 + np.exp(-x))




# 梯度下降
def gradDescent(trainInput, trainOutput):
    
    # 将python的list转换成numpy的ndarray，同时将输入的特征向量规范化
    x = np.array(trainInput) 
    x_norm = normalize(x)
    xTrans = x_norm.transpose()
    y = np.array(trainOutput)
    # print("x \n", x, "\n\n")
    # print("x_norm \n", x_norm, "\n\n")
    # print("y \n", y, "\n\n")
    
    alpha = 20
    numIterations = 500
    m,n = x_norm.shape
    theta = np.ones(n)
    nOnes = np.ones(m)      #用来计算代价函数
    
    # 用于后面画图表示梯度下降的过程
    JTheta = []
    
    # 每次循环更新theta时，都使用所有的训练数据
    for i in range(0, numIterations):
        hypothesis = sigmoid(np.dot(x_norm, theta))
        loss = hypothesis - y
        # 计算J(theta) - 平均每个输入相对于当前theta的代价
        cost = np.sum( y * np.log(hypothesis) + (nOnes - y) * np.log(nOnes - hypothesis) ) / m * -1
        JTheta.append(cost)
        # print("hypothesis \n", hypothesis)
        # print("loss \n", loss, "\n\n")
        #print("Iteration %d | Cost: %f" % (i, cost))
        # 计算梯度
        gradient = np.dot(xTrans, loss) / m
        # 更新theta值
        theta = theta - alpha * gradient
        #print("theta", theta)
        
    return theta, JTheta

    
    
    
# 将模型用于测试数据集，观察每个输入的预测输出与真实值是否相同
def testModel(theta, testInput, testOutput):
    
    # 将python的list转换成numpy的ndarray，同时将输入的特征向量规范化
    testInput = np.array(testInput)
    testInputNormed = normalize(testInput)
    testOutput = np.array(testOutput)
    totalCount = 0
    errorCount = 0
    
    for idx, val in enumerate(testInputNormed):
        #print("type", type(val.dot(theta)))
        predictValue = sigmoid(val.dot(theta))
        
        if(predictValue > 0.5):
            predictValue = 1
        else:
            predictValue = 0
        
        if(predictValue != testOutput[idx]):
            errorCount += 1
        
        totalCount += 1

    print("Total test:", totalCount, "  Error:", errorCount, "  Error rate:", errorCount/totalCount)
    



#显示完整的nparray
# np.set_printoptions(threshold=np.nan)


#训练和测试模型，绘图
a1,a2,a3,a4 = loadData()
theta, JTheta = gradDescent(a1,a2)
print(theta, "\n\n")
testModel(theta,a3,a4)
#print(costValues)


# 画出梯度下降过程中代价函数值随迭代过程的变化
plt.plot(JTheta, linewidth=1.0)
plt.ylabel('cost function')
plt.xlabel('iteration times')
plt.show()


# 画出将模型用于测试数据时，预测值与真实值之间的偏差情况
# costValues = testModel(theta,a3,a4)
# print(costValues)
# plt.plot(costValues, 'ro')
# plt.ylabel('differential')
# plt.xlabel('test input')
# plt.show()


# 同时画出上面两个图
# plt.figure(1)
# plt.subplot(211)
# plt.plot(JTheta, linewidth=1.0)
# plt.ylabel('cost function')
# plt.xlabel('iteration times')

# plt.subplot(212)
# plt.plot(costValues, 'ro')
# plt.ylabel('differential')
# plt.xlabel('test input')
# plt.show()

