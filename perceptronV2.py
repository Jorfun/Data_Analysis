import numpy as np
import matplotlib.pyplot as plt
from random import choice



    
# 从csv文件获取所有数据
def loadData():

    #获取测试和训练数据
    f = open('perceptron(2dim).csv', 'r')
    
    trainInput = []
    trainOutput = []
    testInput = []
    testOutput = []
    boys = []
    girls = []
    
    for idx,line in enumerate(f):
        temp = line.strip('\n').split(',')
        
        if(idx <= 2000):
            trainInput.append([1, int(temp[0]), int(temp[1])])
            trainOutput.append(int(temp[2]))
        else:           
            testInput.append([1, int(temp[0]), int(temp[1])])
            testOutput.append(int(temp[2]))            
            
            if(temp[2] == '1'):
                girls.append([int(temp[0]), int(temp[1])])
            else:
                boys.append([int(temp[0]), int(temp[1])])
     
    print("boys number:", len(boys))
    print("girls number", len(girls), "\n")
     
    return trainInput, trainOutput, testInput, testOutput, boys, girls    
    



# 对数据进行特征归一化处理
def normalize(x):
    xMin = x.min(axis=0)
    xMax = x.max(axis=0)
    range =  xMax - xMin
    print("xMin \n", xMin)
    print("xMax \n", xMax)
    print("range \n", range, "\n\n")
    range[0] = 1    #输入数据第一维分量为1时的特殊处理
    xMin[0] = 0    #输入数据第一维分量为1时的特殊处理
    x_norm = (x - xMin) / range
    
    return x_norm



    
# 用于感知器的sign函数
def sign(x):
    if(x>0):
        return 1
    else:
        return -1
    

    
#  通过口袋算法来学习感知器的权重    
def pocketAlgorithm(trainInput, trainOutput):
    trainInput = np.array(trainInput)
    trainInputNormed = normalize(trainInput)
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
        for idx, val in enumerate(trainInputNormed):
            # print(idx, val)
            # print(bestWeight)
            predictValue = sign(val.dot(tempWeight))
            
            if(predictValue != trainOutput[idx]):
                errorList.append(idx)
         
        # 保存当前感知器错误分类的点个数,并判断是否要更新最优感知器
        errorMade = len(errorList)
        if(errorMade < leastErrorMade):
            bestWeight = tempWeight
            leastErrorMade = errorMade
        
        #print("errorMade \n", errorMade, "\n\n")
        
        # 从所有错误的点中随机选一个用来纠正当前的感知器
        randomErrorIndex= choice(errorList)
        
        # 针对随机选的点对感知器进行修正
        tempWeight = tempWeight + alpha * trainOutput[randomErrorIndex] * trainOutput[randomErrorIndex]
        #print("tempWeight \n", tempWeight, "\n\n")
          
        # 确保相应的迭代次数过后退出循环
        iteration -= 1

        
    print("leastErrorMade:", leastErrorMade, "\n")
    print("bestWeight:", bestWeight, "\n")
        
    return bestWeight



#  测试当前得到感知器在测试数据集上的效果
def testPerceptron(perceptron, testInput, testOutput):
    testInput = np.array(testInput)
    testInputNormed = normalize(testInput)
    errorCount = 0  #预测错误次数
    totalTest = 0   #总的预测次数
    
    for idx, val in enumerate(testInputNormed):
        predictValue = sign(val.dot(perceptron))
        
        if(predictValue != testOutput[idx]):
            errorCount += 1     
        
        totalTest += 1
        
    print("Total test:", totalTest, "  Error:", errorCount, "  Error rate:", errorCount/totalTest)
    
    
    

#从csv文件获取所有数据
trainInput, trainOutput, testInput, testOutput, boys, girls = loadData()

# 通过pocket algorithm求出感知器
perceptron = pocketAlgorithm(trainInput, trainOutput)
#print(perceptron)

testPerceptron(perceptron, testInput, testOutput)


# 观察男女生在肺活量和立定跳的两个维度上是否线性可分
boys = np.array(boys)
girls = np.array(girls)

plt.plot(boys[ : ,0], boys[ : ,1], 'ro')
plt.plot(girls[ : ,0], girls[ : ,1], 'bo')
plt.show()


# 画出数据点与得到的感知器
# plt.plot(testData_normed[ : ,1], testData_normed[ : ,2], 'ro')
# plt.plot([0, perceptron[0]/-perceptron[2]],[perceptron[0]/-perceptron[1], 0])
# plt.show()  


# 画出标记真实类别的数据点和得到的感知器
# boys = np.array(boys)
# girls = np.array(girls)
# boys_normed = boys / boys.max(axis=0)
# girls_normed = girls / girls.max(axis=0)
# plt.plot(boys_normed[ : ,0], boys_normed[ : ,1], 'ro')
# plt.plot(girls_normed[ : ,0], girls_normed[ : ,1], 'bo')
# plt.plot([0, perceptron[0]/-perceptron[1]],[perceptron[0]/-perceptron[2], 0])
# plt.show() 


