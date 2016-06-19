#身高，体重，肺活量， 50 米跑，立定跳远，坐位体前屈 -> 长跑成绩
#只针对女同学

# 导入
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import load_workbook


# 读取excel中的数据，返回训练集和测试集的输入、输出数组
def loadData():

    # 读取excel
    wb = load_workbook(filename='female(all).xlsx', read_only=True)
    ws = wb['Sheet1']
    
    # trainMat：训练集输入    testMat：测试集输入
    # trainTargetMat：训练集输出    testTargetMat：测试集输出
    trainMat = []
    testMat = []
    trainTargetMat = []
    testTargetMat = []
 

    # 读取excel中的数据
    for idx,row in enumerate(ws.rows):
        
        # 跳过第一行列名的读取
        if(idx == 0):
            continue
        
        # 方便之后矩阵运算
        vec = [1.0]
        
        # 把每行数据的输出和输入存在不同的列表中
        for cell in row:

            if(cell.column == 7): 
                # 将原本的跑步成绩转换成秒 
                temp = cell.value
                minASec = temp.split('\'')
                temp = int(minASec[0]) * 60 + int(minASec[1])
                
                # 训练集与测试集输出的划分
                if(idx <= 1500):
                    trainTargetMat.append(temp)
                else:
                    testTargetMat.append(temp)               
                   
            else:
                vec.append(cell.value)

        # 训练集与测试集输入的划分       
        if(idx <= 1500):
            trainMat.append(vec)
        else:
            testMat.append(vec)
        
   
    return trainMat, trainTargetMat, testMat, testTargetMat
    
    
    
# 对数据进行特征归一化处理
def normalize(x):
    xMin = x.min(axis=0)
    xMax = x.max(axis=0)
    range =  xMax - xMin
    range[0] = 1
    xMin[0] = 0
    x_norm = (x - xMin) / range
    
    return x_norm
    
    
    
    
# 梯度下降
def gradDescent(dataMatIn, targetMatIn):
    
    # 将python的list转换成numpy的ndarray，同时将输入的特征向量规范化
    x = np.array(dataMatIn)
    x_norm = normalize(x)
    print(x_norm)
    xTrans = x_norm.transpose()
    y = np.array(targetMatIn)

    alpha = 0.9
    numIterations = 3000
    m,n = x_norm.shape
    theta = np.ones(n)
    
    # 用于后面画图表示梯度下降的过程
    JTheta = []
    
     # 每次循环更新theta时，都使用所有的训练数据
    for i in range(0, numIterations):
        hypothesis = np.dot(x_norm, theta)
        # print("hypothesis")
        # print(hypothesis)
        # print("y")
        # print(y)
        loss = hypothesis - y
        # 计算J(theta) - 平均每个输入相对于当前theta的代价
        cost = np.sum(loss ** 2) / (2 * m)
        JTheta.append(cost)
        print("Iteration %d | Cost: %f" % (i, cost))
        # 计算梯度
        gradient = np.dot(xTrans, loss) / m
        # 更新theta值
        theta = theta - alpha * gradient
        #print("theta", theta)
        
    return theta, JTheta
    
    
    
    
# 将模型用于测试数据集，观察每个输入的预测输出与真实值得差别
def testModel(theta, testMat, testTargetMat):
    
    # 将python的list转换成numpy的ndarray，同时将输入的特征向量规范化
    x = np.array(testMat)
    x_norm = normalize(x)
    y = np.array(testTargetMat)

    predict = np.dot(x_norm, theta)
    loss = predict - y

    return loss




#训练和测试模型，绘图
a1,a2,a3,a4 = loadData()
theta, JTheta = gradDescent(a1,a2)
print(theta)
costValues = testModel(theta,a3,a4)
# print(costValues)


# 画出梯度下降过程中代价函数值随迭代过程的变化
# plt.plot(JTheta, linewidth=1.0)
# plt.ylabel('cost function')
# plt.xlabel('iteration times')
# plt.show()


# 画出将模型用于测试数据时，预测值与真实值之间的偏差情况
# costValues = testModel(theta,a3,a4)
# print(costValues)
# plt.plot(costValues, 'ro')
# plt.ylabel('differential')
# plt.xlabel('test input')
# plt.show()


# 同时画出上面两个图
plt.figure(1)
plt.subplot(211)
plt.plot(JTheta, linewidth=1.0)
plt.ylabel('cost function')
plt.xlabel('iteration times')

plt.subplot(212)
plt.plot(costValues, 'ro')
plt.ylabel('differential')
plt.xlabel('test input')
plt.show()

