"""
================================
手写数字识别
================================
"""


import matplotlib.pyplot as plt

# 导入datasets, classifiers和performance metrics
from sklearn import datasets, svm, metrics

# 从sklearn的datasets里导出手写数字数据
digits = datasets.load_digits()

# 将手写数字图像数据与它们所代表的数字捆绑在一起
images_and_labels = list(zip(digits.images, digits.target))     # zip('ABCD', 'xy') --> Ax By. Make an iterator that aggregates elements from each of the iterables.

# 画出训练集中的手写数字图像并标上它们所代表的数字
for index, (image, label) in enumerate(images_and_labels[6:9]):
    plt.subplot(2, 3, index + 1)
    plt.axis('off')       # turns off the axis lines and labels
    plt.imshow(image, cmap=plt.cm.bone, interpolation='nearest')    #gray_r  cmap: Colormap(绘图所用的颜色)
    plt.title('Training: %i' % label)

# 为了训练SVM分类器，需要将这些手写数字的图像数据以矩阵的形式表示出来（行数：样本数，列数：特征数目）
n_samples = len(digits.images)
print("len:", n_samples, "\n")
data = digits.images.reshape((n_samples, -1))       # sample个数的行，feature个数的列的矩阵(-1表示省略说明列数)
print("data:\n", data, "\n\n"+"shape:", data.shape, "\n")


# 创建一个SVM分类器，并设置它的C和gamma参数（默认使用Radial Basis Function kernel）
classifier = svm.SVC(C=0.9, gamma=0.001)


# 将手写数字数据的前一半用来训练SVM分类器
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# 将手写数字数据的后一半用来测试训练得到的SVM分类器的性能
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

# 输出测试的相关信息
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))       # The support is the number of occurrences of each class in y_true - Ground truth (correct) target values.
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))      # By definition a confusion matrix C is such that C(i, j) is equal to the number of observations known to be in group i but predicted to be in group j.

# 画出测试集中的手写数字图像并为它们标上SVM分类器预测得到的数字
images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[6:9]):
    plt.subplot(2, 3, index + 4)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.bone, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
