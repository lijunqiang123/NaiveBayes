#coding:utf-8
from numpy import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
from decimal import Decimal
#coding:utf-8
# 朴素贝叶斯：朴素是因为整个形式化过程只做最原始、最简单的假设。
# 优点：在数据较少的情况下仍然有效，可以处理多类别问题
# 缺点：对于输入数据的准备方式较为敏感
# 适用类型：标称型数据。
# 标称型：一般在有限的数据中取，而且只存在“是”和“否”两种不同的结果。（一般用于分类）
# 数值型：可以在无限的数据中取，而且数值比较具体化，例如4.01,3.16这种值（一般用于回归分析）
# 贝叶斯决策理论的核心：p1(x,y)>p2(x,y) ，那么点(x,y)属于类别1；p1(x,y)<p2(x,y)，那么属于类别2。注：p为概率
# 朴素贝叶斯假设特征之间相互独立。所谓独立指的是统计意义上的独立，即一个特征或者单词出现的可能性与它和其他单词相邻没有关系。
# 朴素贝叶斯的另一个假设是，每个特征同等重要。

def train_bayes_model(Xtrain, Ytrain):
    # print Ytrain
    feature_freq = {}  # { (class,feature):frequency }
    class_freq = {}  # { class:frequency }

    for i in range(len(Ytrain)):
        class_freq[Ytrain[i]] = class_freq.get(Ytrain[i], 0) + 1
        for j in range(len(Xtrain[i])):
            feature_freq[(Ytrain[i], Xtrain[i][j])] = feature_freq.get((Ytrain[i], Xtrain[i][j]), 0) + 1
    # print feature_freq
    return feature_freq, class_freq

def naive_bayes(Xtest, feature_freq, class_freq):
    predicts = []

    for i in range(len(Xtest)):  # calculate for each array in test set
        temp = Xtest[i]
        probabilities = {}
        # calculate probability for each class 0,1,2 and pick max
        # P(c|x) = P(x1|c).P(x2|c).P(x3|c).P(x4|c).P(c) then argmax
        for j in class_freq.keys():

            V = 10
            prob = Decimal(class_freq[j])/Decimal(sum(class_freq.values()))
            # calculate likelihood probabilities for each feature
            for feature in temp:
                # P(x1|c) = x1 count in C + 1 / feature count in C + V      {=Count(X1,C) + 1 / Count(C) + V}
                # p(c|x)
                #one way to solve feature_freq.get((j, feature),0) == 0
                # prob *= float(feature_freq.get((j, feature),0)+1) / float(class_freq.get(j,0)+V)
                #another way
                if((feature_freq.get((j, feature), 0) == 0) or (class_freq.get(j, 0) == 0)):
                    prob *= Decimal(0.00058)
                    # prob *= Decimal(0.00040)
                    # prob *= Decimal(0.0050)
                else:
                    prob *= Decimal(feature_freq.get((j, feature))) / Decimal((class_freq.get(j)))
                    # print Decimal(feature_freq.get((j, feature))) / Decimal((class_freq.get(j)))


            probabilities[j] = prob  # probabilities for 0,1,2

        best = max(probabilities, key=probabilities.get)  # 选择最大的概率，返回对应的键值作为选择的类别
        predicts.append(best)
    return predicts

def testIris():
    """
    自定义数据集
    :return: 数据集的输入和标签
    """

    # 交叉验证
    iris = datasets.load_iris()
    # print iris.target  #3种target 0 1  2
    kf = KFold(n_splits=5,shuffle=True)   #5折交叉验证
    score = np.zeros(5)
    i = 0
    for train_index,test_index in kf.split(iris.data):

        X_train,X_test = iris.data[train_index],iris.data[test_index]
        y_train,y_test = iris.target[train_index],iris.target[test_index]
        feature_freq, class_freq = train_bayes_model(X_train, y_train)
        predicts = naive_bayes(X_test, feature_freq, class_freq)
        score[i] = metrics.accuracy_score(y_test, predicts)
        # print("Naive Bayes accuracy score for part " + str(i) + " is: {}".format(round(score[i] * 100, 2)))
        i+=1
    print ("交叉验证手写朴素贝叶斯Iris测试的平均正确率：%f\n" % round(score.mean()*100,10))

    #非交叉验证
    X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.3,shuffle=True)   #验证集大小150*0.3=45
    feature_freq, class_freq = train_bayes_model(X_train, y_train)
    predicts = naive_bayes(X_test, feature_freq, class_freq)
    score = metrics.accuracy_score(y_test, predicts)
    print ("非交叉验证手写朴素贝叶斯Iris测试的正确率：%f\n" % round(score*100,10))

def testWine():
    """
    自定义数据集
    :return: 数据集的输入和标签
    """
    #交叉验证
    wine = datasets.load_wine()
    # print wine.target  3种target 0 1 2
    kf = KFold(n_splits=5, shuffle=True)  # 5折交叉验证
    score = np.zeros(5)
    i = 0
    for train_index, test_index in kf.split(wine.data):
        X_train, X_test = wine.data[train_index], wine.data[test_index]
        y_train, y_test = wine.target[train_index], wine.target[test_index]
        feature_freq, class_freq = train_bayes_model(X_train, y_train)
        predicts = naive_bayes(X_test, feature_freq, class_freq)
        score[i] = metrics.accuracy_score(y_test, predicts)
        # print("Naive Bayes accuracy score for part " + str(i) + " is: {}".format(round(score[i] * 100, 2)))
        i += 1
    print ("交叉验证手写朴素贝叶斯Wine测试的平均正确率：%f\n" % round(score.mean() * 100, 10))
    #非交叉验证
    X_train,X_test,y_train,y_test = train_test_split(wine.data,wine.target,test_size=0.3,shuffle=True)   #验证集大小150*0.3=45
    feature_freq, class_freq = train_bayes_model(X_train, y_train)
    predicts = naive_bayes(X_test, feature_freq, class_freq)
    score = metrics.accuracy_score(y_test, predicts)
    print ("非交叉验证手写朴素贝叶斯Wine测试的正确率：%f\n" % round(score*100,10))

def testBreast():
    """
    自定义数据集
    :return: 数据集的输入和标签
    """
    #交叉验证
    breast = datasets.load_breast_cancer()
    # print breast.target  #2种target 0 1
    kf = KFold(n_splits=5,shuffle=True)   #5折交叉验证
    score = np.zeros(5)
    i = 0
    for train_index,test_index in kf.split(breast.data):

        X_train,X_test = breast.data[train_index],breast.data[test_index]
        y_train,y_test = breast.target[train_index],breast.target[test_index]
        feature_freq, class_freq = train_bayes_model(X_train, y_train)
        predicts = naive_bayes(X_test, feature_freq, class_freq)
        score[i] = metrics.accuracy_score(y_test, predicts)
        # print("Naive Bayes accuracy score for part " + str(i) + " is: {}".format(round(score[i] * 100, 2)))
        i+=1
    print ("交叉验证手写朴素贝叶斯Breast测试的平均正确率：%f\n" % round(score.mean()*100,10))

    #非交叉验证
    X_train,X_test,y_train,y_test = train_test_split(breast.data,breast.target,test_size=0.3,shuffle=True)   #验证集大小150*0.3=45
    feature_freq, class_freq = train_bayes_model(X_train, y_train)
    predicts = naive_bayes(X_test, feature_freq, class_freq)
    score = metrics.accuracy_score(y_test, predicts)
    print ("非交叉验证手写朴素贝叶斯Breast测试的正确率：%f\n" % round(score*100,10))

if __name__ == '__main__':
    testIris()
    testWine()
    testBreast()
