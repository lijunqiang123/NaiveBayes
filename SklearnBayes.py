#coding: utf-8
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np

def testIris():
    iris = datasets.load_iris()
    # print iris.data
    kf = KFold(n_splits=5, shuffle=True)  # 5折交叉验证
    # print (kf)
    gnb = GaussianNB()
    mnb = MultinomialNB()
    score_gnb = np.zeros(5)
    score_mnb = np.zeros(5)
    # accuracy = np.zeros(5)
    i = 0
    for train_index, test_index in kf.split(iris.data):
        X_train, X_test = iris.data[train_index], iris.data[test_index]
        y_train, y_test = iris.target[train_index], iris.target[test_index]
        y_gnb = gnb.fit(X_train, y_train).predict(X_test)
        y_mnb = mnb.fit(X_train, y_train).predict(X_test)
        # print ("训练集的数据大小为：%d\n 验证集的数据大小为：%d\n" % (len(X_train), len(X_test)))
        # a = (y_test == y_gnb).sum()
        # accuracy[i] = (float(a)/len(X_test))
        # print accuracy[i]
        # i+=1
        score_gnb[i] = metrics.accuracy_score(y_test, y_gnb)
        score_mnb[i] = metrics.accuracy_score(y_test, y_mnb)
        i += 1
    print ("交叉验证 GNB Iris测试的平均正确率：%f\n" % round(score_gnb.mean() * 100, 10))
    print ("交叉验证 MNB Iris测试的平均正确率：%f\n" % round(score_mnb.mean() * 100, 10))

    #非交叉验证
    X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.3,shuffle=True)
    y_gnb = gnb.fit(X_train, y_train).predict(X_test)
    y_mnb = mnb.fit(X_train, y_train).predict(X_test)
    print ("非交叉验证 GNB Iris测试的正确率：%f\n" % round(score_gnb.mean() * 100, 10))
    print ("非交叉验证 MNB Iris测试的正确率：%f\n" % round(score_mnb.mean() * 100, 10))

def testWine():
    wine = datasets.load_wine()
    kf = KFold(n_splits=5, shuffle=True)  # 5折交叉验证
    gnb = GaussianNB()
    mnb = MultinomialNB()
    score_gnb = np.zeros(5)
    score_mnb = np.zeros(5)
    # accuracy = np.zeros(5)
    i = 0
    for train_index, test_index in kf.split(wine.data):
        X_train, X_test = wine.data[train_index], wine.data[test_index]
        y_train, y_test = wine.target[train_index], wine.target[test_index]
        y_gnb = gnb.fit(X_train, y_train).predict(X_test)
        y_mnb = mnb.fit(X_train, y_train).predict(X_test)
        # print ("训练集的数据大小为：%d\n 验证集的数据大小为：%d\n" % (len(X_train), len(X_test)))
        # a = (y_test == y_gnb).sum()
        # accuracy[i] = (float(a)/len(X_test))
        # print accuracy[i]
        # i+=1
        score_gnb[i] = metrics.accuracy_score(y_test, y_gnb)
        score_mnb[i] = metrics.accuracy_score(y_test, y_mnb)
        i += 1
    print ("交叉验证 GNB Wine测试的平均正确率：%f\n" % round(score_gnb.mean() * 100, 10))
    print ("交叉验证 MNB Wine测试的平均正确率：%f\n" % round(score_mnb.mean() * 100, 10))
    #非交叉验证
    X_train,X_test,y_train,y_test = train_test_split(wine.data,wine.target,test_size=0.3,shuffle=True)
    y_gnb = gnb.fit(X_train, y_train).predict(X_test)
    y_mnb = mnb.fit(X_train, y_train).predict(X_test)
    score_gnb = metrics.accuracy_score(y_test, y_gnb)
    score_mnb = metrics.accuracy_score(y_test, y_mnb)
    print ("非交叉验证 GNB Wine测试的正确率：%f\n" % round(score_gnb * 100, 10))
    print ("非交叉验证 MNB Wine测试的正确率：%f\n" % round(score_mnb * 100, 10))

def testBreast():
    breast = datasets.load_breast_cancer()
    # print iris.data
    kf = KFold(n_splits=5, shuffle=True)  # 5折交叉验证
    # print (kf)
    gnb = GaussianNB()
    mnb = MultinomialNB()
    score_gnb = np.zeros(5)
    score_mnb = np.zeros(5)
    # accuracy = np.zeros(5)
    i = 0
    for train_index, test_index in kf.split(breast.data):
        X_train, X_test = breast.data[train_index], breast.data[test_index]
        y_train, y_test = breast.target[train_index], breast.target[test_index]
        y_gnb = gnb.fit(X_train, y_train).predict(X_test)
        y_mnb = mnb.fit(X_train, y_train).predict(X_test)
        # print ("训练集的数据大小为：%d\n 验证集的数据大小为：%d\n" % (len(X_train), len(X_test)))
        # a = (y_test == y_gnb).sum()
        # accuracy[i] = (float(a)/len(X_test))
        # print accuracy[i]
        # i+=1
        score_gnb[i] = metrics.accuracy_score(y_test, y_gnb)
        score_mnb[i] = metrics.accuracy_score(y_test, y_mnb)
        i += 1
    print ("交叉验证 GNB Breast测试的平均正确率：%f\n" % round(score_gnb.mean() * 100, 10))
    print ("交叉验证 MNB Breast测试的平均正确率：%f\n" % round(score_mnb.mean() * 100, 10))
    #非交叉验证
    X_train,X_test,y_train,y_test = train_test_split(breast.data,breast.target,test_size=0.3,shuffle=True)
    y_gnb = gnb.fit(X_train, y_train).predict(X_test)
    y_mnb = mnb.fit(X_train, y_train).predict(X_test)
    score_gnb = metrics.accuracy_score(y_test, y_gnb)
    score_mnb = metrics.accuracy_score(y_test, y_mnb)
    print ("非交叉验证 GNB Breast测试的正确率：%f\n" % round(score_gnb * 100, 10))
    print ("非交叉验证 MNB Breast测试的正确率：%f\n" % round(score_mnb * 100, 10))


if __name__ == '__main__':
    testIris()
    testWine()
    testBreast()


    # 随机排列data
    # X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.3,shuffle=True)   #验证集大小150*0.3=45
    # y_pred = gnb.fit(X_train,y_train).predict(X_test)
    # y_mnb = mnb.fit(X_train,y_train).predict(X_test)
    # print ("训练集的数据大小为：%d\n 验证集的数据大小为：%d\n GNB 验证集中错误的数目：%d" % (len(X_train),len(X_test),(y_test != y_pred).sum()))
    # print ("MNB 验证集中错误的数目：%d" % (y_test != y_mnb).sum())
    # a = (y_test == y_pred).sum()
    # b = (y_test == y_mnb).sum()
    # print ("GNB测试的正确率：%f\n" % (float(a)/len(X_test)))
    # print ("MNB测试的正确率：%f\n" % (float(b)/len(X_test)))
    # print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(y_test != y_pred).sum()))