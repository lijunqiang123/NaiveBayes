## 朴素贝叶斯

SklearnBayes.py  使用sklearn库中GaussianNB和MultinomialNB，但下面结果表中只填入了GaussianNB的测试结果

SelfBayes.py 手写朴素贝叶斯




## 数据集测试结果


以7:3的比例划分测试集 某一次运行的结果

| 数据集名称 |   手写实现（交叉验证）  | sklearn实现（交叉验证） |手写实现（非交叉验证）| sklearn实现（非交叉验证）|
| ----------------- | ---------------- | ---------------- | ------------------- | ------------------- |
| iris              | 0.84          | 0.95             | 0.84             | 0.95                |
| wine              | 0.70          | 0.97             | 0.62           | 0.98              |
| breast_cancer     | 0.70           | 0.94          | 0.72            | 0.92                |
