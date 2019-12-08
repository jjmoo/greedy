"""
==============================================================
基于 Kernel LDA + KNN 的人脸识别
使用 Kernel Discriminant Analysis 做特征降维
使用 K-Nearest-Neighbor 做分类

数据:
    人脸图像来自于 Olivetti faces data-set from AT&T (classification)
    数据集包含 40 个人的人脸图像, 每个人都有 10 张图像
    我们只使用其中标签(label/target)为 0 和 1 的前 2 个人的图像

算法:
    需要自己实现基于 RBF Kernel 的 Kernel Discriminant Analysis 用于处理两个类别的数据的特征降维
    代码的框架已经给出, 需要学生自己补充 KernelDiscriminantAnalysis 的 fit() 和 transform() 函数的内容
==============================================================
"""

# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print(__doc__)
################################################
"""
Scikit-learn-compatible Kernel Discriminant Analysis.
"""

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class KernelDiscriminantAnalysis(BaseEstimator, ClassifierMixin,
                                 TransformerMixin):
    """Kernel Discriminant Analysis.

    Parameters
    ----------
    n_components: integer.
                  The dimension after transform.
    gamma: float.
           Parameter to RBF Kernel
    lmb: float (>= 0.0), default=0.001. 
         Regularization parameter

    """

    def __init__(self, n_components, gamma, lmb=0.001):
        self.n_components = n_components
        self.gamma = gamma
        self.lmb = lmb
        self.X = None # 用于存放输入的训练数据的 X
        self.K = None # 用于存放训练数据 X 产生的 Kernel Matrix
        self.M = None # 用于存放 Kernel LDA 最优化公式中的 M
        self.N = None # 用于存放 Kernel LDA 最优化公式中的 N
        self.EigenVectors = None # 用于存放 Kernel LDA 最优化公式中的 M 对应的广义特征向量, 每一列为一个特征向量, 按照对应特征值大小排序

    def fit(self, X, y):
        """Fit KDA model.

        Parameters
        ----------
        X: numpy array of shape [n_samples, n_features]
           Training set.
        y: numpy array of shape [n_samples]
           Target values. Only works for 2 classes with label/target 0 and 1.

        Returns
        -------
        self

        """
        pass

    def transform(self, X_test):
        """Transform data with the trained KernelLDA model.

        Parameters
        ----------
        X_test: numpy array of shape [n_samples, n_features]
           The input data.

        Returns
        -------
        y_pred: array-like, shape (n_samples, n_components)
                Transformations for X.

        """
        pass

################################################

# 指定 KNN 中最近邻的个数 (k 的值)
n_neighbors = 3

# 设置随机数种子让实验可以复现
random_state = 0

# 现在人脸数据集
faces = fetch_olivetti_faces()
targets = faces.target

# show sample images
images = faces.images[targets < 2] # save images

features = faces.data  # features
targets = faces.target # targets

fig = plt.figure() # create a new figure window
for i in range(20): # display 20 images
    # subplot : 4 rows and 5 columns
    img_grid = fig.add_subplot(4, 5, i+1)
    # plot features as image
    img_grid.imshow(images[i], cmap='gray')

plt.show()

# Prepare data, 只限于处理类别 0 和 1 的人脸
X, y = faces.data[targets < 2], faces.target[targets < 2]

# Split into train/test
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.5, stratify=y,
                     random_state=random_state)


# Reduce dimension to 2 with KernelDiscriminantAnalysis
# can adjust the value of 'gamma' as needed.
kda = make_pipeline(StandardScaler(),
                    KernelDiscriminantAnalysis(n_components=2, gamma = 0.000005))

# Use a nearest neighbor classifier to evaluate the methods
knn = KNeighborsClassifier(n_neighbors=n_neighbors)


plt.figure()
# plt.subplot(1, 3, i + 1, aspect=1)

# Fit the method's model
kda.fit(X_train, y_train)

# Fit a nearest neighbor classifier on the embedded training set
knn.fit(kda.transform(X_train), y_train)

# Compute the nearest neighbor accuracy on the embedded test set
acc_knn = knn.score(kda.transform(X_test), y_test)

# Embed the data set in 2 dimensions using the fitted model
X_embedded = kda.transform(X)

# Plot the projected points and show the evaluation score
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap='Set1')
plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format('kda',
                                                              n_neighbors,
                                                              acc_knn))
plt.show()
