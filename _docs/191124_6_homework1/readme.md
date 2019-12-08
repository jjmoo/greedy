基础小作业，没有思路的可以找班主任要Jerry老师的视频描述


要求如下：

作业截至时间：12月8日23：59，具体答案将会在下一周公布

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
    
结果：
1.要求识别成功率：100%
2.达到如图所示效果图
