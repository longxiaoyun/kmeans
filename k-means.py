#-*- coding:utf-8 -*-

# Author:longjiang
import numpy as np
import matplotlib.pyplot as plt
import math

# 计算两个向量的欧式距离
def distince(point1,point2):
    return np.sqrt(np.sum(np.square(point1-point2)))

# 随机生成k个簇心
def rand_center(data,k):
    # 数据集维度
    n = data.shape[1]
    # 创建一个k*n维数组，0填充
    centroids = np.zeros((k, n))
    for i in range(n):
        dmin, dmax = np.min(data[:, i]), np.max(data[:, i])
        centroids[:, i] = dmin + (dmax - dmin) * np.random.rand(k)
    return centroids



# kmeans 算法
def kmeans(data,k):
    # 获得行数m
    m = data.shape[0]
    # 初试化一个矩阵，用来记录簇索引和存储误差
    clusterAssment = np.mat(np.zeros((m,2)))
    # 随机的得到一个质心矩阵蔟
    centroids = rand_center(data,k)

    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 对每个数据点寻找最近的质心
        for i in range(m):
            # 正无穷
            minDist = float("inf")
            minIndex = -1
            # 遍历质心蔟，寻找最近的质心
            for j in range(k):
                # 计算数据点和质心的欧式距离
                distJ1 = distince(centroids[j,:],data[i,:])
                if distJ1 < minDist:
                    minDist = distJ1
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print centroids
        # 更新质心，将每个族中的点的均值作为质心
        for ci in range(k):
            # 取出样本所属簇的索引值
            index_all = clusterAssment[:,0].A
            # 取出所有属于第ci个簇的索引值
            value = np.nonzero(index_all==ci)
            # 取出属于第i个簇的所有样本点
            sampleInClust = data[value[0]]
            centroids[ci,:] = np.mean(sampleInClust, axis=0)
    # 返回的第一个变量是质心，第二个是各个簇的分布情况
    return centroids, clusterAssment


def plt_img(data, k, centroids, clusterAssment):
    numSamples, dim = data.shape
    if dim != 2:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print "Sorry! Your k is too large! please contact Zouxy"
        return 1

    # draw all samples
    for i in xrange(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(data[i, 0], data[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)

    plt.show()

# 舞剑度
if __name__=="__main__":
    # 加载数据集
    dataSet = []
    k=4
    fileIn = open('data.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split('\t')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])

    data = np.mat(dataSet)
    # 计算
    centroids, clusterAssment = kmeans(data, k)
    # 显示
    plt_img(data,k,centroids,clusterAssment)








