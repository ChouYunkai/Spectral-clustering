import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
from sklearn.cluster import SpectralClustering

class img_seg():
    def load_img(self, path):
        print("Loading image...")
        sample_img = cv2.imread(path)
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
        sample_img = sample_img / 255.0

        # 显示原始图像
        plt.figure()
        plt.imshow(sample_img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        plt.show()

        # 对图像进行高斯模糊去噪
        print("Applying Gaussian blur...")
        sample_img_blur = cv2.GaussianBlur(sample_img, (5, 5), 0)

        # 显示高斯模糊后的图像
        plt.figure()
        plt.imshow(sample_img_blur, cmap='gray')
        plt.title('Blurred Image')
        plt.axis('off')
        plt.show()

        # 用img_to_graph将img转化为graph，每个位置计算的是相邻像素点这之间的差（梯度）
        print("Converting image to graph...")
        graph = image.img_to_graph(sample_img_blur)

        # 显示图形的邻接矩阵
        plt.figure()
        plt.spy(graph, markersize=0.5)
        plt.title('Adjacency Matrix')
        plt.show()

        # 转化为邻接矩阵，这里做了归一化来保证更好的效果
        print("Normalizing graph...")
        gamma = 20
        graph.data = np.exp(-gamma * graph.data / graph.data.std())

        return sample_img_blur, graph

    def visualize(self, img, labels):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(labels, cmap=plt.cm.nipy_spectral)
        plt.title('Segmentation Result')
        plt.axis('off')
        plt.show()

    def spectral_clustering_seg(self, path):
        print("Starting spectral clustering segmentation...")
        img, graph = self.load_img(path)
        # 运行谱聚类
        print("Running spectral clustering...")
        spectral = SpectralClustering(n_clusters=5, affinity='precomputed', random_state=0)
        labels = spectral.fit_predict(graph)
        labels = labels.reshape(img.shape)

        # 显示分割结果
        self.visualize(img, labels)
        print("Segmentation complete!")

_ = img_seg()
_.spectral_clustering_seg('2.jpg')  # 更正文件扩展名为.jpg
