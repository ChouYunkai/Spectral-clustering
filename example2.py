import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering, KMeans

# 加载数字数据集
digits = load_digits(n_class=6)
X = digits.data  # 特征矩阵
y = digits.target  # 目标向量

# 设置每行图片的数量
n_img_per_row = 20

# 创建一个大图像
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))

# 填充图像
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        if i * n_img_per_row + j < X.shape[0]:
            img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

# 显示图像
plt.figure(figsize=(18, 4))  # 设置图形大小
plt.subplot(1, 5, 1)
plt.imshow(img, cmap=plt.cm.binary)
plt.title('A selection from the 64-dimensional digits dataset')

# TSNE嵌入到三维
tsne = TSNE(n_components=3, init='pca', random_state=0)
X_tsne_3d = tsne.fit_transform(X)

# 绘制3D的TSNE降维结果
ax = plt.subplot(1, 5, 2, projection='3d')
ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=y, cmap='viridis', marker='.')
ax.set_title('t-SNE 3D')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

# TSNE嵌入到二维
tsne = TSNE(n_components=2, init='pca', random_state=0)
X_tsne_2d = tsne.fit_transform(X)

# 绘制TSNE降维的散点图（用于原始标签）
plt.subplot(1, 5, 3)
plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=y, cmap='viridis', marker='.')
plt.title('TSNE 2D (Original Labels)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Spectral Clustering
spectral = SpectralClustering(n_clusters=6, affinity='nearest_neighbors', random_state=0)
spectral.fit(X)

# 绘制Spectral Clustering结果
plt.subplot(1, 5, 4)
plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=spectral.labels_, cmap='viridis', marker='.')
plt.title('Spectral Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# K-means Clustering
kmeans = KMeans(n_clusters=6, random_state=0)
kmeans.fit(X)

# 绘制K-means Clustering结果
plt.subplot(1, 5, 5)
plt.scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], c=kmeans.labels_, cmap='viridis', marker='.')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()  # 调整布局，使子图之间不重叠
plt.show()  # 显示图形
