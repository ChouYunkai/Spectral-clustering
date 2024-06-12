import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# 加载数字数据集
digits = load_digits(n_class=6)
X = digits.data  # 特征矩阵

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
plt.figure(figsize=(12, 4))  # 设置图形大小
plt.subplot(1, 3, 1)
plt.imshow(img, cmap=plt.cm.binary)
plt.title('A selection from the 64-dimensional digits dataset')

# TSNE嵌入到三维
tsne = TSNE(n_components=3, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)


def plot_embedding_3d(X, title=None):
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    # 降维后的坐标为（X[i, 0], X[i, 1],X[i,2]），在该位置画出对应的digits
    fig = plt.subplot(1, 3, 2, projection='3d')
    for i in range(X.shape[0]):
        fig.text(X[i, 0], X[i, 1], X[i, 2], str(digits.target[i]),
                 color=plt.cm.Set1(digits.target[i] / 10.),  # 修改这里
                 fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)


plot_embedding_3d(X_tsne, "t-SNE 3D ")

# K均值聚类
kmeans = KMeans(n_clusters=6, random_state=0)
kmeans.fit(X)
X_kmeans = kmeans.transform(X)

# 绘制K均值聚类结果
plt.subplot(1, 3, 3)
plt.scatter(X_kmeans[:, 0], X_kmeans[:, 1], c=kmeans.labels_, cmap='viridis', marker='.')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()  # 调整布局，使子图之间不重叠
plt.show()  # 显示图形
