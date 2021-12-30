import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


def clust():
    # Обработка данных
    data = pd.read_csv('USArrests.csv', sep=',', encoding='windows-1251', skipinitialspace=True)
    labels = []
    code = []
    for i in range(50):
        labels.append(data.loc[i]['State'])
        code.append(i)
    data.drop('State', axis=1, inplace=True)
    data['code'] = code

    # Дендрограмма
    m = data.mean(axis=1)
    for i, col in enumerate(data):
        data.iloc[:, i] = data.iloc[:, i].fillna(m)
    x = data.iloc[:, 1: -1].values
    link = linkage(x, method='average', metric='euclidean')
    plt.figure(figsize=(10, 10))
    dendrogram(link, labels=labels)
    plt.show()
    label = fcluster(link, 70, criterion='distance')
    np.unique(label)
    data.loc[:, 'label'] = label

    # Метод k средних
    x = data.values[:, :]
    x = np.nan_to_num(x)
    sse = []
    for k in range(1, 10):
        estimator = KMeans(n_clusters=k)
        estimator.fit(x)
        sse.append(estimator.inertia_)
    r = range(1, 10)
    plt.plot(r, sse, 'o-')
    plt.show()

    # Определение кластеров
    clust_count = 2
    k_means = KMeans(init='k-means++', n_clusters=clust_count, n_init=12)
    k_means.fit(x)
    labels = k_means.labels_
    data['cluster'] = labels

    for i in range(clust_count):
        plt.scatter(data.loc[data['cluster'] == i]['code'], data.loc[data['cluster'] == i]['Murder'], label=i)
    plt.xlabel('State')
    plt.ylabel('Murder')
    plt.legend()
    plt.show()
    for i in range(clust_count):
        plt.scatter(data.loc[data['cluster'] == i]['code'], data.loc[data['cluster'] == i]['Assault'], label=i)
    plt.xlabel('State')
    plt.ylabel('Assault')
    plt.legend()
    plt.show()
    for i in range(clust_count):
        plt.scatter(data.loc[data['cluster'] == i]['code'], data.loc[data['cluster'] == i]['UrbanPop'], label=i)
    plt.xlabel('State')
    plt.ylabel('Urban Pop')
    plt.legend()
    plt.show()
    for i in range(clust_count):
        plt.scatter(data.loc[data['cluster'] == i]['code'], data.loc[data['cluster'] == i]['Rape'], label=i)
    plt.xlabel('State')
    plt.ylabel('Rape')
    plt.legend()
    plt.show()


clust()
