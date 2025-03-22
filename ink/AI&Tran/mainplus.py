import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# 自定义多选场景处理器
def process_scenes(X):
    aigc = X['使用AIGC主要场景'].str.get_dummies(sep='，').add_prefix('AIGC_')
    trad = X['使用传统问答平台主要场景'].str.get_dummies(sep='，').add_prefix('TRAD_')
    return pd.concat([aigc, trad], axis=1)


# 构建特征工程管道
preprocessor = ColumnTransformer([
    ('scenes', FunctionTransformer(process_scenes), ['使用AIGC主要场景', '使用传统问答平台主要场景']),
    ('ordinal', StandardScaler(), ['年龄', '月收入', '最高学历', '使用AI频率', '使用传统问答平台频次']),
    ('cat', 'passthrough', ['性别', '职业', '院校类型'])
])

# 读取数据
df = pd.read_excel("data.xls")

# 预处理数据
preprocessed_data = preprocessor.fit_transform(df)

# PCA降维（保留95%方差）
pca = PCA(n_components=0.95)
pca_data = pca.fit_transform(preprocessed_data)


# 自动确定最佳K值（肘部法则）
def find_optimal_k(data, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    # 计算二阶导数找拐点
    deltas = np.diff(inertias)
    deltas_ratio = deltas[1:] / deltas[:-1]
    optimal_k = np.argmin(deltas_ratio) + 2  # +2补偿两次diff操作

    # 可视化肘部曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_k + 1), inertias, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal K')
    plt.axvline(optimal_k, color='r', linestyle='--')
    plt.show()

    return optimal_k


optimal_k = find_optimal_k(pca_data, max_k=8)
print(f"自动确定最佳簇数：{optimal_k}")

# 执行聚类
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['cluster'] = kmeans.fit_predict(pca_data)

# 3D可视化
pca3d = PCA(n_components=3).fit_transform(preprocessed_data)
ax = plt.figure().add_subplot(111, projection='3d')
scatter = ax.scatter(pca3d[:, 0], pca3d[:, 1], pca3d[:, 2],
                     c=df['cluster'], cmap='viridis', s=50)
plt.colorbar(scatter, label='Cluster')
ax.set_xlabel('PC1 (经济实力维度)')
ax.set_ylabel('PC2 (平台偏好维度)')
ax.set_zlabel('PC3 (教育背景维度)')
plt.title(f'3D Cluster Visualization (K={optimal_k})')
plt.show()


# 生成群体特征报告
def analyze_clusters(df):
    # 数值特征
    num_stats = df.groupby('cluster').agg({
        '年龄': ['mean', lambda x: round(x.mean())],
        '月收入': ['mean', lambda x: round(x.mean())],
        '使用AI频率': ['mean', lambda x: round(x.mean())]
    })

    # 分类特征
    cat_stats = df.groupby('cluster').agg({
        '职业': lambda x: x.mode()[0],
        '院校类型': lambda x: x.mode()[0]
    })

    # 场景特征
    scenes = process_scenes(df)
    scene_stats = scenes.groupby(df['cluster']).mean()

    # 合并结果
    report = pd.concat([
        num_stats,
        cat_stats,
        scene_stats.add_prefix('场景比例_')
    ], axis=1)

    # 列名格式化
    report.columns = [
        '平均年龄', '主要年龄段',
        '平均月收入', '主要收入段',
        '平均AI频率', '主要AI频率',
        '主要职业', '主要院校类型',
        'AIGC学习', 'AIGC工作', 'AIGC技术', 'AIGC生活',
        '传统学习', '传统工作', '传统技术', '传统生活'
    ]

    return report


# 输出分析报告
cluster_report = analyze_clusters(df)
print("\n聚类群体特征分析：")
print(cluster_report.round(2))

# 保存结果
cluster_report.to_csv("auto_cluster_report.csv", encoding='utf-8-sig')