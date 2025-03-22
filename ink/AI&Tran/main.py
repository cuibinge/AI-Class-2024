import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 自定义多选场景处理器
def process_scenes(X):
    aigc = X['使用AIGC主要场景'].str.get_dummies(sep='，').add_prefix('AIGC_')
    trad = X['使用传统问答平台主要场景'].str.get_dummies(sep='，').add_prefix('TRAD_')
    return pd.concat([aigc, trad], axis=1)

# 构建特征工程管道
preprocessor = ColumnTransformer([
    ('scenes', FunctionTransformer(process_scenes), ['使用AIGC主要场景','使用传统问答平台主要场景']),
    ('ordinal', StandardScaler(), ['年龄','月收入','最高学历','使用AI频率','使用传统问答平台频次']),
    ('cat', 'passthrough', ['性别','职业','院校类型'])  # 已提前做目标编码
])

# 构建完整Pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('pca', PCA(n_components=0.95)),  # 保留95%方差
    ('cluster', KMeans(n_clusters=3, init='k-means++'))
])

# 执行聚类
df = pd.read_excel("data.xls")
pipeline.fit(df)
df['cluster'] = pipeline.named_steps['cluster'].labels_

# 3D可视化
pca3d = PCA(n_components=3).fit_transform(preprocessor.transform(df))
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(pca3d[:,0], pca3d[:,1], pca3d[:,2], c=df['cluster'], cmap='viridis')
plt.show()


# 生成各簇的统计特征分析
def analyze_clusters(df):
    # 数值型特征均值分析
    numerical_features = ['年龄', '月收入', '最高学历', '使用AI频率', '使用传统问答平台频次']
    numerical_means = df.groupby('cluster')[numerical_features].mean()

    # 类别型特征众数分析（取出现频率最高的类别）
    categorical_features = ['性别', '职业', '院校类型']
    categorical_modes = df.groupby('cluster')[categorical_features].agg(lambda x: x.mode()[0])

    # 场景特征出现频率分析
    scene_features = process_scenes(df)
    scene_features['cluster'] = df['cluster']
    scene_means = scene_features.groupby('cluster').mean().add_prefix('比例_')

    # 合并所有分析结果
    cluster_profile = pd.concat([numerical_means, categorical_modes, scene_means], axis=1)

    # 添加解释性列名
    cluster_profile.columns = [
        '平均年龄编码', '平均月收入编码', '平均学历编码',
        '平均AI使用频率', '平均传统平台使用频次',
        '主要性别', '主要职业', '主要院校类型',
        'AIGC学习场景比例', 'AIGC工作场景比例', 'AIGC娱乐场景比例', 'AIGC生活场景比例',
        '传统平台学习场景比例', '传统平台工作场景比例', '传统平台娱乐场景比例', '传统平台生活场景比例'
    ]
    return cluster_profile


# 执行分析并打印结果
cluster_analysis = analyze_clusters(df)
print(cluster_analysis.round(2))

# 保存到CSV
cluster_analysis.to_csv("cluster_report.csv",
             encoding='utf-8-sig')
