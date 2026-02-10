import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

df = pd.read_csv("student_performance_final.csv")

print(df.info())
print(df.head())

df["Score_Trend"] = df["final_score"] - (
    df["math_score"] + df["science_score"] + df["english_score"]
) / 3

df["Subject_Std"] = df[
    ["math_score", "science_score", "english_score"]
].std(axis=1)

plt.figure(figsize=(8,5))
sns.histplot(df["final_score"], bins=20, kde=True)
plt.title("Final Score Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x="stress_level", y="final_score", data=df)
plt.title("Stress Level vs Final Score")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(
    x=pd.cut(df["attendance_percentage"], bins=4),
    y="final_score",
    data=df
)
plt.title("Attendance vs Final Score")
plt.xlabel("Attendance Range")
plt.show()

features = df[
    [
        "attendance_percentage",
        "study_hours_per_week",
        "homework_completion_rate",
        "class_participation",
        "quiz_average",
        "midterm_score",
        "final_score",
        "math_score",
        "science_score",
        "english_score",
        "stress_level",
        "Score_Trend",
        "Subject_Std"
    ]
]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

wcss = []
k_range = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_features)
    wcss.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(k_range, wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(scaled_features)
    sil_scores.append(silhouette_score(scaled_features, labels))

plt.figure(figsize=(8,5))
plt.plot(k_range, sil_scores, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(scaled_features)

cluster_summary = df.groupby("KMeans_Cluster")[features.columns].mean()
print(cluster_summary)

plt.figure(figsize=(12,6))
sns.heatmap(cluster_summary.T, annot=True, cmap="coolwarm")
plt.title("Cluster-wise Feature Averages")
plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_features)

df["PC1"] = pca_data[:, 0]
df["PC2"] = pca_data[:, 1]

plt.figure(figsize=(8,5))
sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="KMeans_Cluster",
    data=df,
    palette="viridis"
)
plt.title("KMeans Clusters (PCA Projection)")
plt.show()

dbscan = DBSCAN(eps=1.2, min_samples=20)
df["DBSCAN_Cluster"] = dbscan.fit_predict(scaled_features)

plt.figure(figsize=(8,5))
sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="DBSCAN_Cluster",
    data=df,
    palette="tab10"
)
plt.title("DBSCAN Clustering")
plt.show()

agg = AgglomerativeClustering(n_clusters=3)
df["Agglomerative_Cluster"] = agg.fit_predict(scaled_features)

plt.figure(figsize=(8,5))
sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="Agglomerative_Cluster",
    data=df,
    palette="Set2"
)
plt.title("Agglomerative Clustering")
plt.show()

plt.figure(figsize=(7,5))
df["KMeans_Cluster"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Cluster")
plt.ylabel("Number of Students")
plt.title("Students per Cluster")
plt.show()

def intervention_plan(row):
    if row["final_score"] < 40 and row["attendance_percentage"] < 60:
        return "High Risk – Immediate Counseling"
    elif row["final_score"] < 60:
        return "Moderate Risk – Extra Support"
    else:
        return "Low Risk – Self Learning"

df["Intervention_Plan"] = df.apply(intervention_plan, axis=1)

print(df["Intervention_Plan"].value_counts())

plt.figure(figsize=(8,5))
sns.boxplot(x="KMeans_Cluster", y="stress_level", data=df)
plt.title("Stress Level Across Clusters")
plt.show()

df.to_csv("student_performance_unsupervised_full_analysis.csv", index=False)
