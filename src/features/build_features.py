import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction

from FrequencyAbstraction import FourierTransformation
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df=pd.read_pickle("../../data/interim/02_outliers_removed.pkl")


pred_col=list(df.columns[:6])

pred_col
df.info()

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in pred_col:
   df[col]=df[col].interpolate()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df.info()
df = df.dropna()

df.info()

for s in df['set'].unique():
    start=df[df['set']==s].index[0]
    end=df[df['set']==s].index[-1]
    duration=end-start
    df.loc[df['set']==s,'duration']=duration.seconds
    
duration_df=df.groupby('category')['duration'].mean()
# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass=df.copy()
Lowpass=LowPassFilter()
fs=1000/20
cutoff=1
df_lowpass = Lowpass.low_pass_filter(
    data_table=df_lowpass,
    col="accel_y",
    sampling_frequency=fs,
    cutoff_frequency=cutoff,
    order=4
)

subset=df_lowpass[df_lowpass['set']==45]
print(subset["label"][0])
for col in pred_col:
   df_lowpass=Lowpass.low_pass_filter(
       data_table=df_lowpass,
       col=col,
       sampling_frequency=fs,
       cutoff_frequency=cutoff,
       order=4
   )  
   df_lowpass[col]=df_lowpass[col+'_lowpass']
   df_lowpass=df_lowpass.drop(columns=[col+'_lowpass'])
   

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca=df_lowpass.copy()
PCA=PrincipalComponentAnalysis()

pc_values=PCA.determine_pc_explained_variance(df_pca,pred_col)
plt.figure(figsize=(8,5))
plt.plot(range(1,len(pc_values)+1),pc_values)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()


df_pca=PCA.apply_pca(df_pca,pred_col,3)
# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_square=df_pca.copy()
acc_r=df_square["accel_x"]**2+df_square["accel_y"]**2+df_square["accel_z"]**2
gyro_r=df_square["gyro_x"]**2+df_square["gyro_y"]**2+df_square["gyro_z"]**2
df_square["accel_r"]=np.sqrt(acc_r)
df_square["gyro_r"]=np.sqrt(gyro_r)
df_square

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
df_temp = df_square.copy()

# Numerical abstraction object
num_abstract = NumericalAbstraction()

# Predictor columns
pred_col = pred_col + ['accel_r', 'gyro_r']

# Window size (50 samples)
ws = int(1000 / 200)

df_temp_list = []

for s in df_temp['set'].unique():
    subset = df_temp[df_temp['set'] == s].copy()

    for col in pred_col:
        subset = num_abstract.abstract_numerical(
            subset, [col], ws, 'mean'
        )
        subset = num_abstract.abstract_numerical(
            subset, [col], ws, 'std'
        )

    df_temp_list.append(subset)

df_temp = pd.concat(df_temp_list)
subset[["accel_y","accel_y_temp_mean_ws_5","accel_y_temp_std_ws_5"]].plot()
# Inspect result
df_temp.info()

# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
df_freq = df_temp.copy().reset_index()
freqabs=FourierTransformation()

fs=int(1000/200)
ws=int(2800/200)

df_freq=freqabs.abstract_frequency(
    df_freq,
    ["accel_x"],
    ws,
    fs
)

subset=df_freq[df_freq['set']==15]
subset[["accel_y"]].plot()

df_freq_list = []

for s in df_temp['set'].unique():
    subset = df_temp[df_temp['set'] == s].reset_index(drop=True).copy()
    subset = freqabs.abstract_frequency(
        subset,
        pred_col,
        ws,
        fs
    )
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).reset_index(drop=True)

df_freq=df_freq.dropna()
df_freq=df_freq.iloc[::2]
# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df_cluster = df_freq.copy()

cluster_columns = ["accel_x", "accel_y", "accel_z"]
k_values = range(2, 10)
inertias = []

subset = df_cluster[cluster_columns]

for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    kmeans.fit(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias, marker='o')
plt.xlabel("k")
plt.ylabel("Sum of squared distances (Inertia)")
plt.title("Elbow Method for Optimal k")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)

cluster_columns = ["accel_x", "accel_y", "accel_z"]
subset = df_cluster[cluster_columns]

df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plot clusters in 3D
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for cluster_id in sorted(df_cluster["cluster"].unique()):
    cluster_data = df_cluster[df_cluster["cluster"] == cluster_id]
    ax.scatter(
        cluster_data["accel_x"],
        cluster_data["accel_y"],
        cluster_data["accel_z"],
        label=f"Cluster {cluster_id}",
        s=30
    )

ax.set_xlabel("X-axis (acc_x)")
ax.set_ylabel("Y-axis (acc_y)")
ax.set_zlabel("Z-axis (acc_z)")
ax.set_title("3D KMeans Clustering")
ax.legend()

plt.show()

df_cluster.to_pickle("../../data/interim/03_data_feature.pkl")