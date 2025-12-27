import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
df1acc=pd.read_csv('../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv')

df1acc.head()

df1gyro=pd.read_csv('../../data/raw/MetaMotion/A-dead-heavy_MetaWear_2019-01-15T20.35.27.174_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv')
df1gyro.head()

# -------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files=glob('../../data/raw/MetaMotion/*.csv')

len(files)
# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
f=files[0]
dp="../../data/raw/MetaMotion/"
participant=f.split("-")[0].replace(dp,"")
label=f.split("-")[1]
category=f.split("-")[2].rstrip("123")

df=pd.read_csv(f)
df
df['participant']=participant
df['label']=label
df['category']=category
df
# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
acc_df=pd.DataFrame()
gyro_df=pd.DataFrame()

acc_set=1
gyro_set=1

for f in files:
    participant=f.split("-")[0].replace(dp,"")
    label=f.split("-")[1]
    category=f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    df=pd.read_csv(f)
    df['participant']=participant
    df['label']=label
    df['category']=category
    
    if "Accelerometer" in f:
        df['set']=acc_set
        acc_set+=1
        acc_df=pd.concat([acc_df,df])
    elif "Gyroscope" in f:
        df['set']=gyro_set
        gyro_set+=1
        gyro_df=pd.concat([gyro_df,df])
# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
acc_df.info()

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

acc_df.drop(
    columns=["epoch (ms)", "time (01:00)", "elapsed (s)"],
    inplace=True
)

gyro_df.drop(
    columns=["epoch (ms)", "time (01:00)", "elapsed (s)"],
    inplace=True
)

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files=glob('../../data/raw/MetaMotion/*.csv')
data_path='../../data/raw/MetaMotion/'
def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        # Parse metadata from filename
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = (
            f.split("-")[2]
            .rstrip("123")
            .rstrip("_MetaWear_2019")
        )

        # Read CSV
        df = pd.read_csv(f)

        # Add metadata columns
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        # Accelerometer files
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df], ignore_index=True)

        # Gyroscope files
        elif "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df], ignore_index=True)

    # Convert epoch to datetime index
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    # Drop redundant time columns
    acc_df.drop(
        columns=["epoch (ms)", "time (01:00)", "elapsed (s)"],
        inplace=True
    )

    gyr_df.drop(
        columns=["epoch (ms)", "time (01:00)", "elapsed (s)"],
        inplace=True
    )

    return acc_df, gyr_df

acc_df, gyro_df = read_data_from_files(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

df_merger=pd.concat([acc_df.iloc[:,:3], gyro_df], axis=1)
df_merger.head()
df_merger.columns=[
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "participant", "label", "category", "set"
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "accel_x": "mean",
    "accel_y": "mean",
    "accel_z": "mean",
    "gyro_x": "mean",
    "gyro_y": "mean",
    "gyro_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last"
}

# Ensure datetime index is sorted
df_merger = df_merger.sort_index()

# Split by day
days = [g for _, g in df_merger.groupby(pd.Grouper(freq="D"))]

# Resample each day
data_resampled = pd.concat(
    [day.resample("200ms").apply(sampling) for day in days]
)

# Remove empty resample windows (NaNs come from here)
data_resampled = data_resampled.dropna(subset=["set"])

# Safe integer conversion
data_resampled["set"] = data_resampled["set"].astype(int)

# Inspect result
data_resampled.info()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resampled.to_pickle('../../data/interim/01_dataprocessed.pkl')