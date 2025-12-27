import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_dataprocessed.pkl")

outlier_columns = list(df.columns[0:6]) # Select columns to check for outliers

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------

df[["gyro_y", "label"]].boxplot(by="label")

df[outlier_columns[:3] + ["label"]].boxplot(
    by="label",
    layout=(1, 3),
    figsize=(15, 5)
)
df[outlier_columns[3:6] + ["label"]].boxplot(
    by="label",
    layout=(1, 3),
    figsize=(15, 5)
)


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


# Plot a single column
col="accel_x"
dataset=mark_outliers_iqr(df, col)
plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)

# Loop over all columns

for col in outlier_columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)


# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


# Check for normal distribution
df[outlier_columns[:3] + ["label"]].hist(
    by="label",
    layout=(2, 3),
    figsize=(15, 8),
    bins=30
)

df[outlier_columns[3:6] + ["label"]].hist(
    by="label",
    layout=(2, 3),
    figsize=(15, 8),
    bins=30
)


for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)


# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def mark_outliers_lof(dataset, columns, n_neighbors=20):
    dataset = dataset.copy()

    # 1️⃣ Keep only numeric
    data = dataset[columns].apply(pd.to_numeric, errors="coerce")

    # 2️⃣ Remove inf / nan
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # 3️⃣ Scale (CRITICAL for LOF)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # 4️⃣ Safe neighbor count
    n = min(n_neighbors, len(X_scaled) - 1)

    lof = LocalOutlierFactor(n_neighbors=n)
    outliers = lof.fit_predict(X_scaled)
    X_scores = lof.negative_outlier_factor_

    # 5️⃣ Align back to original dataset
    dataset["outlier_lof"] = 0
    dataset.loc[data.index, "outlier_lof"] = (outliers == -1).astype(int)

    return dataset, outliers, X_scores

# Loop over all columns
dataset_lof, outliers, X_scores = mark_outliers_lof(df, outlier_columns)

for col in outlier_columns:
    plot_binary_outliers(
        dataset_lof,
        col,
        "outlier_lof",
        reset_index=True
    )

# Create a loop

outlier_removed_df = df.copy()
for col in outlier_columns:
    for label in df["label"].unique():
        dataset_label = mark_outliers_chauvenet(df[df["label"] == label], col)
        dataset_label.loc[dataset_label[col + "_outlier"], col] = np.nan
        outlier_removed_df.loc[outlier_removed_df["label"] == label, col] = dataset_label[col]
        n_outliers = len(df)-len(outlier_removed_df[col].dropna())
        
        
outlier_removed_df.to_pickle("../../data/interim/02_outliers_removed.pkl")
# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
