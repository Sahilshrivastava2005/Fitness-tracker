import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from LearningAlgorithm import ClassificationAlgorithms

# --------------------------------------------------------------
# Plot settings
# --------------------------------------------------------------
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/03_data_feature.pkl")

# --------------------------------------------------------------
# Prepare dataset
# --------------------------------------------------------------
df_train = df.drop(["participant", "category", "set"], axis=1)

X = df_train.drop("label", axis=1)
y = df_train["label"]

# ---- Magnitude features ----
X["acc_r"] = np.sqrt(
    X["accel_x"] ** 2 + X["accel_y"] ** 2 + X["accel_z"] ** 2
)

X["gyr_r"] = np.sqrt(
    X["gyro_x"] ** 2 + X["gyro_y"] ** 2 + X["gyro_z"] ** 2
)

# --------------------------------------------------------------
# Train / Test split
# --------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# --------------------------------------------------------------
# PCA FEATURES (CRITICAL FIX)
# --------------------------------------------------------------
pca = PCA(n_components=3)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

X_train[["pca1", "pca2", "pca3"]] = X_train_pca
X_test[["pca1", "pca2", "pca3"]] = X_test_pca

# --------------------------------------------------------------
# Feature groups
# --------------------------------------------------------------
basic_features = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
square_features = ["acc_r", "gyr_r"]
pca_features = ["pca1", "pca2", "pca3"]

time_features = [c for c in X_train.columns if "_temp_" in c]
freq_features = [c for c in X_train.columns if ("_freq_" in c) or ("_pse" in c)]
cluster_features = ["cluster"]

# --------------------------------------------------------------
# Safe feature validation helper
# --------------------------------------------------------------
def valid_features(features, X):
    return [f for f in features if f in X.columns]

# --------------------------------------------------------------
# Feature sets
# --------------------------------------------------------------
feature_set_1 = valid_features(basic_features + square_features, X_train)
feature_set_2 = valid_features(basic_features + square_features + pca_features, X_train)
feature_set_3 = valid_features(
    basic_features + square_features + pca_features + time_features,
    X_train,
)
feature_set_4 = valid_features(
    basic_features
    + square_features
    + pca_features
    + time_features
    + freq_features
    + cluster_features,
    X_train,
)

# --------------------------------------------------------------
# Forward feature selection
# --------------------------------------------------------------
learner = ClassificationAlgorithms()

max_features = 10
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(ordered_scores) + 1), ordered_scores, marker="o")
plt.xticks(range(1, len(ordered_scores) + 1))
plt.xlabel("Number of Features Selected")
plt.ylabel("Cross-Validated Accuracy")
plt.title("Forward Feature Selection Performance")
plt.grid()
plt.show()

# --------------------------------------------------------------
# Model evaluation
# --------------------------------------------------------------
possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features,
]

feature_names = [
    "Feature Set 1",
    "Feature Set 2",
    "Feature Set 3",
    "Feature Set 4",
    "Selected Features",
]

iterations = 1

# --------------------------------------------------------------
# Score storage (CRITICAL FIX)
# --------------------------------------------------------------
score_df = pd.DataFrame(columns=["feature_set", "model", "accuracy"])

# --------------------------------------------------------------
# Training loop
# --------------------------------------------------------------
for feature_set, feature_name in zip(possible_feature_sets, feature_names):

    print("Feature set:", feature_name)

    # Safety check
    missing = set(feature_set) - set(X_train.columns)
    if missing:
        raise ValueError(f"{feature_name} missing features: {missing}")

    selected_train_X = X_train[feature_set]
    selected_test_X = X_test[feature_set]

    # ---- Non-deterministic models ----
    performance_test_nn = 0.0
    performance_test_rf = 0.0

    for it in range(iterations):
        print("\tTraining neural network,", it)
        _, class_test_y, _, _ = learner.feedforward_neural_network(
            selected_train_X, y_train, selected_test_X, gridsearch=False
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        _, class_test_y, _, _ = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn /= iterations
    performance_test_rf /= iterations

    # ---- Deterministic models ----
    print("\tTraining KNN")
    _, class_test_y, _, _ = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    _, class_test_y, _, _ = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    _, class_test_y, _, _ = learner.naive_bayes(
        selected_train_X, y_train, selected_test_X
    )
    performance_test_nb = accuracy_score(y_test, class_test_y)

    # ---- Save results ----
    new_scores = pd.DataFrame(
        {
            "feature_set": [feature_name] * 5,
            "model": ["NN", "RF", "KNN", "DT", "NB"],
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )

    score_df = pd.concat([score_df, new_scores], ignore_index=True)

# --------------------------------------------------------------
# Final results
# --------------------------------------------------------------
print("\nFinal model performance:")
print(score_df)

from sklearn.metrics import confusion_matrix

(class_train_y, class_test_y, prob_train_y, prob_test_y) = learner.random_forest(
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=False
)

accuracy=accuracy_score(y_test, class_test_y)

classes=class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2f})")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)     

