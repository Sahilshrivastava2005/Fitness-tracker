import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display

df = pd.read_pickle('../../data/interim/01_dataprocessed.pkl')
df.head()

set_df=df[df['set']==1]
set_df
plt.plot(set_df['accel_y'])
plt.plot(set_df['accel_y'].reset_index(drop=True))

for label in df["label"].unique():
    subset=df[df["label"]==label]
    fig, ax = plt.subplots()
    plt.plot(subset['accel_y'].reset_index(drop=True),label=label)
    plt.legend()
    plt.show()
for label in df["label"].unique():
    subset=df[df["label"]==label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]['accel_y'].reset_index(drop=True),label=label)
    plt.legend()
    plt.show()



category_df=df.query("label=='squat'").query("participant=='A'").reset_index(drop=True)
fig,ax=plt.subplots()
category_df.groupby('category')['accel_y'].plot()
ax.set_ylabel("accel_y")
ax.set_xlabel("sample")
plt.legend()
plt.show()


participant_df=df.query("label=='bench'").sort_values('participant').reset_index(drop=True)
fig,ax=plt.subplots()
participant_df.groupby('participant')['accel_y'].plot()
ax.set_ylabel("accel_y")
ax.set_xlabel("sample")
plt.legend()
plt.show()


label = "squat"
participant = "A"

all_axes_df = (
    df
    .query("label == @label and participant == @participant")
    .reset_index(drop=True)
)

fig, ax = plt.subplots()

all_axes_df[["accel_x", "accel_y", "accel_z"]].plot(ax=ax)

ax.set_ylabel("acceleration")
ax.set_xlabel("sample")
ax.legend()

plt.show()


labels = df["label"].unique()
participants = df["participant"].unique()
sensors = ["accel_x", "accel_y", "accel_z"]

for label in labels:
    for participant in participants:
        subset = (
                df
                .query("label == @label and participant == @participant")
                .reset_index(drop=True)
            )
        if len(subset) == 0:
            continue
        fig, ax = plt.subplots()
        all_axes_df = subset[sensors].plot(ax=ax)
        ax.set_ylabel("acceleration")
        ax.set_xlabel("sample")
        plt.title(f"Exercise: {label} - Participant: {participant}")
        plt.legend()
        plt.show()

sensors = ["gyro_x", "gyro_y", "gyro_z"]
for label in labels:
    for participant in participants:
        subset = (
                df
                .query("label == @label and participant == @participant")
                .reset_index(drop=True)
            )
        if len(subset) == 0:
            continue
        fig, ax = plt.subplots()
        all_axes_df = subset[sensors].plot(ax=ax)
        ax.set_ylabel("acceleration")
        ax.set_xlabel("sample")
        plt.title(f"Exercise: {label} - Participant: {participant}")
        plt.legend()
        plt.show()

labels = df["label"].unique()

participants = df["participant"].unique()

for label in labels:
    for participant in participants:

        combined_plot_df = (
            df.query("label == @label")
              .query("participant == @participant")
              .reset_index(drop=True)
        )

        if combined_plot_df.empty:
            continue

        fig, ax = plt.subplots(
            nrows=2,
            sharex=True,
            figsize=(20, 10)
        )

        # Accelerometer
        combined_plot_df[["accel_x", "accel_y", "accel_z"]].plot(ax=ax[0])
        ax[0].set_title(f"Accelerometer | Label: {label} | Participant: {participant}")
        ax[0].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=3,
            fancybox=True,
            shadow=True
        )

        # Gyroscope
        combined_plot_df[["gyro_x", "gyro_y", "gyro_z"]].plot(ax=ax[1])
        ax[1].set_title("Gyroscope")
        ax[1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=3,
            fancybox=True,
            shadow=True
        )
        plt.savefig(f"../../reports/figures/{label}_{participant}_sensors.png")
        plt.close()