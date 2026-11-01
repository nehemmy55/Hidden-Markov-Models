# Merge raw Sensor Logger exports into one CSV per recording in data/train and data/test.
import argparse
import glob
import os
import re

import pandas as pd

G = 9.80665                  # g -> m/s^2
TOLERANCE_NS = 20_000_000    # match accelerometer and gyroscope samples within 20 ms
FOLDER_NAME = re.compile(r"^(standing|walking|jumping|still)(\d*)-")


def read_sensor(folder, file_name, prefix):
    """Read one sensor file and rename its x, y, z columns to <prefix>x, <prefix>y, <prefix>z."""
    df = pd.read_csv(os.path.join(folder, file_name)).sort_values("time")
    return df.rename(columns={axis: prefix + axis for axis in "xyz"})


def merge_recording(folder):
    """Return one table with timestamp, ax, ay, az (m/s^2) and gx, gy, gz (rad/s)."""
    # the uncalibrated accelerometer keeps gravity, which shows how the phone is held
    acc = read_sensor(folder, "AccelerometerUncalibrated.csv", "a")
    gyr = read_sensor(folder, "Gyroscope.csv", "g")
    df = pd.merge_asof(acc[["time", "seconds_elapsed", "ax", "ay", "az"]], gyr[["time", "gx", "gy", "gz"]],
                       on="time", direction="nearest", tolerance=TOLERANCE_NS).dropna()
    df[["ax", "ay", "az"]] *= G
    return df.rename(columns={"seconds_elapsed": "timestamp"})[["timestamp", "ax", "ay", "az", "gx", "gy", "gz"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="raw", help="folder with the Sensor Logger exports")
    parser.add_argument("--out", default="data", help="output folder")
    parser.add_argument("--member", default="", help="optional prefix for file names")
    parser.add_argument("--test-idx", type=int, nargs="*", default=[5], help="recording numbers used as unseen test data")
    args = parser.parse_args()

    for folder in sorted(glob.glob(os.path.join(args.raw, "*"))):
        match = FOLDER_NAME.match(os.path.basename(folder))
        if not match or not os.path.isdir(folder):
            continue
        activity, number = match.group(1), int(match.group(2) or 1)
        df = merge_recording(folder)

        split = "test" if number in args.test_idx else "train"
        file_name = (f"{args.member}_" if args.member else "") + f"{activity}_{number:02d}.csv"
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        df.to_csv(os.path.join(args.out, split, file_name), index=False)

        seconds = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
        print(f"{split:5s} {file_name}  {len(df)} rows  {seconds:.1f}s")


if __name__ == "__main__":
    main()
