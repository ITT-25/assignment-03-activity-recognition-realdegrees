import pandas as pd
import os
import glob

# Directory containing the CSV files
DATA_DIR = "data"

# Find all CSV files in the data directory
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))


# Function to resample a single CSV file
def resample_csv(file_path):
    print(f"Processing file: {file_path}")

    file_name = os.path.basename(file_path)
    print(f"  Processing: {file_name}")

    try:
        # read csv
        df = pd.read_csv(file_path)

        # convert timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        # resample at 100 Hz
        df.set_index("timestamp", inplace=True)
        df = df.resample("10ms").mean().interpolate()

        # reset index → bring timestamps back as a column
        df.reset_index(inplace=True)

        # convert back to integer ms
        df["timestamp"] = (df["timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1ms")

        df.to_csv(file_path, index=True)

        print(f"  Successfully resampled and saved back to {file_path}")
    except Exception as e:
        print(f"  Error processing {file_path}: {str(e)}")


# Process all CSV files in the data directory
if __name__ == "__main__":
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR} directory.")
    else:
        print(f"Found {len(csv_files)} CSV files to process.")

        for csv_file in csv_files:
            resample_csv(csv_file)

        print("All files processed.")
