#!/usr/bin/env python3
# filepath: /home/fasc/Uni/ITT/assignment-03-activity-recognition-realdegrees/cleanup_duplicates_fixed.py

import pandas as pd
import os
import glob
import csv

# Directory containing the CSV files
DATA_DIR = "data"

# Find all CSV files in the data directory
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

# Function to fix a single CSV file with duplicate ID columns
def fix_duplicate_columns(file_path):
    print(f"Processing file: {file_path}")
    
    file_name = os.path.basename(file_path)
    print(f"  Processing: {file_name}")

    try:
        # First check the raw header line to detect duplicate columns
        with open(file_path, 'r') as f:
            header_line = f.readline().strip()
            # Skip comment line if present
            if header_line.startswith("//"):
                header_line = f.readline().strip()
            
            raw_columns = header_line.split(',')
            
            # Check for duplicates
            seen = set()
            duplicates = []
            for col in raw_columns:
                if col in seen:
                    duplicates.append(col)
                else:
                    seen.add(col)
            
            if duplicates:
                print(f"  Found duplicate columns: {duplicates}")
                
                # Create a list of unique column names preserving the original order
                unique_columns = []
                seen = set()
                for col in raw_columns:
                    if col not in seen:
                        seen.add(col)
                        unique_columns.append(col)
                
                # Read the data while handling the duplicate columns
                df = pd.read_csv(file_path)
                
                # If pandas auto-renamed columns, we need to map them back
                # Create a subset with only the columns we want
                df_cleaned = pd.DataFrame()
                for i, col in enumerate(unique_columns):
                    # Find the actual column name pandas might have used
                    if i < len(df.columns):
                        if col in df.columns:
                            df_cleaned[col] = df[col]
                        else:
                            # Try to find it with a suffix
                            for df_col in df.columns:
                                if df_col.startswith(f"{col}."):
                                    df_cleaned[col] = df[df_col]
                                    break
                            else:
                                # If not found, use positional index
                                df_cleaned[col] = df.iloc[:, i]
                
                # Save the cleaned data
                df_cleaned.to_csv(file_path, index=False)
                print(f"  Successfully cleaned duplicate columns in {file_path}")
            else:
                print(f"  No duplicate columns found in {file_name}")
    except Exception as e:
        print(f"  Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()

# Process all CSV files in the data directory
if __name__ == "__main__":
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR} directory.")
    else:
        print(f"Found {len(csv_files)} CSV files to process.")
        
        for csv_file in csv_files:
            fix_duplicate_columns(csv_file)
            
        print("All files processed.")
