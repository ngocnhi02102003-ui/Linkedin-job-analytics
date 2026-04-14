import sys
from pathlib import Path
import pandas as pd

# Ensure we can import config
sys.path.append(str(Path(__file__).resolve().parent))
import config

def main():
    data_dir = config.DATA_RAW_DIR
    print(f"Scanning directory: {data_dir}\n")
    
    # Check if directory exists
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        return
        
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in directory.")
        return
        
    for index, file_path in enumerate(csv_files, 1):
        table_name = file_path.name
        file_exists = file_path.exists()
        
        print("-" * 50)
        print(f"[{index}] Table name: {table_name}")
        print(f"Path: {file_path}")
        print(f"Exists: {file_exists}")
        
        if file_exists:
            try:
                # Read only first 5 rows to be fast and memory-efficient
                df = pd.read_csv(file_path, nrows=5)
                print(f"Columns: {list(df.columns)}")
                print("First 5 rows:")
                print(df)
            except Exception as e:
                print(f"Error reading file: {e}")
                
        print() # Empty line for readability

if __name__ == "__main__":
    main()
