import sys
from pathlib import Path
import pandas as pd

# Ensure we can import config
sys.path.append(str(Path(__file__).resolve().parent))
import config

def main():
    test_file = config.DATA_RAW_DIR / "skills.csv"
    
    print(f"Testing CSV read from: {test_file}")
    
    try:
        if not test_file.exists():
            print(f"Warning: File {test_file} does not exist. Cannot test reading.")
            return
            
        df = pd.read_csv(test_file)
        print("\nSuccessfully read CSV! Here are the first few rows:")
        print(df.head())
        print(f"\nTotal rows loaded: {len(df)}")
        
    except Exception as e:
        print(f"Error reading CSV: {e}")

if __name__ == "__main__":
    main()
