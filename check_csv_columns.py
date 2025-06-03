#!/usr/bin/env python3
import pandas as pd
import os
import glob

print("Checking available CSV files in the current directory...")
csv_files = glob.glob("*.csv")

if not csv_files:
    print("No CSV files found in the current directory.")
    print("Please make sure your data files are in the correct location.")
else:
    print(f"Found {len(csv_files)} CSV files:")
    
    for file in csv_files:
        print(f"\n--- {file} ---")
        try:
            df = pd.read_csv(file)
            print(f"Shape: {df.shape} (rows, columns)")
            print("Columns:")
            for col in df.columns:
                print(f"- {col}")
                
            # If this is the team stats file, check for specific columns
            if "team" in file.lower() and "stats" in file.lower():
                print("\nChecking for expected team stats columns:")
                expected_cols = ['team_rank', 'market_value', 'avg_attendance', 'historical_success']
                for col in expected_cols:
                    similar_cols = [c for c in df.columns if col.lower() in c.lower()]
                    if col in df.columns:
                        print(f"✅ '{col}' found")
                    elif similar_cols:
                        print(f"⚠️ '{col}' not found, but similar columns exist: {similar_cols}")
                    else:
                        print(f"❌ '{col}' not found")
        except Exception as e:
            print(f"Error reading file: {str(e)}")

print("\nBased on the above information, you may need to update the notebook code to match the actual column names in your data files.") 