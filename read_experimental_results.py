#!/usr/bin/env python3
"""
Script to read and process all data from EXPERIMENTAL_RESULT.xlsx
"""

import pandas as pd
import os

def read_excel_file(file_path):
    """Read all sheets from the Excel file and return a dictionary of DataFrames."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")
    
    print(f"Reading Excel file: {file_path}")
    
    # Load the Excel file
    excel_file = pd.ExcelFile(file_path)
    print(f"Found sheets: {excel_file.sheet_names}")
    
    # Dictionary to store all DataFrames
    dataframes = {}
    
    # Read each sheet
    for sheet_name in excel_file.sheet_names:
        print(f"\nReading sheet: '{sheet_name}'")
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        dataframes[sheet_name] = df
        
        # Print basic info
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Show first few rows for small DataFrames
        if df.shape[0] <= 10:
            print(f"  Data:\n{df}")
        else:
            print(f"  First 3 rows:\n{df.head(3)}")
            print(f"  Last 3 rows:\n{df.tail(3)}")
    
    return dataframes

def save_as_csv(dataframes, output_dir="experimental_results_csv"):
    """Save each DataFrame as a CSV file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\nSaving DataFrames as CSV files in '{output_dir}' directory:")
    
    for sheet_name, df in dataframes.items():
        # Clean sheet name for filename
        safe_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        file_path = os.path.join(output_dir, f"{safe_name}.csv")
        
        df.to_csv(file_path, index=False)
        print(f"  Saved '{sheet_name}' to '{file_path}'")

def main():
    """Main function to read and process the Excel file."""
    excel_path = "Experimental result-20260526T064624Z-3-001/Experimental result/EXPERIMENTAL_RESULT.xlsx"
    
    try:
        # Read all data
        dataframes = read_excel_file(excel_path)
        
        # Save as CSV files
        save_as_csv(dataframes)
        
        print("\nProcessing complete!")
        
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())