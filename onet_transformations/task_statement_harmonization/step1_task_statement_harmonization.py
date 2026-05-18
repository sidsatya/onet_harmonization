"""
This script harmonizes historical O*NET task statement data by mapping SOC codes
from various years to the 2010 and 2018 SOC standards.

The script performs the following steps:
1.  **Configuration**: Sets up file paths and defines the years of data to be processed.
2.  **Load Crosswalks**: Reads multiple SOC crosswalk files and creates dictionaries that map a single SOC code to a list of one or more corresponding codes to handle splits.
3.  **Define SOC Conversion Logic**: Creates a function that chains lookups through the crosswalk dictionaries to convert any given SOC code from its original year to a list of 2010 SOC codes.
4.  **Process Data in a Loop**:
    - Iterates through each year of O*NET task data (2003-2025).
    - Reads the corresponding CSV file.
    - Applies the conversion logic and uses `explode` to duplicate rows for each one-to-many mapping, ensuring all relationships are preserved.
    - Chains mappings to convert the resulting 2010 codes to the 2018 standard.
5.  **Combine and Clean Data**:
    - Concatenates all the yearly DataFrames into a single master DataFrame.
    - Selects a final set of relevant columns.
6.  **Analyze and Export**:
    - Calculates the frequency of each unique task statement.
    - Saves the unique task counts to a CSV file.
    - Saves a de-duplicated list of task IDs and statements.
    - Saves the final, harmonized dataset.
"""

import pandas as pd
import numpy as np
import os

# --- Configuration ---
ROOT_DIR = "/Users/sidsatya/dev/ailabor"
print(f"Using ROOT_DIR: {ROOT_DIR}")

ONET_DATA_DIR = os.path.join(ROOT_DIR, "data/onet")
TASK_STATEMENTS_DIR = os.path.join(ONET_DATA_DIR, "historical_onet_task_statements")
CROSSWALK_DIR = os.path.join(ONET_DATA_DIR, "onet_occsoc_crosswalks")
OUTPUT_DIR = os.path.join(ROOT_DIR, "onet_transformations/task_statement_harmonization/intermediate_data")

# --- Load raw crosswalks and build list mappings ---
print("Loading SOC crosswalks...")

# Helper to create a mapping dictionary from a crosswalk DataFrame
def create_list_mapping(df, key_col, val_col):
    return df.groupby(key_col)[val_col].apply(lambda s: s.str.strip().tolist()).to_dict()

# 2000->2006, 2006->2009, 2009->2010
cw_00_06 = pd.read_csv(os.path.join(CROSSWALK_DIR, "onet_2000_to_2006_crosswalk.csv"), dtype=str)
map_00_06 = create_list_mapping(cw_00_06, 'O*NET-SOC 2000 Code', 'O*NET-SOC 2006 Code')

cw_06_09 = pd.read_csv(os.path.join(CROSSWALK_DIR, "onet_2006_to_2009_crosswalk.csv"), dtype=str)
map_06_09 = create_list_mapping(cw_06_09, 'O*NET-SOC 2006 Code', 'O*NET-SOC 2009 Code')

cw_09_10 = pd.read_csv(os.path.join(CROSSWALK_DIR, "onet_2009_to_2010_crosswalk.csv"), dtype=str)
map_09_10 = create_list_mapping(cw_09_10, 'O*NET-SOC 2009 Code', 'O*NET-SOC 2010 Code')

# 2010->2019 and 2019->2010
cw_10_19 = pd.read_csv(os.path.join(CROSSWALK_DIR, "onet_2010_to_2019_crosswalk.csv"), dtype=str)
map_10_19 = create_list_mapping(cw_10_19, 'O*NET-SOC 2010 Code', 'O*NET-SOC 2019 Code')
map_19_10 = create_list_mapping(cw_10_19, 'O*NET-SOC 2019 Code', 'O*NET-SOC 2010 Code')

# 2019->2018
cw_19_18 = pd.read_csv(os.path.join(CROSSWALK_DIR, "onet_2019_to_2018_crosswalk.csv"), dtype=str)
map_19_18 = create_list_mapping(cw_19_18, 'O*NET-SOC 2019 Code', '2018 SOC Code')

print("Crosswalks loaded.")

# --- Crosswalk mapping functions --- 
def get_00_06_mapping(codes):
    mapped_codes = []
    for code in codes: 
        if code not in map_00_06:
            print(f"Code {code} not found in 2000-2006 mapping.")
            continue
        first_level_map = map_00_06[code]
        for subc in first_level_map:
            mapped_codes.append(subc)
    return list(set(mapped_codes))

def get_06_09_mapping(codes):
    mapped_codes = []
    for code in codes: 
        if code not in map_06_09:
            print(f"Code {code} not found in 2006-2009 mapping.")
            continue
        first_level_map = map_06_09[code]
        for subc in first_level_map:
            mapped_codes.append(subc)
    return list(set(mapped_codes))

def get_09_10_mapping(codes):
    mapped_codes = []
    for code in codes: 
        if code not in map_09_10:
            print(f"Code {code} not found in 2009-2010 mapping.")
            continue
        first_level_map = map_09_10[code]
        for subc in first_level_map:
            mapped_codes.append(subc)
    return list(set(mapped_codes))

def get_10_19_mapping(codes):
    mapped_codes = []
    for code in codes: 
        if code not in map_10_19:
            print(f"Code {code} not found in 2010-2019 mapping.")
            continue
        first_level_map = map_10_19[code]
        for subc in first_level_map:
            mapped_codes.append(subc)
    return list(set(mapped_codes))

def get_19_18_mapping(codes):
    mapped_codes = []
    for code in codes: 
        if code not in map_19_18:
            print(f"Code {code} not found in 2019-2018 mapping.")
            continue
        first_level_map = map_19_18[code]
        for subc in first_level_map:
            mapped_codes.append(subc)
    return list(set(mapped_codes))

def get_mapping_for_code(code, year): 
    if year >= 2003 and year <= 2005: 
        code_list_2006 = get_00_06_mapping([code])
        code_list_2009 = get_06_09_mapping(code_list_2006)
        code_list_2010 = get_09_10_mapping(code_list_2009)
        code_list_2019 = get_10_19_mapping(code_list_2010)
        code_list_2018 = get_19_18_mapping(code_list_2019)
        return code_list_2018
    elif year >= 2006 and year <= 2008:
        code_list_2009 = get_06_09_mapping([code])
        if not code_list_2009:
            print(f"Code {code} not found in 2006-2009 mapping.")
        code_list_2010 = get_09_10_mapping(code_list_2009)
        if not code_list_2010:
            print(f"Code {code} not found in 2009-2010 mapping.")
        code_list_2019 = get_10_19_mapping(code_list_2010)
        if not code_list_2019:
            print(f"Code {code} not found in 2010-2019 mapping.")
        code_list_2018 = get_19_18_mapping(code_list_2019)
        if not code_list_2018:
            print(f"Code {code} not found in 2019-2018 mapping.")
        return code_list_2018
    elif year >= 2009 and year <= 2010:
        code_list_2010 = get_09_10_mapping([code])
        code_list_2019 = get_10_19_mapping(code_list_2010)
        code_list_2018 = get_19_18_mapping(code_list_2019)
        return code_list_2018
    elif year >= 2011 and year <= 2019:
        code_list_2019 = get_10_19_mapping([code])
        code_list_2018 = get_19_18_mapping(code_list_2019)
        return code_list_2018
    elif year >= 2020 and year <= 2025: 
        code_list_2018 = get_19_18_mapping([code])
        return code_list_2018
    else: 
        return None

# --- Main processing ---
def main():
    # Load in all ONET task_statements
    """Main function to orchestrate the data loading, processing, and saving."""
    files = {year: f"task_statements_{year}_{suffix}.csv" for year, suffix in [
        (2003,'nov'),(2004,'dec'),(2005,'dec'),(2006,'dec'),(2007,'jun'),
        (2008,'jun'),(2009,'jun'),(2010,'jul'),(2011,'jul'),(2012,'jul'),
        (2013,'jul'),(2014,'jul'),(2015,'oct'),(2016,'nov'),(2017,'oct'),
        (2018,'nov'),(2019,'nov'),(2020,'nov'),(2021,'nov'),(2022,'nov'),
        (2023,'nov'),(2024,'nov'),(2025,'feb')]}

    all_dataframes = []
    print("\n--- Starting Data Processing ---")
    for year, filename in files.items():
        filepath = os.path.join(TASK_STATEMENTS_DIR, filename)
        print(f"Processing {year}: {filename}...")

        try:
            df = pd.read_csv(filepath, encoding='latin1', dtype=str)
        except FileNotFoundError:
            print(f"  -> Warning: File not found. Skipping.")
            continue

        if "O*NET-SOC Code" not in df.columns:
            print(f"  -> Warning: 'O*NET-SOC Code' column not found in {filename}.")
            continue

        df['ONET_release_year'] = year
        df['O*NET-SOC Code'] = df['O*NET-SOC Code'].str.strip()

        all_dataframes.append(df)

    print(f"Total dataframes loaded: {len(all_dataframes)}")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")

    # --- Combine, Clean, and Save ---
    if not all_dataframes:
        print("\nNo data was processed. Exiting.")
        return

    print("\n--- Combining and Finalizing Data ---")
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)

    columns_of_interest = [
        "O*NET-SOC Code", "O*NET 2010 SOC Code", "O*NET 2018 SOC Code",
        "ONET_release_year", "Task ID", "Task", "Task Type",
        "Incumbents Responding", "Date", "Domain Source"
    ]
    final_columns = [col for col in columns_of_interest if col in combined_df.columns]
    combined_df_relevant_cols = combined_df[final_columns].copy()

    dict_of_map_dicts = {}
    for year in range(2003, 2026):
        print(f"Processing year: {year}")
        year_df = combined_df_relevant_cols[combined_df_relevant_cols['ONET_release_year'] == year].copy()
        unique_codes_year = year_df['O*NET-SOC Code'].unique()
        print(f"  Unique O*NET-SOC Codes for {year}: {len(unique_codes_year)}")

        if year not in dict_of_map_dicts:
            dict_of_map_dicts[year] = {}
        
        for code in unique_codes_year:
            mapped_codes = get_mapping_for_code(code, year)
            if mapped_codes is not None:
                dict_of_map_dicts[year][code] = mapped_codes
        print(f"  Mapped codes for {year}: {len(dict_of_map_dicts[year])}")


    dict_of_map_dfs = {}
    for year, map_dict in dict_of_map_dicts.items():
        if map_dict:
            map_df = pd.DataFrame(map_dict.items(), columns=['Code', '2018 SOC Codes']).explode('2018 SOC Codes')
            dict_of_map_dfs[year] = map_df
            print(f"  Created mapping DataFrame for {year} with {len(map_df)} entries. There are {len(map_df['2018 SOC Codes'].unique())} unique 2018 SOC Codes.")

    def safe_get_2018_soc_codes(row):
        year = row['ONET_release_year']
        code = row['O*NET-SOC Code']
        return dict_of_map_dicts.get(year, {}).get(code, np.nan)
    
    combined_df_relevant_cols['2018 SOC Codes'] = combined_df_relevant_cols.apply(safe_get_2018_soc_codes, axis=1)
    all_onet_data_harmonized = combined_df_relevant_cols.explode('2018 SOC Codes').copy()

 
    # organize 
    all_onet_data_harmonized = all_onet_data_harmonized[['ONET_release_year', 'O*NET-SOC Code', '2018 SOC Codes', 'Task ID', 'Task', 'Task Type', 'Incumbents Responding', 'Date', 'Domain Source']]
    all_onet_data_harmonized.rename(columns={'2018 SOC Codes': '2018 SOC Code'}, inplace=True)

    # --- Analyze and Export ---
    print("Analyzing unique task statements...")
    task_counts = all_onet_data_harmonized['Task'].value_counts().reset_index()
    task_counts.columns = ['Task', 'Count']
    print(f"Found {len(task_counts)} unique task statements.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file_tasks = os.path.join(OUTPUT_DIR, "unique_task_statements.csv")
    task_counts.to_csv(output_file_tasks, index=False, encoding="utf-8")
    print(f"Saved unique task counts to: {output_file_tasks}")

    output_file_tasks_only = os.path.join(OUTPUT_DIR, "task_statements_and_ids.csv")
    tasks_only = all_onet_data_harmonized[['Task ID', 'Task']].drop_duplicates().dropna()
    tasks_only.to_csv(output_file_tasks_only, index=False, encoding="utf-8")
    print(f"Saved task statements to: {output_file_tasks_only}")

    output_file_all_data = os.path.join(OUTPUT_DIR, "all_onet_data_mapped_soc_codes.csv")
    all_onet_data_harmonized.to_csv(output_file_all_data, index=False, encoding="utf-8")
    print(f"Saved final harmonized data to: {output_file_all_data}")
    print(f"Final dataset contains {len(all_onet_data_harmonized)} rows.")
    print("\nProcessing complete.")

if __name__ == '__main__':
    main()

