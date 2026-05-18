"""
This script harmonizes O*NET task rating data from different time periods (2003-2007 and 2008-onwards).
It performs the following steps:
1.  Loads task rating data from separate yearly files.
2.  Cleans the data by filtering out suppressed or irrelevant entries.
3.  Calculates normalized importance (IM) scores for tasks.
4.  Calculates expected frequency (FT) scores for tasks.
5.  Merges the data from the two time periods into a single, harmonized dataset.
6.  Saves the final dataset to a CSV file.
"""

import pandas as pd
import os
import numpy as np
import re

# --- Configuration ---
# Define base directories for better path management.
# Update these paths if your directory structure is different.
BASE_DATA_DIR = '/Users/sidsatya/dev/ailabor/data/onet'
HISTORICAL_RATINGS_DIR = os.path.join(BASE_DATA_DIR, 'historical_onet_task_ratings')
HISTORICAL_STATEMENTS_DIR = os.path.join(BASE_DATA_DIR, 'historical_onet_task_statements')
OUTPUT_DIR = '/Users/sidsatya/dev/ailabor/onet_transformations/output_data'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'task_ratings_harmonized.csv')

TASK_STATEMENT_INTERMEDIATE_DIR = '/Users/sidsatya/dev/ailabor/onet_transformations/task_statement_harmonization/intermediate_data'
canon_df = pd.read_csv(os.path.join(TASK_STATEMENT_INTERMEDIATE_DIR, 'task_statements_with_canon_id.csv'))
task_statements_and_ids_df = pd.read_csv(os.path.join(TASK_STATEMENT_INTERMEDIATE_DIR, 'task_statements_and_ids.csv'))

# --- Helper Functions ---

def clean_text(text: str) -> str:
    # 2) remove any occurrences of "x92"
    text = re.sub(r'\x92', "'", text)
    # 1) replace any punctuation (i.e. non-word, non-space) with a space
    text = re.sub(r'[^\w\s]', '', text)
    # 3) lowercase everything
    text = text.lower()
    # 4) collapse multiple whitespace into one, and strip ends
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_file_dict(directory, file_info):
    """
    Creates a dictionary mapping years to their corresponding file paths.

    Args:
        directory (str): The directory where the files are located.
        file_info (dict): A dictionary where keys are years and values are filenames.

    Returns:
        dict: A dictionary of {year: full_file_path}.
    """
    return {year: os.path.join(directory, filename) for year, filename in file_info.items()}

def read_onet_task_data(onet_task_files):
    """
    Reads and concatenates multiple O*NET task files into a single DataFrame.

    Args:
        onet_task_files (dict): A dictionary of {year: full_file_path}.

    Returns:
        pd.DataFrame: A single DataFrame containing data from all files,
                      with an added 'year' column. Returns an empty DataFrame
                      if no files are found.
    """
    all_files = []
    for year, file_path in onet_task_files.items():
        try:
            df = pd.read_csv(file_path, encoding='latin1')
            df['year'] = year
            all_files.append(df)
        except FileNotFoundError:
            print(f"Warning: File not found for year {year} at {file_path}")
    
    if not all_files:
        return pd.DataFrame()
    return pd.concat(all_files, ignore_index=True)

def add_canon_id(df):
    """
    Adds a canonical ID to the DataFrame for easier merging and identification.
    - Skips rows where both 'Task' and 'Task ID' are null.
    - If 'Task' column is not null, it uses it to look up the canon_id.
    - If 'Task' is null but 'Task ID' is not, it uses 'Task ID' to first look up the 'Task' statement, then the canon_id.
    """
    # Ensure 'Task' and 'Task ID' columns exist, if not, create them with NaNs
    if 'Task' not in df.columns:
        df['Task'] = np.nan
    if 'Task ID' not in df.columns:
        df['Task ID'] = np.nan

    # Skip rows where both 'Task' and 'Task ID' are null.
    original_len = len(df)
    df.dropna(subset=['Task', 'Task ID'], how='all', inplace=True)
    if original_len > len(df):
        print(f"Dropped {original_len - len(df)} rows with null Task and Task ID.")

    # Initialize with proper data types to avoid dtype warnings
    df['canon_id'] = pd.Series(dtype='object')
    df['task_clean'] = pd.Series(dtype='object')

    task_to_canon_map = canon_df.drop_duplicates(subset=['task_clean']).set_index('task_clean')['canon_id']

    # Use 'Task' column if available
    mask_with_task = df['Task'].notna()
    if mask_with_task.any():
        df.loc[mask_with_task, 'task_clean'] = df.loc[mask_with_task, 'Task'].str.lower().apply(clean_text)
        df.loc[mask_with_task, 'canon_id'] = df.loc[mask_with_task, 'task_clean'].map(task_to_canon_map)

    # Use 'Task ID' if 'Task' is not available
    mask_without_task = df['Task'].isna() & df['Task ID'].notna()
    if mask_without_task.any():
        id_to_task_map = task_statements_and_ids_df.drop_duplicates(subset=['Task ID']).set_index('Task ID')['Task']
        task_statements = df.loc[mask_without_task, 'Task ID'].map(id_to_task_map)
        df.loc[mask_without_task, 'task_clean'] = task_statements.apply(clean_text)
        df.loc[mask_without_task, 'canon_id'] = df.loc[mask_without_task, 'task_clean'].map(task_to_canon_map)

    print("Total number of matched rows with canon df are: ", len(df[df['canon_id'].notna()]))
    # write tasks that do not have a canon_id to a file for further investigation
    missing_canon_id = df[df['canon_id'].isna()]
    if not missing_canon_id.empty:
        # make sure to append to file if already exists
        output_path = os.path.join(OUTPUT_DIR, 'missing_canon_ids.csv')
        # Only write headers if the file doesn't exist
        write_header = not os.path.exists(output_path)
        missing_canon_id.to_csv(output_path, mode='a', index=False, header=write_header)
        print(f"Warning: {len(missing_canon_id)} tasks do not have a canon_id. Check 'missing_canon_ids.csv' for details.")
    return df

# --- Data Processing for 2008 onwards ---

def process_2008_onwards_data(task_files):
    """
    Processes O*NET task rating data from 2008 onwards. This involves
    calculating normalized importance and expected frequency for each task.

    Args:
        task_files (dict): A dictionary of file paths for the 2008+ data.

    Returns:
        pd.DataFrame: A processed DataFrame with harmonized task ratings.
    """
    # Step 1: Load and perform initial filtering on the data.
    # We keep only records that are recommended for use and are either
    # Importance (IM) or Frequency (FT) scales.
    df = read_onet_task_data(task_files)

    df = add_canon_id(df)

    if df.empty:
        return pd.DataFrame()
 
    print(f"Initial rows for 2008 onwards: {len(df)}")
    df_filtered = df[
        (df['Recommend Suppress'] != 'Y') &
        (df['Scale ID'].isin(['IM', 'FT']))
    ].copy()
    print(f"Rows after filtering for 'Recommend Suppress' and 'Scale ID': {len(df_filtered)}")


    # Step 2: Process Importance (IM) data.
    # We calculate a normalized importance score for each task within its
    # occupation and year.
    df_im = df_filtered[df_filtered['Scale ID'] == 'IM'].copy()
    im_sum = df_im.groupby(['O*NET-SOC Code', 'year'])['Data Value'].transform('sum')
    df_im['IM_normalized'] = df_im['Data Value'] / im_sum

    # Step 3: Process Frequency (FT) data.
    # We calculate the expected frequency by creating a weighted sum of frequency ratings.
    df_ft = df_filtered[df_filtered['Scale ID'] == 'FT'].copy()
    df_ft['weighted_frequency'] = df_ft['Category'] * df_ft['Data Value']
    df_ft_agg = df_ft.groupby(['O*NET-SOC Code', 'canon_id', 'year', 'Scale ID'])['weighted_frequency'].sum().reset_index()
    df_ft_agg['expected_freq'] = df_ft_agg['weighted_frequency'] / 100

    # Step 4: Merge the calculated IM and FT metrics.
    # We merge the separately processed IM and FT dataframes to combine the metrics.
    im_data = df_im[['O*NET-SOC Code', 'canon_id', 'year', 'Date', 'Data Value', 'IM_normalized']]
    ft_data = df_ft_agg[['O*NET-SOC Code', 'canon_id', 'year', 'expected_freq']]

    merged_data = pd.merge(im_data, ft_data, on=['O*NET-SOC Code', 'canon_id', 'year'], how='outer')

    # Step 5: Rename columns for clarity and consistency.
    merged_data.rename(columns={
        'Data Value': 'Mean Importance',
        'IM_normalized': 'Importance Normalized All',
        'expected_freq': 'Mean Frequency'
    }, inplace=True)

    # Step 6: Select and order the final columns.
    final_cols = ['O*NET-SOC Code', 'canon_id', 'year', 'Date', 'Mean Importance', 'Importance Normalized All', 'Mean Frequency']
    return merged_data[final_cols]

# --- Data Processing for 2003-2007 ---

def process_2003_to_2007_data(task_files):
    """
    Processes O*NET task statement data from 2003 to 2007. This data has a
    different structure and requires separate processing logic.

    Args:
        task_files (dict): A dictionary of file paths for the 2003-2007 data.

    Returns:
        pd.DataFrame: A processed DataFrame with harmonized task ratings.
    """
    # Step 1: Load the data.
    df = read_onet_task_data(task_files)
    df = add_canon_id(df)
    if df.empty:
        return pd.DataFrame()

    print(f"Initial rows for 2003-2007: {len(df)}")

    # Step 2: Process Importance (IM) data.
    # Filter for valid importance data and calculate normalized scores.
    im_mask = (
        (df['Recommend Suppress'] != 'Y') &
        (df['Data Value'].notna()) &
        (df['Scale ID'] == 'IM')
    )
    df_im = df[im_mask].copy()
    im_sum = df_im.groupby(['O*NET-SOC Code', 'year'])['Data Value'].transform('sum')
    df_im['IM_normalized'] = df_im['Data Value'] / im_sum

    # Step 3: Process Frequency (FT) data.
    # Define frequency and suppression columns for clarity.
    freq_cols = [f'Percent Frequency: {cat}-F{i+1}' for i, cat in enumerate(['Yearly Or Less', 'More Than Yearly', 'More Than Monthly', 'More Than Weekly', 'Daily', 'Several Times Daily', 'Hourly Or More'])]
    suppress_cols = [f'Recommend Suppress-F{i+1}' for i in range(0, 7)]

    # Filter for valid frequency data and calculate expected frequency.
    ft_mask = (df[suppress_cols] != 'Y').all(axis=1) & (df[freq_cols].notna().all(axis=1))
    df_ft = df[ft_mask].copy()

    # Ensuring that any missing bucket shares are treated as 0
    df_ft[freq_cols] = df_ft[freq_cols].fillna(0)

    weights = np.arange(1, 8)
    df_ft['expected_frequency'] = (df_ft[freq_cols].values * weights).sum(axis=1) / 100

    # Step 4: Merge calculated metrics back into a base dataframe.
    # We use the original dataframe `df` as the base to merge our new metrics into.
    df_final = df.copy()
    df_final = pd.merge(
        df_final,
        df_im[['O*NET-SOC Code', 'canon_id', 'year', 'IM_normalized']],
        on=['O*NET-SOC Code', 'canon_id', 'year'],
        how='left'
    )
    df_final = pd.merge(
        df_final,
        df_ft[['O*NET-SOC Code', 'canon_id', 'year', 'expected_frequency']],
        on=['O*NET-SOC Code', 'canon_id', 'year'],
        how='left'
    )

    # Step 5: Filter for rows with importance data and rename columns.
    # This step aligns with the original script's logic to only keep tasks with an importance score.
    df_final = df_final[df_final['Scale ID'] == 'IM'].copy()
    df_final.rename(columns={
        'Data Value': 'Mean Importance',
        'IM_normalized': 'Importance Normalized All',
        'expected_frequency': 'Mean Frequency'
    }, inplace=True)

    # Step 6: Select and order the final columns.
    output_cols = ['O*NET-SOC Code', 'canon_id', 'year', 'Date', 'Mean Importance', 'Importance Normalized All', 'Mean Frequency']
    return df_final[output_cols]

# --- Main Execution ---

def main():
    """
    Main function to orchestrate the entire data harmonization process.
    """
    # --- Define File Lists ---
    # These dictionaries map years to their respective data files.
    files_2008_onwards_info = {
        2008: "task_ratings_2008_jun.csv", 2009: "task_ratings_2009_jun.csv",
        2010: "task_ratings_2010_jul.csv", 2011: "task_ratings_2011_jul.csv",
        2012: "task_ratings_2012_jul.csv", 2013: "task_ratings_2013_jul.csv",
        2014: "task_ratings_2014_jul.csv", 2015: "task_ratings_2015_oct.csv",
        2016: "task_ratings_2016_nov.csv", 2017: "task_ratings_2017_oct.csv",
        2018: "task_ratings_2018_nov.csv", 2019: "task_ratings_2019_nov.csv",
        2020: "task_ratings_2020_nov.csv", 2021: "task_ratings_2021_nov.csv",
        2022: "task_ratings_2022_nov.csv", 2023: "task_ratings_2023_nov.csv",
        2024: "task_ratings_2024_nov.csv", 2025: "task_ratings_2025_feb.csv"
    }
    task_files_2008_onwards = create_file_dict(HISTORICAL_RATINGS_DIR, files_2008_onwards_info)

    files_2003_to_2007_info = {
        2003: "task_statements_2003_nov.csv", 2004: "task_statements_2004_dec.csv",
        2005: "task_statements_2005_dec.csv", 2006: "task_statements_2006_dec.csv",
        2007: "task_statements_2007_jun.csv"
    }
    task_files_2003_to_2007 = create_file_dict(HISTORICAL_STATEMENTS_DIR, files_2003_to_2007_info)

    # --- Process Data from Both Periods ---
    print("Processing data from 2008 onwards...")
    data_2008_onwards = process_2008_onwards_data(task_files_2008_onwards)

    print("\nProcessing data from 2003 to 2007...")
    data_2003_to_2007 = process_2003_to_2007_data(task_files_2003_to_2007)

    # --- Combine and Save Final Dataset ---
    print("\nCombining datasets...")
    # Concatenate the two processed dataframes into a single master dataframe.
    final_data = pd.concat([data_2008_onwards, data_2003_to_2007], ignore_index=True)

    # If there are multiple rows with the same O*NET-SOC Code, canon_id, and year, aggregate them
    # by taking the mean of the importance and frequency scores. If either the importance or frequency
    # is NaN, it will be ignored in the mean calculation.
    final_data = final_data.groupby(['O*NET-SOC Code', 'canon_id', 'year']).agg({
        'Mean Importance': 'mean',
        'Importance Normalized All': 'mean',
        'Mean Frequency': 'mean',
        'Date': 'first'  # Keep the first date for each group
    }).reset_index()

    # --- Final Cleanup ---
    # Ensure that the final dataset has no NaN values in the key columns.
    final_data.dropna(subset=['O*NET-SOC Code', 'canon_id', 'year'], inplace=True)

    # Print summary of the final dataset.
    print(f"\nFinal dataset contains {len(final_data)} rows with {len(final_data.columns)} columns.")

    # --- Save the Final Dataset ---
    print(f"\nSaving the final harmonized data to {OUTPUT_FILE}...")

    # Ensure the output directory exists before saving the file.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the final harmonized data to a CSV file.
    final_data.to_csv(OUTPUT_FILE, index=False)
    print(f"\nProcessing complete. Final data saved to {OUTPUT_FILE}")
    print(f"Final dataset has {len(final_data)} rows.")

if __name__ == '__main__':
    main()