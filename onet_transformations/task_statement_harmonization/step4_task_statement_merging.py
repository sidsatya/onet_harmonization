"""
This script merges the harmonized task data with task ratings and computes several derived metrics.

The workflow is as follows:
1.  **Load Data**: Loads three key datasets:
    - All O*NET task data with SOC codes mapped from previous steps.
    - Task statements with their canonical (harmonized) IDs from the clustering step.
    - Task ratings data (e.g., importance, frequency).
2.  **Merge Data**: Joins these datasets to create a comprehensive file that links tasks, canonical IDs, and ratings.
3.  **Compute Normalized Scores**: Calculates normalized scores for task importance and frequency.
    This is done for all tasks and for "Core" tasks separately. Normalization is performed
    within occupation-year groups to ensure comparability.
4.  **Compute Task Intensity**: Calculates a "task intensity" score, defined as the product of
    importance and frequency, and normalizes it within occupation-year groups. This is also
    done for all tasks and for "Core" tasks separately.
5.  **Add Temporal Features**: For each harmonized task (canon_id), it determines the first and
    last year it was seen in the data, providing a lifespan for each task.
6.  **Save Full Dataset**: Saves the fully merged and processed dataset.
7.  **Create Healthcare Subset**: Filters the final dataset to include only healthcare-related
    occupations, based on a list of SOC codes from BLS OES data, and saves this subset
    and its unique tasks for further analysis.
"""
import pandas as pd
import numpy as np
import os
import re

# --- Configuration ---
# Centralized configuration for file paths and filenames.
CONFIG = {
    "root_path": "/Users/sidsatya/dev/ailabor/",
    "output_path": "/Users/sidsatya/dev/ailabor/onet_transformations/output_data/",
    "intermediate_data_path": "/Users/sidsatya/dev/ailabor/onet_transformations/task_statement_harmonization/intermediate_data/",
    "bls_output_path": "/Users/sidsatya/dev/ailabor/bls_transformations/output_data1/",
    "all_task_data_file": "all_onet_data_mapped_soc_codes.csv",
    "canon_id_file": "task_statements_with_canon_id.csv",
    "task_ratings_file": "task_ratings_harmonized.csv",
    "merged_output_file": "task_statements_harmonized_with_attributes.csv",
    "bls_healthcare_file": "oes_data_filtered_healthcare_soc_2018.csv",
    "healthcare_filtered_file": "task_data_healthcare_filtered.csv",
    "unique_healthcare_tasks_file": "unique_task_statements_healthcare.csv",
    "grouping_cols": ['ONET_release_year', '2018 SOC Code']
}

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

def load_data(config):
    """
    Loads the necessary CSV files into pandas DataFrames.

    Args:
        config (dict): The configuration dictionary with file paths.

    Returns:
        tuple: A tuple of three pandas DataFrames (all_task_data, canon_ids, task_ratings).
    """
    print("Step 1: Loading data...")
    output_path = config["output_path"]
    intermediate_path = config["intermediate_data_path"]
    
    all_task_data = pd.read_csv(os.path.join(intermediate_path, config["all_task_data_file"]))
    canon_ids = pd.read_csv(os.path.join(intermediate_path, config["canon_id_file"]))
    task_ratings = pd.read_csv(os.path.join(output_path, config["task_ratings_file"]))

    # Clean up task_ratings data
    task_ratings = task_ratings.drop(columns=['Date'])
    task_ratings = task_ratings.dropna(subset=['canon_id'])
    
    print("Data loaded successfully.")
    print(f"  - All task data shape: {all_task_data.shape}")
    print(f"  - Canonical IDs shape: {canon_ids.shape}")
    print(f"  - Task ratings shape: {task_ratings.shape}")
    return all_task_data, canon_ids, task_ratings

def merge_data(all_task_data, canon_ids, task_ratings):
    """
    Merges the loaded DataFrames into a single comprehensive DataFrame.

    Args:
        all_task_data (pd.DataFrame): DataFrame with all task data and mapped SOCs.
        canon_ids (pd.DataFrame): DataFrame with tasks and their canonical IDs.
        task_ratings (pd.DataFrame): DataFrame with task ratings.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    print("\nStep 2: Merging datasets...")
    # Merge canonical IDs into the main task data
    all_task_data['task_clean'] = all_task_data['Task'].apply(clean_text)
    print(f"Duplicates in canon id data? ", canon_ids.groupby('task_clean').size().reset_index(name='count').query('count > 1').shape[0])
    task_data_with_canons = pd.merge(all_task_data, canon_ids, on='task_clean', how='left')
    print(f"Shape after merging canonical IDs: {task_data_with_canons.shape}")
    print(f"  - {task_data_with_canons['canon_id'].notna().sum()} of {len(task_data_with_canons)} rows ({task_data_with_canons['canon_id'].notna().sum() / len(task_data_with_canons):.2%}) have a canonical ID.")

    # Merge task ratings
    merged_df = pd.merge(
        task_data_with_canons, 
        task_ratings, 
        left_on=['canon_id', 'ONET_release_year', 'O*NET-SOC Code'], 
        right_on=['canon_id', 'year', 'O*NET-SOC Code'], 
        how='left'
    )
    # Clean up columns from the merge
    #merged_df = merged_df.drop(columns=['year', 'O*NET-SOC Code_y']).rename(columns={'O*NET-SOC Code_x': 'O*NET-SOC Code'})
    print(f"Shape after merging task ratings: {merged_df.shape}")
    print(f"  - {merged_df['Mean Importance'].notna().sum()} of {len(merged_df)} rows ({merged_df['Mean Importance'].notna().sum() / len(merged_df):.2%}) have an Importance score.")
    print(f"  - {merged_df['Mean Frequency'].notna().sum()} of {len(merged_df)} rows ({merged_df['Mean Frequency'].notna().sum() / len(merged_df):.2%}) have a Frequency score.")

    # Drop Task_y and rename Task_x to Task
    merged_df = merged_df.drop(columns=['Task_y']).rename(columns={'Task_x': 'Task'})
    return merged_df

def add_normalized_score(df, base_col, new_col, group_cols, core_only=False):
    """
    Calculates a normalized score and adds it as a new column to the DataFrame.
    The score is normalized by the sum of the base column within specified groups.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        base_col (str): The name of the column to normalize (e.g., 'Mean Importance').
        new_col (str): The name of the new column for the normalized score.
        group_cols (list): The list of columns to group by for normalization.
        core_only (bool): If True, calculate the score only for 'Core' tasks.

    Returns:
        pd.DataFrame: The DataFrame with the new normalized column.
    """
    print(f"Calculating '{new_col}'...")
    # Define the mask for rows to be processed
    if core_only:
        mask = (df['Task Type'] == 'Core') & df[base_col].notna() & df['canon_id'].notna()
    else:
        mask = df[base_col].notna() & df['canon_id'].notna()

    print(f"  - Calculating for {mask.sum()} rows.")
    # Calculate the sum of the base column for the masked rows, grouped by specified columns.
    # .transform('sum') broadcasts the sum back to the original shape of the filtered data.
    sums = df.loc[mask].groupby(group_cols)[base_col].transform('sum')
    
    # Calculate the normalized score and assign it to the new column for the masked rows.
    df.loc[mask, new_col] = df.loc[mask, base_col] / sums
    return df

def add_task_intensity(df, new_col, group_cols, core_only=False):
    """
    Calculates a normalized task intensity score and adds it as a new column.
    Task intensity is defined as (Mean Importance * Mean Frequency), normalized by its sum within groups.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        new_col (str): The name for the new task intensity column.
        group_cols (list): The list of columns to group by for normalization.
        core_only (bool): If True, calculate the score only for 'Core' tasks.

    Returns:
        pd.DataFrame: The DataFrame with the new task intensity column.
    """
    print(f"Calculating '{new_col}'...")
    # Define the mask for rows to be processed
    if core_only:
        mask = (df['Task Type'] == 'Core') & df['Mean Importance'].notna() & df['Mean Frequency'].notna()
    else:
        mask = df['Mean Importance'].notna() & df['Mean Frequency'].notna()
    
    print(f"  - Calculating for {mask.sum()} rows.")
    # Calculate the raw task intensity score for the masked rows
    ti = df.loc[mask, 'Mean Importance'] * df.loc[mask, 'Mean Frequency']
    
    # To perform a grouped transformation, we need to add the 'ti' to a temporary DataFrame
    df_temp = df.loc[mask].copy()
    df_temp['ti'] = ti
    
    # Calculate the sum of 'ti' within groups and broadcast it
    ti_sums = df_temp.groupby(group_cols)['ti'].transform('sum')
    
    # Calculate the normalized intensity and assign it to the new column
    df.loc[mask, new_col] = ti / ti_sums
    return df

def add_temporal_features(df):
    """
    Adds columns indicating the first and last year a harmonized task was observed.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with 'first_seen' and 'last_seen' columns.
    """
    print("\nStep 4: Adding temporal features (first/last seen year)...")
    # Group by the harmonized canon ID and find the min and max release year
    grp = df.groupby(['O*NET-SOC Code', 'canon_id'])
    df['first_seen'] = grp['ONET_release_year'].transform('min')
    df['last_seen'] = grp['ONET_release_year'].transform('max')
    return df

def filter_and_save_healthcare_subset(df, config):
    """
    Filters the data for healthcare occupations and saves the results.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        config (dict): The configuration dictionary.
    """
    print("\nStep 6: Filtering for healthcare occupations...")
    # Read in BLS OES healthcare data
    bls_path = os.path.join(config["bls_output_path"], config["bls_healthcare_file"])
    bls_oes_healthcare = pd.read_csv(bls_path)
    oes_unique_healthcare_soc_codes = bls_oes_healthcare['soc_2018'].unique().tolist()
    print(f"Found {len(oes_unique_healthcare_soc_codes)} unique healthcare SOC codes from BLS data.")

    # Filter task data for healthcare occupations using the 2018 SOC code
    task_data_healthcare = df[df['2018 SOC Code'].isin(oes_unique_healthcare_soc_codes)].copy()
    print(f"Filtered down to {task_data_healthcare['2018 SOC Code'].nunique()} unique healthcare occupations in the task data.")
    print(f"This corresponds to {task_data_healthcare['Task'].nunique()} unique tasks.")

    # Save the filtered healthcare data
    healthcare_output_path = os.path.join(config["intermediate_data_path"], config["healthcare_filtered_file"])
    task_data_healthcare.to_csv(healthcare_output_path, index=False)
    print(f"Healthcare task data saved to '{healthcare_output_path}'.")

    # Save unique task statements from the healthcare subset
    unique_tasks_df = pd.DataFrame(task_data_healthcare['Task'].unique(), columns=['Task'])
    unique_tasks_output_path = os.path.join(config["intermediate_data_path"], config["unique_healthcare_tasks_file"])
    unique_tasks_df.to_csv(unique_tasks_output_path, index=False)
    print(f"Unique healthcare task statements saved to '{unique_tasks_output_path}'.")

def main():
    """
    Main function to orchestrate the data merging and feature engineering workflow.
    """
    # 1. Load data
    all_task_data, canon_ids, task_ratings = load_data(CONFIG)

    # 2. Merge data
    merged_data = merge_data(all_task_data, canon_ids, task_ratings)

    # 3. Calculate normalized scores and task intensity
    print("\nStep 3: Calculating normalized scores and task intensity...")
    group_cols = CONFIG["grouping_cols"]
    
    # Importance scores
    merged_data = add_normalized_score(merged_data, 'Mean Importance', 'normalized_importance', group_cols)
    merged_data = add_normalized_score(merged_data, 'Mean Importance', 'normalized_importance_core', group_cols, core_only=True)
    
    # Frequency scores
    merged_data = add_normalized_score(merged_data, 'Mean Frequency', 'normalized_frequency', group_cols)
    merged_data = add_normalized_score(merged_data, 'Mean Frequency', 'normalized_frequency_core', group_cols, core_only=True)
    
    # Task intensity scores
    merged_data = add_task_intensity(merged_data, 'task_intensity', group_cols)
    merged_data = add_task_intensity(merged_data, 'task_intensity_core', group_cols, core_only=True)

    # 4. Add temporal features
    final_data = add_temporal_features(merged_data)

    # 5. Save the fully processed data
    print("\nStep 5: Saving fully merged and processed data...")
    output_path = os.path.join(CONFIG["output_path"], CONFIG["merged_output_file"])
    final_data.to_csv(output_path, index=False)
    print(f"Final dataset saved to '{output_path}'.")

    # 6. Filter for healthcare subset and save
    filter_and_save_healthcare_subset(final_data, CONFIG)
    
    print("\nWorkflow completed successfully.")

if __name__ == "__main__":
    main()