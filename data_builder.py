import pandas as pd
import os
import glob
import csv
import random
from typing import List, Set, Dict
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
# To be set in main.py, but defaults are provided here
LOG_FILES = ['logon.csv', 'file.csv', 'http.csv', 'device.csv', 'email.csv']


# ==========================================
# 1. USER SAMPLING
# ==========================================

def sample_users(logon_file: str, insiders_file: str, n_normal: int = 80, n_hackers: int = 20) -> List[str]:
    """
    Identifies all active users in the dataset and samples a specific number
    of normal employees and malicious insiders (hackers).
    
    Args:
        logon_file (str): Path to a representative log file (e.g., logon.csv) to get active users.
        insiders_file (str): Path to the CSV containing known malicious insiders.
        n_normal (int): Number of benign users to sample.
        n_hackers (int): Number of malicious users to sample.
        
    Returns:
        List[str]: A combined, shuffled list of selected user IDs.
    """
    print("Initializing user sampling...")
    
    df_all = pd.read_csv(logon_file, usecols=['user'])
    df_insiders = pd.read_csv(insiders_file, usecols=['user'])

    all_users = set(df_all['user'].unique())
    all_insiders = set(df_insiders['user'].unique())

    # 1. Normal Employees (Active minus Insiders)
    normal_users = list(all_users - all_insiders)
    
    # 2. Active Insiders (Intersection of active users and known insiders)
    active_insiders = list(all_users & all_insiders)

    print(f"Dataset stats -> Total Active Users: {len(all_users)}")
    print(f"Dataset stats -> Active Insiders: {len(active_insiders)}")
    print(f"Dataset stats -> Benign Employees: {len(normal_users)}")

    # Sample safely
    sampled_hackers = random.sample(active_insiders, min(n_hackers, len(active_insiders)))
    sampled_normal = random.sample(normal_users, min(n_normal, len(normal_users)))

    test_group = sampled_hackers + sampled_normal
    
    # Validate uniqueness
    assert len(test_group) == len(set(test_group)), "Error: Duplicates found in sampled group."
    
    print(f"Successfully sampled {len(sampled_normal)} normal users and {len(sampled_hackers)} insiders.")
    return test_group


# ==========================================
# 2. LOG EXTRACTION
# ==========================================

def extract_and_merge_logs(data_dir: str, target_users: List[str], output_file: str):
    """
    Iterates over massive log files in chunks, extracts records for target users,
    standardizes the 'activity' column, and saves the final sample to a CSV.
    
    Args:
        data_dir (str): Directory containing the CERT dataset (r4.2).
        target_users (List[str]): List of users to extract logs for.
        output_file (str): Path to save the merged dataset.
    """
    all_actions = []
    
    for file_name in LOG_FILES:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_name} not found in {data_dir}. Skipping.")
            continue
            
        print(f"Processing {file_name} in chunks...")
        log_type = file_name.replace('.csv', '')
        
        # Process in chunks to prevent MemoryError (e.g., http.csv is massive)
        chunk_iterator = pd.read_csv(file_path, chunksize=500000, low_memory=False)
        
        for chunk in chunk_iterator:
            if 'user' in chunk.columns:
                target_chunk = chunk[chunk['user'].isin(target_users)].copy()
                
                if not target_chunk.empty:
                    target_chunk['log_type'] = log_type
                    all_actions.append(target_chunk)

    if not all_actions:
        print("Error: No logs found for the selected users.")
        return

    print("Merging and sorting extracted logs...")
    df_final = pd.concat(all_actions, ignore_index=True)
    df_final['date'] = pd.to_datetime(df_final['date'])
    df_final = df_final.sort_values(by=['user', 'date'])

    # Standardize 'activity' column. If missing, use log_type as a fallback.
    if 'activity' not in df_final.columns:
        df_final['activity'] = 'unknown'
    
    df_final['activity'] = df_final['activity'].fillna(df_final['log_type'])

    # Save lightweight version
    df_final[['date', 'user', 'activity']].to_csv(output_file, index=False)
    print(f"Successfully saved merged dataset to: {output_file}")


# ==========================================
# 3. GROUND TRUTH GENERATION
# ==========================================

def build_ground_truth(answer_dirs: List[str], output_file: str = "ground_truth.csv") -> Dict[str, List]:
    """
    Parses answer key folders provided by CERT to extract the exact dates 
    of malicious actions, generating a Ground Truth dataset.
    
    Args:
        answer_dirs (List[str]): List of paths to folders containing malicious action logs.
        output_file (str): Path to save the ground truth CSV.
        
    Returns:
        Dict: Mapping of user IDs to sorted lists of attack dates.
    """
    actual_attacks = {}
    print(f"Building Ground Truth from {len(answer_dirs)} scenarios...")
    
    all_csv_files = []
    for directory in answer_dirs:
        if os.path.exists(directory):
            all_csv_files.extend(glob.glob(os.path.join(directory, '*.csv')))
        else:
            print(f"Warning: Answer directory not found: {directory}")

    for file in all_csv_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 4:
                        continue 
                    
                    date_str = row[2]
                    user = row[3]
                    
                    try:
                        day = pd.to_datetime(date_str).date()
                        if user not in actual_attacks:
                            actual_attacks[user] = set()
                        actual_attacks[user].add(day)
                    except Exception:
                        continue
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            
    # Convert sets to sorted lists
    for u in actual_attacks:
        actual_attacks[u] = sorted(list(actual_attacks[u]))
        
    # Save to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user', 'date'])
        for user, days in actual_attacks.items():
            for day in days:
                writer.writerow([user, day])
                
    print(f"Ground truth saved to {output_file} (Contains {len(actual_attacks)} attackers).")
    return actual_attacks