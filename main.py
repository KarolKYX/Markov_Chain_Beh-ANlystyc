import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Import custom modules
import data_builder as db
import markov_utils as mu

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    # Relative paths - users should place the CERT dataset in the 'data' folder
    'raw_r42_dir': "./data/r4.2",
    'insiders_file': "./data/answers/insiders.csv",
    'answers_dirs': [
        "./data/answers/r4.2-1",
        "./data/answers/r4.2-2",
        "./data/answers/r4.2-3"
    ],
    
    # Output files
    'sample_dataset': "sample_dataset.csv",
    'ground_truth': "ground_truth.csv",
    
    # Model parameters
    'orders_to_test': [1, 2, 3, 4],
    'best_order': 2 
}

# ==========================================
# HELPER: EVALUATE SINGLE USER
# ==========================================
def evaluate_single_user(user_id: str, df_user: pd.DataFrame, order: int) -> dict:
    """Trains and tests a Markov model for a single user, returning daily anomaly predictions."""
    df_user = df_user.sort_values('date').copy()
    df_user['date_only'] = df_user['date'].dt.date
    
    all_days = sorted(df_user['date_only'].unique())
    if len(all_days) < 10:
        return {} # Not enough data to establish a baseline
        
    split_point = int(len(all_days) * 0.7)
    train_days = all_days[:split_point]
    test_days = all_days[split_point:]
    
    # Training phase
    train_actions = df_user[df_user['date_only'].isin(train_days)]['activity'].tolist()
    model = mu.train_markov_model(train_actions, order=order)
    _, _, alarm_threshold = mu.calculate_baseline(df_user, train_days, model, order)
    
    # Testing phase
    predictions = {}
    for day in test_days:
        daily_actions = df_user[df_user['date_only'] == day]['activity'].tolist()
        if len(daily_actions) > order:
            ll_score = mu.score_sequence(daily_actions, model, order=order)
            # Flag anomaly if the score drops below the threshold
            predictions[day] = 1 if ll_score < alarm_threshold else 0
            
    return predictions

# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    print("\n" + "="*50)
    print("🚀 BEHAVIORAL THREAT ANALYSIS PIPELINE STARTED")
    print("="*50)

    # ------------------------------------------
    # STEP 1: DATA PREPARATION (Auto-run if missing)
    # ------------------------------------------
    if not os.path.exists(CONFIG['sample_dataset']) or not os.path.exists(CONFIG['ground_truth']):
        print("\n[STEP 1] Generating required datasets...")
        target_users = db.sample_users(
            logon_file=os.path.join(CONFIG['raw_r42_dir'], 'logon.csv'),
            insiders_file=CONFIG['insiders_file'],
            n_normal=80, n_hackers=20
        )
        db.extract_and_merge_logs(CONFIG['raw_r42_dir'], target_users, CONFIG['sample_dataset'])
        db.build_ground_truth(CONFIG['answers_dirs'], CONFIG['ground_truth'])
    else:
        print("\n[STEP 1] Datasets found. Skipping data generation phase.")

    # ------------------------------------------
    # STEP 2: LOAD DATA
    # ------------------------------------------
    print("\n[STEP 2] Loading data into memory...")
    df_all = pd.read_csv(CONFIG['sample_dataset'])
    df_all['date'] = pd.to_datetime(df_all['date'])
    
    actual_attacks = mu.load_ground_truth(CONFIG['ground_truth'])
    analyzed_users = df_all['user'].unique()
    
    # Secure one hacker ID for attack visualization (Timeline plot)
    hacker_for_timeline = None
    for user in actual_attacks.keys():
        if user in analyzed_users:
            hacker_for_timeline = user
            break

    # ------------------------------------------
    # STEP 3: MODEL TRAINING & EVALUATION
    # ------------------------------------------
    print("\n[STEP 3] Evaluating Markov Chain Models (Orders 1 to 4)...")
    metrics_history = {'precision': [], 'recall': [], 'f1': []}

    for order in CONFIG['orders_to_test']:
        print(f"\nEvaluating Order {order}...")
        y_true, y_pred = [], []
        
        for user_id in analyzed_users:
            df_user = df_all[df_all['user'] == user_id]
            user_predictions = evaluate_single_user(user_id, df_user, order)
            
            user_attack_days = actual_attacks.get(user_id, [])
            
            for day, is_alert in user_predictions.items():
                y_pred.append(is_alert)
                y_true.append(1 if day in user_attack_days else 0)

        # System-wide metrics for the current Markov order
        if 1 in y_true:
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            metrics_history['precision'].append(p)
            metrics_history['recall'].append(r)
            metrics_history['f1'].append(f1)
            
            print(f"-> Precision: {p:.2f} | Recall: {r:.2f} | F1-Score: {f1:.2f}")

        # Generate detailed plots if this is our designated "best" model
        if order == CONFIG['best_order']:
            print(f"Generating detailed plots for Best Order ({order})...")
            mu.plot_confusion_matrix(y_true, y_pred, order, save_path=f"confusion_matrix_order_{order}.png")
            
            # Generate timeline for the selected hacker
            if hacker_for_timeline:
                df_hacker = df_all[df_all['user'] == hacker_for_timeline].copy()
                df_hacker['date_only'] = df_hacker['date'].dt.date
                
                # Quick re-evaluation to extract exact Log-Likelihood scores for the plot
                split = int(len(df_hacker['date_only'].unique()) * 0.7)
                train_d = sorted(df_hacker['date_only'].unique())[:split]
                test_d = sorted(df_hacker['date_only'].unique())[split:]
                
                h_model = mu.train_markov_model(df_hacker[df_hacker['date_only'].isin(train_d)]['activity'].tolist(), order)
                _, _, threshold = mu.calculate_baseline(df_hacker, train_d, h_model, order)
                
                test_scores = []
                test_dates_valid = []
                for d in test_d:
                    acts = df_hacker[df_hacker['date_only'] == d]['activity'].tolist()
                    if len(acts) > order:
                        test_scores.append(mu.score_sequence(acts, h_model, order))
                        test_dates_valid.append(d)
                
                mu.plot_anomaly_timeline(
                    user_id=hacker_for_timeline,
                    test_dates=test_dates_valid,
                    test_scores=test_scores,
                    threshold=threshold,
                    attack_dates=actual_attacks[hacker_for_timeline],
                    save_path=f"timeline_{hacker_for_timeline}.png"
                )

    # ------------------------------------------
    # STEP 4: FINAL SUMMARY CHART
    # ------------------------------------------
    if len(metrics_history['f1']) == len(CONFIG['orders_to_test']):
        print("\n[STEP 4] Generating final summary chart...")
        labels = [f"Order {o}" for o in CONFIG['orders_to_test']]
        mu.plot_metrics_summary(
            orders=labels, 
            precision=metrics_history['precision'], 
            recall=metrics_history['recall'], 
            f1_scores=metrics_history['f1'],
            save_path="metrics_summary.png"
        )
        
    print("\n✅ Analysis Complete! All plots saved successfully.")

if __name__ == "__main__":
    main()