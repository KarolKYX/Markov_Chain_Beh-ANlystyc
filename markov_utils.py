import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from typing import List, Dict, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. MARKOV MODEL CORE FUNCTIONS
# ==========================================

def train_markov_model(user_actions: List[str], order: int) -> Dict:
    """
    Trains a Markov Chain model of a given order based on a sequence of user actions.
    
    Args:
        user_actions (List[str]): A chronological sequence of user activities.
        order (int): The order of the Markov Chain.
        
    Returns:
        Dict: A nested dictionary representing the transition probability matrix.
    """
    transitions_count = defaultdict(lambda: defaultdict(int))
    
    # Count transitions
    for i in range(len(user_actions) - order):
        curr_state = tuple(user_actions[i : i + order])
        next_state = user_actions[i + order]
        transitions_count[curr_state][next_state] += 1
        
    # Calculate probabilities
    prob_matrix = defaultdict(dict)
    for state, next_states in transitions_count.items():
        total_transitions = sum(next_states.values())
        for next_state, count in next_states.items():
            prob_matrix[state][next_state] = count / total_transitions
            
    return prob_matrix


def score_sequence(sequence: List[str], prob_matrix: Dict, order: int, epsilon: float = 1e-5) -> float:
    """
    Calculates the average log-likelihood of a sequence given a trained Markov model.
    
    Args:
        sequence (List[str]): Sequence of activities to be scored.
        prob_matrix (Dict): Transition probability matrix.
        order (int): Order of the Markov Chain.
        epsilon (float): Small value to avoid log(0) for unseen transitions.
        
    Returns:
        float: Normalized log-likelihood score of the sequence. Returns 0.0 if sequence is too short.
    """
    if len(sequence) <= order:
        return 0.0 
        
    log_likelihood = 0.0
    for i in range(len(sequence) - order):
        curr_state = tuple(sequence[i : i + order])
        next_state = sequence[i + order]
        
        prob = prob_matrix.get(curr_state, {}).get(next_state, epsilon)
        log_likelihood += np.log(prob)
        
    # Normalize by the number of transitions evaluated
    return log_likelihood / (len(sequence) - order)

# ==========================================
# 2. ANALYSIS & THRESHOLD CALCULATION
# ==========================================

def calculate_baseline(df_user: pd.DataFrame, train_days: List, model: Dict, order: int) -> Tuple[float, float, float]:
    """
    Calculates the baseline log-likelihood and sets the anomaly threshold for a user.
    Uses the 3-sigma rule (Mean - 3 * Standard Deviation).
    """
    train_scores = []
    
    for day in train_days:
        daily_actions = df_user[df_user['date_only'] == day]['activity'].tolist()
        if len(daily_actions) > order:
            ll = score_sequence(daily_actions, model, order=order)
            train_scores.append(ll)
            
    mean_ll = np.mean(train_scores) if train_scores else 0.0
    std_dev = np.std(train_scores) if train_scores else 0.0
    
    if std_dev == 0:
        std_dev = 0.01 # Prevent zero standard deviation
        
    alarm_threshold = mean_ll - (3 * std_dev)
    
    return mean_ll, std_dev, alarm_threshold


def load_ground_truth(file_path: str) -> Dict[str, List]:
    """
    Loads actual attack dates (ground truth) from a CSV file.
    
    Returns:
        Dict: Mapping of user IDs to a list of attack dates.
    """
    df_gt = pd.read_csv(file_path)
    df_gt['date'] = pd.to_datetime(df_gt['date']).dt.date
    
    actual_attacks = {}
    for user in df_gt['user'].unique():
        days = df_gt[df_gt['user'] == user]['date'].tolist()
        actual_attacks[user] = days
        
    return actual_attacks

# ==========================================
# 3. VISUALIZATION MODULE
# ==========================================

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], order: int, save_path: str = None):
    """Generates and saves a styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=['Predicted: Normal', 'Predicted: Attack'],
                     yticklabels=['Actual: Normal', 'Actual: Attack'],
                     annot_kws={"size": 14}) 
    
    plt.title(f'Confusion Matrix: Markov Chain Order {order}\n', fontsize=15, fontweight='bold')
    
    # Add textual labels for quadrants
    ax.text(0.5, 0.2, 'True Negatives', ha='center', va='center', color='green', fontsize=12, fontweight='bold')
    ax.text(1.5, 0.2, 'False Positives', ha='center', va='center', color='red', fontsize=12, fontweight='bold')
    ax.text(0.5, 1.2, 'False Negatives', ha='center', va='center', color='red', fontsize=12, fontweight='bold')
    ax.text(1.5, 1.2, 'True Positives', ha='center', va='center', color='green', fontsize=12, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved as '{save_path}'.")
    plt.show()


def plot_anomaly_timeline(user_id: str, test_dates: List, test_scores: List[float], 
                          threshold: float, attack_dates: List, save_path: str = None):
    """Generates a professional timeline showing log-likelihood and detected anomalies."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Mark actual attack days in the background
    legend_added = False
    for attack_date in attack_dates:
        label = 'Actual Attack' if not legend_added else ""
        ax.axvline(x=attack_date, color='darkorange', linewidth=10, alpha=0.25, label=label)
        legend_added = True

    # Main Log-Likelihood line
    ax.plot(test_dates, test_scores, marker='o', linestyle='-', color='#3498DB', 
            linewidth=2, markersize=5, label='Behavior Score (Log-Likelihood)')

    # Anomaly Threshold line
    ax.axhline(y=threshold, color='#E74C3C', linestyle='--', linewidth=2, label='Anomaly Threshold')

    # Mark specific alert points
    for i, score in enumerate(test_scores):
        if score < threshold:
            ax.scatter(test_dates[i], score, color='red', s=80, zorder=5, edgecolors='black')

    # Styling
    ax.set_title(f'Behavioral Analysis Timeline: User {user_id}', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Log-Likelihood Score', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    plt.xticks(rotation=45)

    ax.grid(True, linestyle='-', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='lower left', fontsize=10, frameon=True, edgecolor='#cccccc')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Timeline plot saved as '{save_path}'.")
    plt.show()


def plot_metrics_summary(orders: List[str], precision: List[float], recall: List[float], 
                         f1_scores: List[float], save_path: str = None):
    """Generates a clean bar chart summarizing model performance across different orders."""
    x = np.arange(len(orders))
    width = 0.25  

    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - width, precision, width, label='Precision', color='#85C1E9') 
    rects2 = ax.bar(x, recall, width, label='Recall', color='#3498DB')       
    rects3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#154360') 

    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Detection Performance vs Markov Chain Order', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(orders, fontsize=12)
    ax.set_ylim(0, max(max(precision), max(recall), max(f1_scores)) + 0.15)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12, frameon=False)
    ax.grid(axis='y', linestyle='-', alpha=0.15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary chart saved as '{save_path}'.")
    plt.show()