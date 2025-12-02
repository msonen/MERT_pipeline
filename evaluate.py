import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import UnifiedMertDataset
from model import DownstreamHead
import config
from train_unified import load_labels
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import argparse

def evaluate_model(task_name):
    print(f"--- Evaluating {task_name.upper()} ---")
    conf = config.TASKS[task_name]
    
    # 1. Load Data & Labels
    labels_map = load_labels(task_name)
    dataset = UnifiedMertDataset(conf['folder'], labels_map, conf['type'], conf['classes'])
    
    # Recreate the split (Must use same seed/logic as train.py to be strictly accurate, 
    # but for a quick check, this approximates the hold-out set)
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    _, test_set = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))
    
    loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    head_path = f"./features/trained_{task_name}.pth"
    model = DownstreamHead(conf['classes'])
    model.load_state_dict(torch.load(head_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running inference on Test Set...")
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            
            # Get Predictions
            if conf['type'] == 'multiclass':
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
            else:
                print("Evaluation script currently optimized for Multiclass only.")
                return

    # 3. Report
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(all_labels, all_preds))
    
    # 4. Confusion Matrix (Optional: Prints simple text version)
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix (Diagonal = Correct):")
    print(cm)
    
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nTotal Test Set Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    args = parser.parse_args()
    
    evaluate_model(args.task)