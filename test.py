import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from dataset import new_CombDataset
from collator import MDMSD_collate_fn
from model_test import MDMSD
from model_config import config
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support
)
from scipy.stats import pearsonr, spearmanr

# ============================================================
# Config
# ============================================================
cfg = config['combine']
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- Manual Parameters ----------------
moa_csv = "datasets/moa/moa_test.csv"
syn_csv = "datasets/syn/my_oneil.csv"
cell_expr_csv = "datasets/ccle_expr_norm.csv"
emb_file1 = "datasets/ids/kpgt_base.npz"
emb_file2 = "datasets/ids/data_mol_repr_2000drug_conf1.npz"
ids_csv = "datasets/ids_smiles/ids_smiles.csv"
checkpoint = "checkpoints/model.pt"

selected_desc_labels = cfg.get("selected_desc_labels", None)
selected_metrics = cfg.get("selected_metrics", None)

# ============================================================
# Dataset & Loader
# ============================================================
test_dataset = new_CombDataset(
    moa_csv=moa_csv,
    syn_csv=syn_csv,
    cell_expr_csv=cell_expr_csv,
    emb_file1=emb_file1,
    emb_file2=emb_file2,
    ids_csv=ids_csv,
    selected_desc_labels=selected_desc_labels,
    selected_metrics=selected_metrics
)

test_loader = DataLoader(
    test_dataset,
    batch_size=cfg['batch_size'],
    shuffle=False,
    collate_fn=MDMSD_collate_fn,
    num_workers=cfg.get("num_workers", 4),
    pin_memory=True
)

# ============================================================
# Model
# ============================================================
model = MDMSD(
    k_embed=cfg['k_embed'],
    u_embed=cfg['u_embed'],
    num_desc_classes=cfg['num_desc_classes'],
    num_action_classes=cfg['num_action_classes'],
    cell_expr_dim=cfg['cell_expr_dim'],
    dimreduct_dim=cfg['dimreduct_dim'],
    expr_hidden=cfg['expr_hidden'],
    regress_hidden=cfg['regress_hidden'],
    lstm_hdims=cfg['lstm_hdims'],
    dropout_atten=cfg['dropout_atten'],
    dropout_dimreduct=cfg['dropout_dimreduct'],
    dropout_cellfcn=cfg['dropout_cellfcn'],
    dropout_downstream=cfg['dropout_downstream'],
    num_heads=cfg['num_heads'],
    mode=cfg['mode'],
    use_bn=cfg['use_bn']
)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)

# ============================================================
# Load Checkpoint (safe loading)
# ============================================================
print(f"Loading checkpoint: {checkpoint}")
state_dict = torch.load(checkpoint, map_location=device)
if any(k.startswith("module.") for k in state_dict.keys()):
    print("Detected multi-GPU checkpoint, removing 'module.' prefixes…")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

if hasattr(model, "module"):
    model.module.load_state_dict(state_dict, strict=False)
else:
    model.load_state_dict(state_dict, strict=False)
print("Model loaded successfully.\n")
model.eval()

# ============================================================
# Test Loop
# ============================================================
all_desc_preds, all_desc_labels = [], []
all_action_preds, all_action_labels = [], []
all_syn_preds, all_syn_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", ncols=None):
        batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        (desc_logits, action_logits), syn_pred = model(
            batch_device["k_moa1"], batch_device["u_moa1"],
            batch_device["k_moa2"], batch_device["u_moa2"],
            batch_device["k_syn1"], batch_device["u_syn1"],
            batch_device["k_syn2"], batch_device["u_syn2"],
            batch_device["cell_expr"]
        )

        all_desc_preds.append(torch.argmax(desc_logits, dim=-1).cpu().numpy())
        all_desc_labels.append(batch_device["desc_label"].cpu().numpy())
        all_action_preds.append(torch.argmax(action_logits, dim=-1).cpu().numpy())
        all_action_labels.append(batch_device["action_label"].cpu().numpy())
        all_syn_preds.append(syn_pred.squeeze(-1).cpu().numpy())
        all_syn_labels.append(batch_device["syn_score"][:, 0].cpu().numpy())

desc_labels_all = np.concatenate(all_desc_labels)
desc_preds_all = np.concatenate(all_desc_preds)
action_labels_all = np.concatenate(all_action_labels)
action_preds_all = np.concatenate(all_action_preds)
syn_labels_all = np.concatenate(all_syn_labels)
syn_preds_all = np.concatenate(all_syn_preds)

# ============================================================
# Classification Metrics
# ============================================================
def per_class_metrics(y_true, y_pred, task_name):
    """calculate Accuracy, Precision, Recall, F1 for each class"""
    unique_classes = np.unique(y_true)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_classes, zero_division=0
    )
    results = []
    for cls, p, r, f, s in zip(unique_classes, precision, recall, f1, support):
        mask = y_true == cls
        acc = accuracy_score(y_true[mask], y_pred[mask])
        results.append({
            "task": task_name,
            "class": int(cls),
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f,
            "support": s
        })
    return pd.DataFrame(results)

desc_df = per_class_metrics(desc_labels_all, desc_preds_all, "Description")
action_df = per_class_metrics(action_labels_all, action_preds_all, "Action")

desc_acc = accuracy_score(desc_labels_all, desc_preds_all)
action_acc = accuracy_score(action_labels_all, action_preds_all)

desc_precision, desc_recall, desc_f1, _ = precision_recall_fscore_support(
    desc_labels_all, desc_preds_all, average='macro', zero_division=0
)
action_precision, action_recall, action_f1, _ = precision_recall_fscore_support(
    action_labels_all, action_preds_all, average='macro', zero_division=0
)

# ============================================================
# Regression Metrics
# ============================================================
def compute_mae_mse_rmse(target, pred):
    err = np.asarray(target) - np.asarray(pred)
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    return mae, mse, rmse

def compute_r2(x, y):
    try:
        r = np.corrcoef(x, y)[0, 1]
        return r ** 2
    except Exception:
        return 0.0

mae, mse, rmse = compute_mae_mse_rmse(syn_labels_all, syn_preds_all)
r2 = compute_r2(syn_labels_all, syn_preds_all)
pearson_corr, _ = pearsonr(syn_labels_all, syn_preds_all)
spearman_corr, _ = spearmanr(syn_labels_all, syn_preds_all)

# ============================================================
# Print Summary
# ============================================================
print("=" * 70)
print(f"Description  Acc={desc_acc:.4f}  Prec={desc_precision:.4f}  Rec={desc_recall:.4f}  F1={desc_f1:.4f}")
print(f"Action       Acc={action_acc:.4f}  Prec={action_precision:.4f}  Rec={action_recall:.4f}  F1={action_f1:.4f}")
print("--------------------------------------------------------------")
print(f"Synergy MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, "
      f"Pearson={pearson_corr:.4f}, Spearman={spearman_corr:.4f}")
print("=" * 70)

print("\nPer-class Description Metrics:")
print(desc_df.to_string(index=False, justify="center"))
print("\nPer-class Action Metrics:")
print(action_df.to_string(index=False, justify="center"))
