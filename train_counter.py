# ========================= train.py =========================
import os
import numpy as np
import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from tqdm import tqdm

from model_config import config
from dataset import new_CombDataset
from collator import combined_model_collate_fn
from model import MDMSD

# ---------------- Argument Parser ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--moa_csv', type=str, required=True)
parser.add_argument('--syn_csv', type=str, required=True)
parser.add_argument('--cell_expr_csv', type=str, required=True)
parser.add_argument('--emb_file1', type=str, required=True)
parser.add_argument('--emb_file2', type=str, required=True)
parser.add_argument('--ids_file', type=str, required=True)
parser.add_argument('--save_dir', type=str, default='./checkpoints')
parser.add_argument('--gpus', type=str, default='0')
args = parser.parse_args()

# ---------------- Device ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- Config ----------------
cfg = config['combine']
batch_size = cfg['batch_size']
lr = cfg['lr']
weight_decay = cfg['weight_decay']
num_workers = cfg.get('num_workers', 4)
epochs = cfg['epochs']
val_ratio = 0.2  

# ---------------- Dataset ----------------
dataset = new_CombDataset(
    moa_csv=args.moa_csv,
    syn_csv=args.syn_csv,
    cell_expr_csv=args.cell_expr_csv,
    emb_file1=args.emb_file1,
    emb_file2=args.emb_file2,
    ids_csv=args.ids_file,
    selected_desc_labels=cfg.get("selected_desc_labels", None),
    selected_metrics=cfg.get("selected_metrics", None)
)

# ---------------- Compute class weights ----------------
labels = dataset.moa_data['desc_label'].to_numpy()
num_classes = len(dataset.desc_map)
total_samples = len(labels)
counts = np.bincount(labels, minlength=num_classes)
weights = total_samples / counts
weights = weights / weights.sum() * num_classes
weights = torch.tensor(weights, dtype=torch.float32, device=device)

# ---------------- Train / Val split ----------------
val_len = int(len(dataset) * val_ratio)
train_len = len(dataset) - val_len
train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=combined_model_collate_fn,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=combined_model_collate_fn,
                        num_workers=num_workers, pin_memory=True)

# ---------------- Model ----------------
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
    model = nn.DataParallel(model)
model.to(device)

# ---------------- Loss & Optimizer ----------------
criterion_desc = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
criterion_action = nn.CrossEntropyLoss()
criterion_syn = nn.MSELoss()

log_sigma_desc = torch.nn.Parameter(torch.tensor(0.0, device=device))
log_sigma_action = torch.nn.Parameter(torch.tensor(0.5, device=device))
log_sigma_syn = torch.nn.Parameter(torch.tensor(1.0, device=device))

optimizer = optim.AdamW(
    list(model.parameters()) + [log_sigma_desc, log_sigma_action, log_sigma_syn],
    lr=lr, weight_decay=weight_decay
)

best_val_loss = float('inf')
best_ckpt_path = None
metrics = []

# ---------------- Training Loop ----------------
epoch_bar = tqdm(range(epochs), desc="Training Progress", ncols=100)
for epoch in epoch_bar:
    torch.cuda.empty_cache()
    model.train()
    train_loss_total = 0.0

    # ---- TRAIN LOOP ----
    train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, ncols=100)
    for batch in train_iter:
        batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()

        (desc_logits, action_logits), syn_pred = model(
            batch_device["k_moa1"], batch_device["u_moa1"],
            batch_device["k_moa2"], batch_device["u_moa2"],
            batch_device["k_syn1"], batch_device["u_syn1"],
            batch_device["k_syn2"], batch_device["u_syn2"],
            batch_device["cell_expr"]
        )

        loss_desc = criterion_desc(desc_logits, batch_device["desc_label"])
        loss_action = criterion_action(action_logits, batch_device["action_label"])
        loss_syn = criterion_syn(syn_pred.squeeze(-1), batch_device["syn_score"][:, 0])

        total_loss = (
            0.5 * torch.exp(-2 * log_sigma_desc) * loss_desc + log_sigma_desc +
            0.5 * torch.exp(-2 * log_sigma_action) * loss_action + log_sigma_action +
            0.5 * torch.exp(-2 * log_sigma_syn) * loss_syn + log_sigma_syn
        )

        total_loss.backward()
        optimizer.step()
        train_loss_total += total_loss.item()

        train_iter.set_postfix({
            "desc": f"{loss_desc.item():.4f}",
            "action": f"{loss_action.item():.4f}",
            "syn": f"{loss_syn.item():.4f}",
            "total": f"{total_loss.item():.4f}"
        })

    avg_train_loss = train_loss_total / len(train_loader)

    # ---- VALIDATION LOOP ----
    model.eval()
    val_loss_total = 0.0
    all_val_desc_preds, all_val_desc_labels = [], []
    all_val_action_preds, all_val_action_labels = [], []
    all_val_syn_preds, all_val_syn_labels = [], []

    val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, ncols=100)
    with torch.no_grad():
        for batch in val_iter:
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            (desc_logits, action_logits), syn_pred = model(
                batch_device["k_moa1"], batch_device["u_moa1"],
                batch_device["k_moa2"], batch_device["u_moa2"],
                batch_device["k_syn1"], batch_device["u_syn1"],
                batch_device["k_syn2"], batch_device["u_syn2"],
                batch_device["cell_expr"]
            )

            loss_desc = criterion_desc(desc_logits, batch_device["desc_label"])
            loss_action = criterion_action(action_logits, batch_device["action_label"])
            loss_syn = criterion_syn(syn_pred.squeeze(-1), batch_device["syn_score"][:, 0])

            total_loss = loss_desc + loss_action + loss_syn
            val_loss_total += total_loss.item()

            all_val_desc_preds.append(torch.argmax(desc_logits, dim=-1).cpu().numpy())
            all_val_desc_labels.append(batch_device["desc_label"].cpu().numpy())
            all_val_action_preds.append(torch.argmax(action_logits, dim=-1).cpu().numpy())
            all_val_action_labels.append(batch_device["action_label"].cpu().numpy())
            all_val_syn_preds.append(syn_pred.squeeze(-1).detach().cpu().numpy())
            all_val_syn_labels.append(batch_device["syn_score"][:, 0].cpu().numpy())

    # ---- Metrics ----
    desc_labels_all = np.concatenate(all_val_desc_labels)
    desc_preds_all = np.concatenate(all_val_desc_preds)
    action_labels_all = np.concatenate(all_val_action_labels)
    action_preds_all = np.concatenate(all_val_action_preds)
    syn_labels_all = np.concatenate(all_val_syn_labels)
    syn_preds_all = np.concatenate(all_val_syn_preds)

    # Desc metrics
    desc_acc = accuracy_score(desc_labels_all, desc_preds_all)
    desc_precision = precision_score(desc_labels_all, desc_preds_all, average='weighted', zero_division=0)
    desc_recall = recall_score(desc_labels_all, desc_preds_all, average='weighted', zero_division=0)
    desc_f1 = f1_score(desc_labels_all, desc_preds_all, average='weighted', zero_division=0)
    # Action metrics
    action_acc = accuracy_score(action_labels_all, action_preds_all)
    action_precision = precision_score(action_labels_all, action_preds_all, average='weighted', zero_division=0)
    action_recall = recall_score(action_labels_all, action_preds_all, average='weighted', zero_division=0)
    action_f1 = f1_score(action_labels_all, action_preds_all, average='weighted', zero_division=0)
    # Synergy
    syn_mae = mean_absolute_error(syn_labels_all, syn_preds_all)

    # per-class accuracy
    per_class_desc_acc = {cls: accuracy_score(desc_labels_all[desc_labels_all==cls], desc_preds_all[desc_labels_all==cls])
                          for cls in np.unique(desc_labels_all)}
    per_class_action_acc = {cls: accuracy_score(action_labels_all[action_labels_all==cls], action_preds_all[action_labels_all==cls])
                            for cls in np.unique(action_labels_all)}

    # ---- Print ----
    print(f"\n=== Epoch {epoch+1} Validation Metrics ===")
    print(f"Desc: ACC={desc_acc:.4f}, Precision={desc_precision:.4f}, Recall={desc_recall:.4f}, F1={desc_f1:.4f}")
    print(f"Action: ACC={action_acc:.4f}, Precision={action_precision:.4f}, Recall={action_recall:.4f}, F1={action_f1:.4f}")
    print(f"Synergy MAE: {syn_mae:.4f}")
    print("Per-class Desc Accuracy:")
    for cls, acc in per_class_desc_acc.items():
        print(f"  Class {cls}: {acc:.4f}")
    print("Per-class Action Accuracy:")
    for cls, acc in per_class_action_acc.items():
        print(f"  Class {cls}: {acc:.4f}")

    combined_val_loss = val_loss_total / len(val_loader)
    metrics.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": combined_val_loss,
        "desc_acc": desc_acc,
        "desc_precision": desc_precision,
        "desc_recall": desc_recall,
        "desc_f1": desc_f1,
        "action_acc": action_acc,
        "action_precision": action_precision,
        "action_recall": action_recall,
        "action_f1": action_f1,
        "syn_mae": syn_mae,
        **{f"desc_acc_class{cls}": acc for cls, acc in per_class_desc_acc.items()},
        **{f"action_acc_class{cls}": acc for cls, acc in per_class_action_acc.items()}
    })

    # ---- Save Best Model Only ----
    if combined_val_loss < best_val_loss:
        best_val_loss = combined_val_loss
        ckpt_filename = f"best_epoch{epoch+1}_descAcc{desc_acc:.2f}_actionAcc{action_acc:.2f}_synMAE{syn_mae:.2f}.pt"
        ckpt_path = os.path.join(args.save_dir, ckpt_filename)
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        best_ckpt_path = ckpt_path

    epoch_bar.set_postfix({
        "train_loss": f"{avg_train_loss:.4f}",
        "val_loss": f"{combined_val_loss:.4f}",
        "desc_acc": f"{desc_acc:.3f}",
        "action_acc": f"{action_acc:.3f}",
        "syn_MAE": f"{syn_mae:.3f}"
    })

# ---------------- Save metrics ----------------
os.makedirs(args.save_dir, exist_ok=True)
all_metrics_path = os.path.join(args.save_dir, "all_epochs_metrics_counter.csv")
pd.DataFrame(metrics).to_csv(all_metrics_path, index=False)
print(f"\n All metrics saved to {all_metrics_path}")
