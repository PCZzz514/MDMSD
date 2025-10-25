import torch

def moaddi_collate_fn(batch):
    """
    Collate function for MoADDI model (multi-class classification for MoA).
    
    batch: list of tuples from Dataset_moA:
           ((emb1_d1, emb2_d1, emb1_d2, emb2_d2), (label_desc, label_action))
           
    Returns:
        drug_emb_batch: tuple of 4 tensors (batch, emb_dim)
        desc_labels: tensor (batch,) long
        action_labels: tensor (batch,) long
    """
    emb1_list, emb2_list, emb3_list, emb4_list = [], [], [], []
    desc_labels, action_labels = [], []

    for (emb1_d1, emb2_d1, emb1_d2, emb2_d2), (label_desc, label_action) in batch:
        emb1_list.append(emb1_d1)
        emb2_list.append(emb2_d1)
        emb3_list.append(emb1_d2)
        emb4_list.append(emb2_d2)
        desc_labels.append(label_desc)
        action_labels.append(label_action)
    
    drug_emb_batch = (
        torch.stack(emb1_list, dim=0),
        torch.stack(emb2_list, dim=0),
        torch.stack(emb3_list, dim=0),
        torch.stack(emb4_list, dim=0)
    )
    
    desc_labels = torch.tensor(desc_labels, dtype=torch.long)
    action_labels = torch.tensor(action_labels, dtype=torch.long)
    
    return drug_emb_batch, desc_labels, action_labels

def synergy_collate_fn(batch):
    """
    Collate function for SynergyDDI regression model.
    
    batch: list of tuples from Dataset_syn:
           ((emb1_d1, emb2_d1, emb1_d2, emb2_d2), (scores, expr_tensor))
           
    Returns:
        drug_emb_batch: tuple of 4 tensors (batch, emb_dim)
        target_batch: (batch, num_scores)
        expr_batch: (batch, cell_expr_dim)
    """
    emb1_list, emb2_list, emb3_list, emb4_list = [], [], [], []
    target_list, expr_list = [], []
    
    for (emb1_d1, emb2_d1, emb1_d2, emb2_d2), (scores, expr_tensor) in batch:
        emb1_list.append(emb1_d1)
        emb2_list.append(emb2_d1)
        emb3_list.append(emb1_d2)
        emb4_list.append(emb2_d2)
        target_list.append(scores)
        expr_list.append(expr_tensor)
    
    drug_emb_batch = (
        torch.stack(emb1_list, dim=0),
        torch.stack(emb2_list, dim=0),
        torch.stack(emb3_list, dim=0),
        torch.stack(emb4_list, dim=0)
    )
    
    target_batch = torch.stack(target_list, dim=0)
    expr_batch = torch.stack(expr_list, dim=0)
    
    return drug_emb_batch, target_batch, expr_batch



def combined_collate_fn(batch):
    """
    Collate function for CombinedDataset.

    Each batch element is a tuple:
    (emb1_d1_moa, emb2_d1_moa, emb1_d2_moa, emb2_d2_moa,
     emb1_d1_syn, emb2_d1_syn, emb1_d2_syn, emb2_d2_syn,
     cell_expr_tensor, syn_score, (desc_label, action_label))
    """

    # MoA embeddings
    emb1_d1_moa = torch.stack([item[0] for item in batch])
    emb2_d1_moa = torch.stack([item[1] for item in batch])
    emb1_d2_moa = torch.stack([item[2] for item in batch])
    emb2_d2_moa = torch.stack([item[3] for item in batch])

    # Synergy embeddings
    emb1_d1_syn = torch.stack([item[4] for item in batch])
    emb2_d1_syn = torch.stack([item[5] for item in batch])
    emb1_d2_syn = torch.stack([item[6] for item in batch])
    emb2_d2_syn = torch.stack([item[7] for item in batch])

    # cell expression
    cell_expr_tensor = torch.stack([item[8] for item in batch])

    # synergy score
    syn_score = torch.stack([item[10].clone().detach().float() if torch.is_tensor(item[10]) 
                         else torch.tensor(item[10], dtype=torch.float32)
                         for item in batch])



    # MoA labes
    desc_label = torch.stack([item[9][0] for item in batch])
    action_label = torch.stack([item[9][1] for item in batch])

    return {
        "k_moa1": emb1_d1_moa,
        "u_moa1": emb2_d1_moa,
        "k_moa2": emb1_d2_moa,
        "u_moa2": emb2_d2_moa,
        "k_syn1": emb1_d1_syn,
        "u_syn1": emb2_d1_syn,
        "k_syn2": emb1_d2_syn,
        "u_syn2": emb2_d2_syn,
        "cell_expr": cell_expr_tensor,
        "desc_label": desc_label,
        "action_label": action_label,
        "syn_score": syn_score
    }



import torch

def combined_model_collate_fn(batch):
    """
    Collate function for CombDataset to feed CombinedTwoModel.
    
    Args:
        batch: list of dicts, each dict from CombDataset
    Returns:
        dict with:
            - model inputs: tensors in order of forward()
            - labels: MoA labels, Synergy scores
            - optional ids/cell_line
    """
    batch_dict = {}

    # --- MoA embeddings ---
    batch_dict["k_moa1"] = torch.stack([x["k_moa1"] for x in batch])
    batch_dict["u_moa1"] = torch.stack([x["u_moa1"] for x in batch])
    batch_dict["k_moa2"] = torch.stack([x["k_moa2"] for x in batch])
    batch_dict["u_moa2"] = torch.stack([x["u_moa2"] for x in batch])

    # --- Synergy embeddings ---
    batch_dict["k_syn1"] = torch.stack([x["k_syn1"] for x in batch])
    batch_dict["u_syn1"] = torch.stack([x["u_syn1"] for x in batch])
    batch_dict["k_syn2"] = torch.stack([x["k_syn2"] for x in batch])
    batch_dict["u_syn2"] = torch.stack([x["u_syn2"] for x in batch])

    # --- Cell expression ---
    batch_dict["cell_expr"] = torch.stack([x["cell_expr"] for x in batch])

    # --- MoA labels ---
    batch_dict["desc_label"] = torch.stack([x["desc_label"] for x in batch])
    batch_dict["action_label"] = torch.stack([x["action_label"] for x in batch])

    # --- Synergy score ---
    batch_dict["syn_score"] = torch.stack([x["syn_score"] for x in batch])

    # --- Optional: IDs / cell_line for debug / tracking ---
    batch_dict["moa_drug1_id"] = [x["moa_drug1_id"] for x in batch]
    batch_dict["moa_drug2_id"] = [x["moa_drug2_id"] for x in batch]
    batch_dict["syn_drug1_id"] = [x["syn_drug1_id"] for x in batch]
    batch_dict["syn_drug2_id"] = [x["syn_drug2_id"] for x in batch]
    batch_dict["cell_expr_line"] = [x["cell_expr_line"] for x in batch]

    return batch_dict


def MoADDI_collator(batch):
    batch_dict = {}
    batch_dict["k_moa1"] = torch.stack([x["k_moa1"] for x in batch])
    batch_dict["u_moa1"] = torch.stack([x["u_moa1"] for x in batch])
    batch_dict["k_moa2"] = torch.stack([x["k_moa2"] for x in batch])
    batch_dict["u_moa2"] = torch.stack([x["u_moa2"] for x in batch])

    batch_dict["desc_label"] = torch.stack([x["desc_label"] for x in batch])
    batch_dict["action_label"] = torch.stack([x["action_label"] for x in batch])

    return batch_dict

def SynergyDDI_collator(batch):
    batch_dict = {}
    batch_dict["k_syn1"] = torch.stack([x["k_syn1"] for x in batch])
    batch_dict["u_syn1"] = torch.stack([x["u_syn1"] for x in batch])
    batch_dict["k_syn2"] = torch.stack([x["k_syn2"] for x in batch])
    batch_dict["u_syn2"] = torch.stack([x["u_syn2"] for x in batch])

    batch_dict["cell_expr"] = torch.stack([x["cell_expr"] for x in batch])
    batch_dict["syn_score"] = torch.stack([x["syn_score"] for x in batch])

    return batch_dict

def simple_collator(batch):
    """
    Collate function for new_CombDataset to feed the Simplereduction-based CombinedTwoModel.
    
    Args:
        batch: list of dicts, each dict from new_CombDataset
    Returns:
        dict with:
            - model inputs: tensors in order of forward()
            - labels: MoA labels, Synergy scores
            - optional ids/cell_line
    """
    batch_dict = {}

    # --- MoA embeddings ---
    batch_dict["moa1"] = torch.stack([x["moa1"] for x in batch])
    batch_dict["moa2"] = torch.stack([x["moa2"] for x in batch])

    # --- Synergy embeddings ---
    batch_dict["syn1"] = torch.stack([x["syn1"] for x in batch])
    batch_dict["syn2"] = torch.stack([x["syn2"] for x in batch])

    # --- Cell expression ---
    batch_dict["cell_expr"] = torch.stack([x["cell_expr"] for x in batch])

    # --- MoA labels ---
    batch_dict["desc_label"] = torch.stack([x["desc_label"] for x in batch])
    batch_dict["action_label"] = torch.stack([x["action_label"] for x in batch])

    # --- Synergy score ---
    batch_dict["syn_score"] = torch.stack([x["syn_score"] for x in batch])

    return batch_dict


