import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class Dataset_moA(Dataset):
    def __init__(self, csv_file, emb_file1, emb_file2, ids_csv, key_name="fps",selected_desc_labels=None):
        """
        Args:
            csv_file: contain drug1_id, drug2_id, interaction_description, action csv file
            emb_file1: .npz file
            emb_file2: .npz file
            ids: list, emb_file1/emb_file2 each row's corresponding drug_id (the same for two models)
            key_name: the array name inside npz  (default 'fps')
        """
        # read CSV
        self.data = pd.read_csv(csv_file)

        # delete missing rows
        #before = len(self.data)
        self.data = self.data.dropna(subset=['mechanism', 'action']).reset_index(drop=True)
        #print(f"[INFO] Dropped {before - len(self.data)} rows with missing labels.")

        if selected_desc_labels is not None:
            self.data = self.data[self.data['mechanism'].isin(selected_desc_labels)].reset_index(drop=True)
        # load npz matrix
        self.embeddings1 = np.load(emb_file1)[key_name]  # shape = (n, dim1)
        self.embeddings2 = np.load(emb_file2)[key_name]  # shape = (n, dim2)

        # drug_id -> row index
        ids_df = pd.read_csv(ids_csv)
        ids = ids_df['drug_id'].tolist()
        self.drug_map = {drug: idx for idx, drug in enumerate(ids)}

        # label encode
        self.data['desc_label'], self.desc_classes = pd.factorize(self.data['mechanism'])
        self.data['action_label'], self.action_classes = pd.factorize(self.data['action'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        d1, d2 = row['drug1_id'], row['drug2_id']

        # via drug_id find embedding
        emb1_d1 = torch.from_numpy(self.embeddings1[self.drug_map[d1]]).float()
        emb2_d1 = torch.from_numpy(self.embeddings2[self.drug_map[d1]]).float()
        emb1_d2 = torch.from_numpy(self.embeddings1[self.drug_map[d2]]).float()
        emb2_d2 = torch.from_numpy(self.embeddings2[self.drug_map[d2]]).float()

        # labels
        label_desc = torch.tensor(row['desc_label'], dtype=torch.long)
        label_action = torch.tensor(row['action_label'], dtype=torch.long)

        return (emb1_d1, emb2_d1, emb1_d2, emb2_d2), (label_desc, label_action)


class Dataset_syn(Dataset):
    def __init__(self, syn_csv, cell_expr_file, emb_file1, emb_file2, ids_csv, 
                 key_name="fps", selected_metrics=None):
        """
        Args:
            syn_csv: CSV, includes drug1_id, drug2_id, cell_line, and synergy scores
            cell_expr_file: CSV, row: cell_line; column: expression features
            emb_file1: npz embedding file (drug embeddings model 1)
            emb_file2: npz embedding file (drug embeddings model 2)
            ids: list, emb_file1/2 each row's corresponding drug_id 
            key_name: the array name inside npz  (default 'fps')
            selected_metrics: list, the synergy score column that need to be returned (defaulr ['ZIP','Bliss','Loewe','HSA'])
        """
        self.data = pd.read_csv(syn_csv)

        expr_df = pd.read_csv(cell_expr_file)
        expr_df = expr_df.set_index("cell_line")   # use cell_line as index
        self.cell_expr = expr_df.to_dict(orient="index")

        # 3. load drug embeddings
        self.embeddings1 = np.load(emb_file1)[key_name]  # shape = (n, dim1)
        self.embeddings2 = np.load(emb_file2)[key_name]  # shape = (n, dim2)

        # 4. construct drug_id -> index mapping
        ids_df = pd.read_csv(ids_csv)
        ids = ids_df['drug_id'].tolist()
        self.drug_map = {drug: idx for idx, drug in enumerate(ids)}

        # 5. synergy score columns
        if selected_metrics is None:
            self.metrics = ["ZIP", "Bliss", "Loewe", "HSA"]
        else:
            self.metrics = selected_metrics

        # 6. check for missing cell_lines
        #missing = set(self.data["cell_line"]) - set(expr_df.index)
        #if len(missing) > 0:
        #    print(f"[WARNING] {len(missing)} cell lines in full_syn not found in cell_expr: {list(missing)[:5]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        d1, d2 = row["drug1_id"], row["drug2_id"]

        # drug embeddings
        emb1_d1 = torch.from_numpy(self.embeddings1[self.drug_map[d1]]).float()
        emb2_d1 = torch.from_numpy(self.embeddings2[self.drug_map[d1]]).float()
        emb1_d2 = torch.from_numpy(self.embeddings1[self.drug_map[d2]]).float()
        emb2_d2 = torch.from_numpy(self.embeddings2[self.drug_map[d2]]).float()

        # synergy scores
        scores = torch.tensor([row[m] for m in self.metrics], dtype=torch.float)

        # cell expression
        expr_values = list(self.cell_expr[row["cell_line"]].values())
        expr_tensor = torch.tensor(expr_values, dtype=torch.float)

        return (emb1_d1, emb2_d1, emb1_d2, emb2_d2), (scores, expr_tensor)


class CombDataset(Dataset):
    """
    Dataset for combined MoA classification and Synergy regression.
    make sure the orders of drugs as input into both two pretrained models are the same.
    Args:
        moa_csv, syn_csv, cell_expr_csv: data files
        emb_file1, emb_file2: pretrained embeddings
        ids_csv: CSV file with a single column 'drug_id' listing all drugs
        key_name: key in .npz embeddings file
        selected_desc_labels: optional list to filter MoA mechanisms
        selected_metrics: optional list to filter synergy metrics
    """
    def __init__(self, moa_csv, syn_csv, cell_expr_csv,
                 emb_file1, emb_file2, ids_csv,
                 key_name="fps", selected_desc_labels=None, selected_metrics=None):

        # ----------------- Load IDs CSV -----------------
        ids_df = pd.read_csv(ids_csv)
        ids = ids_df['drug_id'].tolist()
        self.drug_map = {drug: idx for idx, drug in enumerate(ids)}

        # ----------------- MoA -----------------
        self.moa_data = pd.read_csv(moa_csv).dropna(subset=['mechanism','action']).reset_index(drop=True)
        if selected_desc_labels:
            self.moa_data = self.moa_data[self.moa_data['mechanism'].isin(selected_desc_labels)].reset_index(drop=True)

        self.emb1 = np.load(emb_file1)[key_name]
        self.emb2 = np.load(emb_file2)[key_name]

        self.moa_data['desc_label'], self.desc_classes = pd.factorize(self.moa_data['mechanism'])
        self.moa_data['action_label'], self.action_classes = pd.factorize(self.moa_data['action'])

        # ----------------- Synergy -----------------
        self.syn_data = pd.read_csv(syn_csv).reset_index(drop=True)
        expr_df = pd.read_csv(cell_expr_csv).set_index("cell_line")
        self.cell_expr_dict = expr_df.to_dict(orient="index")
        self.expr_df = expr_df
        if selected_metrics:
            if isinstance(selected_metrics, list):
                assert len(selected_metrics) == 1, "Only one metric should be selected."
                self.metric = selected_metrics[0]
            else:
                self.metric = selected_metrics
        else:
            self.metric = "Loewe"

        # ----------------- Length -----------------
        self.len_moa = len(self.moa_data)
        self.len_syn = len(self.syn_data)
        self.total_len = max(self.len_moa, self.len_syn)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # ----------------- MoA -----------------
        idx_moa = idx % self.len_moa
        row_moa = self.moa_data.iloc[idx_moa]
        d1_moa, d2_moa = row_moa['drug1_id'], row_moa['drug2_id']
        emb1_d1_moa = torch.from_numpy(self.emb1[self.drug_map[d1_moa]]).float()
        emb2_d1_moa = torch.from_numpy(self.emb2[self.drug_map[d1_moa]]).float()
        emb1_d2_moa = torch.from_numpy(self.emb1[self.drug_map[d2_moa]]).float()
        emb2_d2_moa = torch.from_numpy(self.emb2[self.drug_map[d2_moa]]).float()
        desc_label = torch.tensor(row_moa['desc_label'], dtype=torch.long)
        action_label = torch.tensor(row_moa['action_label'], dtype=torch.long)

        # ----------------- Synergy -----------------
        idx_syn = idx % self.len_syn
        row_syn = self.syn_data.iloc[idx_syn]
        d1_syn, d2_syn = row_syn['drug1_id'], row_syn['drug2_id']
        emb1_d1_syn = torch.from_numpy(self.emb1[self.drug_map[d1_syn]]).float()
        emb2_d1_syn = torch.from_numpy(self.emb2[self.drug_map[d1_syn]]).float()
        emb1_d2_syn = torch.from_numpy(self.emb1[self.drug_map[d2_syn]]).float()
        emb2_d2_syn = torch.from_numpy(self.emb2[self.drug_map[d2_syn]]).float()
        syn_score = torch.tensor([row_syn[self.metric]], dtype=torch.float)
        cell_expr_tensor = torch.tensor(self.expr_df.loc[row_syn["cell_line"]].values, dtype=torch.float)

        return (emb1_d1_moa, emb2_d1_moa, emb1_d2_moa, emb2_d2_moa,
                emb1_d1_syn, emb2_d1_syn, emb1_d2_syn, emb2_d2_syn,
                cell_expr_tensor, (desc_label, action_label), syn_score)

class new_CombDataset(Dataset):
    """
    Combined MoA + Synergy dataset
    - Find embedding thorugh drug_id 
    - return a dict that includes drug_id、cell_line info
    """



    def __init__(self, moa_csv, syn_csv, cell_expr_csv,
                 emb_file1, emb_file2, ids_csv,
                 key_name="fps", selected_desc_labels=None, selected_metrics=None):
        import pandas as pd
        import numpy as np
        import torch

        # selected_desc_labels: current task's mechanism list


        # ----------------- Load IDs & embeddings -----------------
        ids = pd.read_csv(ids_csv)['drug_id'].tolist()
        emb1_arr = np.load(emb_file1)[key_name]
        emb2_arr = np.load(emb_file2)[key_name]

        assert len(ids) == len(emb1_arr) == len(emb2_arr), "ids_csv and npz embedding have inconsistent number of rows!"

        # directly generate drug_id -> embedding dict
        self.emb1_dict = {drug: torch.from_numpy(emb1_arr[i]).float() for i, drug in enumerate(ids)}
        self.emb2_dict = {drug: torch.from_numpy(emb2_arr[i]).float() for i, drug in enumerate(ids)}

        # ----------------- MoA -----------------
        self.moa_data = pd.read_csv(moa_csv).dropna(subset=['mechanism','action']).reset_index(drop=True)
        if selected_desc_labels:
            self.moa_data = self.moa_data[self.moa_data['mechanism'].isin(selected_desc_labels)].reset_index(drop=True)
        
        desc_classes = list(selected_desc_labels)
        action_classes = ["increase", "decrease"]

        self.desc_map = {m: i for i, m in enumerate(desc_classes)}
        self.action_map = {a: i for i, a in enumerate(action_classes)}

        self.moa_data['desc_label'] = self.moa_data['mechanism'].map(lambda x: self.desc_map.get(x, -1))
        self.moa_data['action_label'] = self.moa_data['action'].map(lambda x: self.action_map.get(x, -1))

        # ----------------- Synergy -----------------
        self.syn_data = pd.read_csv(syn_csv).reset_index(drop=True)
        expr_df = pd.read_csv(cell_expr_csv).set_index("cell_line")
        self.cell_expr_dict = {cl: torch.tensor(row.values, dtype=torch.float) for cl, row in expr_df.iterrows()}
        if selected_metrics:
            if isinstance(selected_metrics, list):
                assert len(selected_metrics) == 1, "Only one metric should be selected."
                self.metric = selected_metrics[0]
            else:
                self.metric = selected_metrics
        else:
            self.metric = "Loewe"

        # ----------------- Length -----------------
        self.len_moa = len(self.moa_data)
        self.len_syn = len(self.syn_data)
        self.total_len = max(self.len_moa, self.len_syn)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # ----------------- MoA -----------------
        idx_moa = idx % self.len_moa
        row_moa = self.moa_data.iloc[idx_moa]
        d1_moa, d2_moa = row_moa['drug1_id'], row_moa['drug2_id']

        emb1_d1_moa = self.emb1_dict[d1_moa]
        emb2_d1_moa = self.emb2_dict[d1_moa]
        emb1_d2_moa = self.emb1_dict[d2_moa]
        emb2_d2_moa = self.emb2_dict[d2_moa]

        #mask1 = ((self.moa_data['drug1_id'] == d1_moa) &(self.moa_data['drug2_id'] == d2_moa))

        #desc_label = self.moa_data.loc[mask1, 'desc_label'].values

        desc_label = torch.tensor(row_moa['desc_label'], dtype=torch.long)
        #desc_label = torch.tensor(desc_label, dtype=torch.long)
        action_label = torch.tensor(row_moa['action_label'], dtype=torch.long)

        # ----------------- Synergy -----------------
        idx_syn = idx % self.len_syn
        row_syn = self.syn_data.iloc[idx_syn]
        d1_syn, d2_syn = row_syn['drug1_id'], row_syn['drug2_id']

        emb1_d1_syn = self.emb1_dict[d1_syn]
        emb2_d1_syn = self.emb2_dict[d1_syn]
        emb1_d2_syn = self.emb1_dict[d2_syn]
        emb2_d2_syn = self.emb2_dict[d2_syn]

        cell_line = row_syn["cell_line"]
        #mask2 = ((self.syn_data['drug1_id'] == d1_syn) &(self.syn_data['drug2_id'] == d2_syn) &(self.syn_data['cell_line'] == cell_line))

        #syn_score = self.syn_data.loc[mask2, self.metric].values

        #syn_score = torch.tensor(syn_score, dtype=torch.float)
        syn_score = torch.tensor([row_syn[self.metric]], dtype=torch.float)
        cell_expr_tensor = self.cell_expr_dict[row_syn["cell_line"]]

        # return a dict
        return {
            "k_moa1": emb1_d1_moa,
            "u_moa1": emb2_d1_moa,
            "k_moa2": emb1_d2_moa,
            "u_moa2": emb2_d2_moa,
            "desc_label": desc_label,
            "action_label": action_label,
            "moa_drug1_id": d1_moa,
            "moa_drug2_id": d2_moa,

            "k_syn1": emb1_d1_syn,
            "u_syn1": emb2_d1_syn,
            "k_syn2": emb1_d2_syn,
            "u_syn2": emb2_d2_syn,
            "syn_score": syn_score,
            "cell_expr": cell_expr_tensor,
            "syn_drug1_id": d1_syn,
            "syn_drug2_id": d2_syn,
            "cell_expr_line": row_syn["cell_line"]
        }


class MoADataset(Dataset):
    """
    """
    def __init__(self, syn_csv, moa_csv, emb_file1, emb_file2, ids_csv,
                 key_name="fps", selected_desc_labels=None):
        ids = pd.read_csv(ids_csv)['drug_id'].tolist()
        emb1_arr = np.load(emb_file1)[key_name]
        emb2_arr = np.load(emb_file2)[key_name]
        assert len(ids) == len(emb1_arr) == len(emb2_arr), "ids_csv and npz embedding have inconsistent number of rows!"

        # generate embedding dict
        self.emb1_dict = {drug: torch.from_numpy(emb1_arr[i]).float() for i, drug in enumerate(ids)}
        self.emb2_dict = {drug: torch.from_numpy(emb2_arr[i]).float() for i, drug in enumerate(ids)}

        # ----------------- MoA dataset -----------------
        self.moa_data = pd.read_csv(moa_csv).dropna(subset=['mechanism', 'action']).reset_index(drop=True)
        if selected_desc_labels:
            self.moa_data = self.moa_data[self.moa_data['mechanism'].isin(selected_desc_labels)].reset_index(drop=True)
        else:
            selected_desc_labels = self.moa_data['mechanism'].unique().tolist()

        desc_classes = list(selected_desc_labels)
        action_classes = ["increase", "decrease"]

        self.desc_map = {m: i for i, m in enumerate(desc_classes)}
        self.action_map = {a: i for i, a in enumerate(action_classes)}

        self.moa_data['desc_label'] = self.moa_data['mechanism'].map(lambda x: self.desc_map.get(x, -1))
        self.moa_data['action_label'] = self.moa_data['action'].map(lambda x: self.action_map.get(x, -1))

        self.syn_data = pd.read_csv(syn_csv).reset_index(drop=True)
        self.len_moa = len(self.moa_data)
        self.len_syn = len(self.syn_data)
        self.total_len = max(self.len_moa, self.len_syn)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        idx_moa = idx % self.len_moa
        row = self.moa_data.iloc[idx_moa]
        d1, d2 = row['drug1_id'], row['drug2_id']

        emb1_d1 = self.emb1_dict[d1]
        emb2_d1 = self.emb2_dict[d1]
        emb1_d2 = self.emb1_dict[d2]
        emb2_d2 = self.emb2_dict[d2]

        desc_label = torch.tensor(row['desc_label'], dtype=torch.long)
        action_label = torch.tensor(row['action_label'], dtype=torch.long)

        return {
            "k_moa1": emb1_d1,
            "u_moa1": emb2_d1,
            "k_moa2": emb1_d2,
            "u_moa2": emb2_d2,
            "desc_label": desc_label,
            "action_label": action_label,
            "drug1_id": d1,
            "drug2_id": d2
        }


import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SynergyDataset(Dataset):
    """
    Synergy-only dataset
    - for synergy regression task for drug pairs
    - includes cell expression
    """
    def __init__(self, syn_csv, cell_expr_csv,
                 emb_file1, emb_file2, ids_csv,
                 key_name="fps", selected_metrics=None):
        ids = pd.read_csv(ids_csv)['drug_id'].tolist()
        emb1_arr = np.load(emb_file1)[key_name]
        emb2_arr = np.load(emb_file2)[key_name]
        assert len(ids) == len(emb1_arr) == len(emb2_arr), "ids_csv and npz embedding have inconsistent number of rows!"

        # generate embedding dict
        self.emb1_dict = {drug: torch.from_numpy(emb1_arr[i]).float() for i, drug in enumerate(ids)}
        self.emb2_dict = {drug: torch.from_numpy(emb2_arr[i]).float() for i, drug in enumerate(ids)}

        # ----------------- Synergy dataset -----------------
        self.syn_data = pd.read_csv(syn_csv).reset_index(drop=True)
        expr_df = pd.read_csv(cell_expr_csv).set_index("cell_line")
        self.cell_expr_dict = {cl: torch.tensor(row.values, dtype=torch.float) for cl, row in expr_df.iterrows()}

        if selected_metrics:
            if isinstance(selected_metrics, list):
                assert len(selected_metrics) == 1, "only support one synergy metric."
                self.metric = selected_metrics[0]
            else:
                self.metric = selected_metrics
        else:
            self.metric = "Loewe"

    def __len__(self):
        return len(self.syn_data)

    def __getitem__(self, idx):
        row = self.syn_data.iloc[idx]
        d1, d2, cell_line = row['drug1_id'], row['drug2_id'], row['cell_line']

        emb1_d1 = self.emb1_dict[d1]
        emb2_d1 = self.emb2_dict[d1]
        emb1_d2 = self.emb1_dict[d2]
        emb2_d2 = self.emb2_dict[d2]
        syn_score = torch.tensor([row[self.metric]], dtype=torch.float)
        cell_expr = self.cell_expr_dict[cell_line]

        return {
            "k_syn1": emb1_d1,
            "u_syn1": emb2_d1,
            "k_syn2": emb1_d2,
            "u_syn2": emb2_d2,
            "syn_score": syn_score,
            "cell_expr": cell_expr,
            "drug1_id": d1,
            "drug2_id": d2,
            "cell_line": cell_line
        }

class Simple_CombDataset(Dataset):
    """
    Combined MoA + Synergy dataset
    - use single embedding file (emb_file)
    - adapt to Simplereduction model: every drug only correspond to one embedding
    """

    def __init__(self, moa_csv, syn_csv, cell_expr_csv,
                 emb_file, ids_csv,
                 key_name="fps", selected_desc_labels=None, selected_metrics=None):
        # ----------------- Load IDs & embeddings -----------------
        ids = pd.read_csv(ids_csv)['drug_id'].tolist()
        emb_arr = np.load(emb_file)[key_name]
        assert len(ids) == len(emb_arr), "ids_csv and npz embedding have inconsistent number of rows!"

        # directly generate drug_id -> embedding dict
        self.emb_dict = {drug: torch.from_numpy(emb_arr[i]).float() for i, drug in enumerate(ids)}

        # ----------------- MoA -----------------
        self.moa_data = pd.read_csv(moa_csv).dropna(subset=['mechanism', 'action']).reset_index(drop=True)
        if selected_desc_labels:
            self.moa_data = self.moa_data[self.moa_data['mechanism'].isin(selected_desc_labels)].reset_index(drop=True)
        
        desc_classes = list(selected_desc_labels)
        action_classes = ["increase", "decrease"]

        self.desc_map = {m: i for i, m in enumerate(desc_classes)}
        self.action_map = {a: i for i, a in enumerate(action_classes)}

        self.moa_data['desc_label'] = self.moa_data['mechanism'].map(lambda x: self.desc_map.get(x, -1))
        self.moa_data['action_label'] = self.moa_data['action'].map(lambda x: self.action_map.get(x, -1))

        # ----------------- Synergy -----------------
        self.syn_data = pd.read_csv(syn_csv).reset_index(drop=True)
        expr_df = pd.read_csv(cell_expr_csv).set_index("cell_line")
        self.cell_expr_dict = {cl: torch.tensor(row.values, dtype=torch.float) for cl, row in expr_df.iterrows()}
        
        if selected_metrics:
            if isinstance(selected_metrics, list):
                assert len(selected_metrics) == 1, "Only one metric should be selected."
                self.metric = selected_metrics[0]
            else:
                self.metric = selected_metrics
        else:
            self.metric = "Loewe"

        # ----------------- Length -----------------
        self.len_moa = len(self.moa_data)
        self.len_syn = len(self.syn_data)
        self.total_len = max(self.len_moa, self.len_syn)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # ----------------- MoA -----------------
        idx_moa = idx % self.len_moa
        row_moa = self.moa_data.iloc[idx_moa]
        d1_moa, d2_moa = row_moa['drug1_id'], row_moa['drug2_id']

        emb_d1_moa = self.emb_dict[d1_moa]
        emb_d2_moa = self.emb_dict[d2_moa]

        desc_label = torch.tensor(row_moa['desc_label'], dtype=torch.long)
        action_label = torch.tensor(row_moa['action_label'], dtype=torch.long)

        # ----------------- Synergy -----------------
        idx_syn = idx % self.len_syn
        row_syn = self.syn_data.iloc[idx_syn]
        d1_syn, d2_syn = row_syn['drug1_id'], row_syn['drug2_id']

        emb_d1_syn = self.emb_dict[d1_syn]
        emb_d2_syn = self.emb_dict[d2_syn]

        syn_score = torch.tensor([row_syn[self.metric]], dtype=torch.float)
        cell_expr_tensor = self.cell_expr_dict[row_syn["cell_line"]]

        # ----------------- Return dictionary -----------------
        return {
            "moa1": emb_d1_moa,
            "moa2": emb_d2_moa,
            "desc_label": desc_label,
            "action_label": action_label,
            "moa_drug1_id": d1_moa,
            "moa_drug2_id": d2_moa,

            "syn1": emb_d1_syn,
            "syn2": emb_d2_syn,
            "syn_score": syn_score,
            "cell_expr": cell_expr_tensor,
            "syn_drug1_id": d1_syn,
            "syn_drug2_id": d2_syn,
            "cell_expr_line": row_syn["cell_line"]
        }
