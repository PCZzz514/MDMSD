import os
import numpy as np
import pandas as pd
import lmdb
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import pickle
import glob
from multiprocessing import Pool
from collections import defaultdict


smi_list = [
'CC1=C(C(=O)OC2CCCC2)[C@H](c2ccccc2OC(C)C)C2=C(O)CC(C)(C)CC2=[N+]1',
'COc1cccc(-c2nc(C(=O)NC[C@H]3CCCO3)cc3c2[nH]c2ccccc23)c1',
'O=C1c2ccccc2C(=O)c2c1ccc(C(=O)n1nc3c4c(cccc41)C(=O)c1ccccc1-3)c2[N+](=O)[O-]',
'COc1cc(/C=N/c2nonc2NC(C)=O)ccc1OC(C)C',
'CCC[C@@H]1CN(Cc2ccc3nsnc3c2)C[C@H]1NS(C)(=O)=O',
'CCc1nnc(N/C(O)=C/CCOc2ccc(OC)cc2)s1',
'CC(C)(C)SCCN/C=C1\C(=O)NC(=O)N(c2ccc(Br)cc2)C1=O',
'CC(C)(C)c1nc(COc2ccc3c(c2)CCn2c-3cc(OCC3COCCO3)nc2=O)no1',
'N#CCCNS(=O)(=O)c1ccc(/C(O)=N/c2ccccc2Oc2ccccc2Cl)cc1',
'O=C(Nc1ncc(Cl)s1)c1cccc(S(=O)(=O)Nc2ccc(Br)cc2)c1',
]

def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates


def smi2_3Dcoords(smi,cnt):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)            
                    
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi) 
        except:
            print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi) 

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def inner_smi2coords(content):
    smi = content
    cnt = 10 # conformer num,all==11, 10 3d + 1 2d

    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list =  [smi2_2Dcoords(smi)] * (cnt+1)
        print("atom num >400,use 2D coords",smi)
    else:
        coordinate_list = smi2_3Dcoords(smi,cnt)
        # add 2d conf
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H 
    return pickle.dumps({'atoms': atoms, 'coordinates': coordinate_list, 'smi': smi }, protocol=-1)


def smi2coords(content):
    try:
        return inner_smi2coords(content)
    except:
        print("failed smiles: {}".format(content[0]))
        return None


def write_lmdb(smiles_list, job_name, seed=42, outpath='./results', nthreads=8):
    os.makedirs(outpath, exist_ok=True)
    output_name = os.path.join(outpath,'{}.lmdb'.format(job_name))
    try:
        os.remove(output_name)
    except:
        pass
    env_new = lmdb.open(
        output_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    with Pool(nthreads) as pool:
        i = 0
        for inner_output in tqdm(pool.imap(smi2coords, smiles_list)):
            if inner_output is not None:
                txn_write.put(f'{i}'.encode("ascii"), inner_output)
                i += 1
        print('{} process {} lines'.format(job_name, i))
        txn_write.commit()
        env_new.close()
        
        
# 读取 CSV 并转换为列表
csv_path = 'oneil_ids(2).csv'
df = pd.read_csv(csv_path)
data_list = df['smiles'].values.tolist()  # 转换为二维列表

seed = 42
job_name = '1200drug_new'   # replace to your custom name
data_path = './results'  # replace to your data path
weight_path='../ckp/mol_pre_no_h_220816.pt'  # replace to your ckpt path
only_polar=0  # no h
dict_name='dict.txt'
batch_size=16
conf_size=1  # default 10 3d + 1 2d
results_path=data_path   # replace to your save path
write_lmdb(data_list, job_name=job_name, seed=seed, outpath=data_path)