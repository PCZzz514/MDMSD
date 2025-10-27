# MDMSD

MDMSD is a multitask model that could predict two drug-drug interaction tasks: synergy and Mechanism of Action at the same time
![Model Sturcture](./model structure.jpg)


## Extract features

First, you need to extract drug features from Uni-Mol and KPGT. 

For Uni-Mol, first follow Uni-Mol's original GitHub (https://github.com/deepmodeling/Uni-Mol/tree/main) to build an environment, and then copy the three files in MDMSD's unimol folder and pasted them into unimol's folder. Finally, use the below command to generate embeddings: 

```
python data_repr.py
CUDA_VISIBLE_DEVICES="1" python ./unimol/infer.py --user-dir ./unimol ./results --valid-subset your_name --results-path ./results --num-workers 8 --ddp-backend=c10d --batch-size 16 --task unimol --loss unimol_infer --arch unimol_base --path ../ckp/mol_pre_no_h_220816.pt --only-polar 1 --dict-name dict.txt --conf-size 1 --log-interval 50 --log-format simple --random-token-prob 0 --leave-unmasked-prob 1.0 --mode infer
python read_pkl.py
```

For KPGT, also follow KPGT's "Generate latent features for your datasets" part on GitHub (https://github.com/lihan97/KPGT) to extract features.

## Train and test MDMSD

Using the following command to train the model:
```
python train_counter.py --moa_csv datasets/moa/moa_train.csv --syn_csv datasets/syn/syn_filtered.csv --cell_expr_csv datasets/ccle_expr_norm.csv --emb_file1 datasets/ids/kpgt_base.npz --emb_file2 datasets/ids/data_mol_repr_2000drug_conf1.npz --ids_file datasets/ids_smiles/ids_smiles.csv --gpus 0
```

and use the following command to test the model:
```
python test.py
```
