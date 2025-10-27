import pickle
import numpy as np

path = './results/ckp_1200drug_new.out.pkl'
# 读取 .pkl 文件
with open(path, 'rb') as f:  # 注意必须是 'rb' 模式（二进制读取）
    data = pickle.load(f)
print('data_length:', len(data))
conf = 1
tail = len(data[-1]['mol_repr_cls'])
num_smiles = int(((len(data) - 1) * 16 + tail) / conf)
print('num_smiles:', num_smiles)

# # 通用属性查看
# print("字典的键:", data[0]['data_name'])

def inspect_value(value, indent=0):
    prefix = "  " * indent
    if isinstance(value, dict):
        print(f"{prefix}字典 (长度: {len(value)})")
        for k, v in value.items():
            print(f"{prefix}  Key: {k}")
            inspect_value(v, indent + 2)
    elif hasattr(value, 'shape'):  # NumPy/PyTorch/TensorFlow 对象
        print(f"{prefix}数组/张量, Shape: {value.shape}, 类型: {type(value)}")
    elif isinstance(value, (list, tuple)):
        print(f"{prefix}列表/元组 (长度: {len(value)}), 首元素类型: {type(value[0]) if len(value) > 0 else '空'}")
        if len(value) > 0:
            inspect_value(value[0], indent + 1)
    else:
        print(f"{prefix}值类型: {type(value)}, 示例: {str(value)[:50]}...")

# 遍历字典的每个键值对
for key, value in data[0].items():
    print(f"\n=== 键: {key} ===")
    inspect_value(value)
   
data_mol_repr = np.zeros((len(data), 16, 512))  # conf_size 4 or 11 
for i in range(len(data) - 1) :
    data_mol_repr[i] = data[i]['mol_repr_cls']
original = data[i + 1]['mol_repr_cls']  # 形状 (2, 512)
padded = np.zeros((16, 512))
padded[:original.shape[0], :] = original  # 将原始数据放入前2行
data_mol_repr[i + 1] = padded
    
data_mol_repr_flatten = data_mol_repr.reshape(-1, 512)  # (num_molecules * conf_size, 512)
data_mol_repr_flatten = data_mol_repr_flatten[:conf * num_smiles].reshape(-1, 512 * conf)  # (num_molecules, 512 * conf_size)
# data_mol_repr_flatten = data_mol_repr_flatten[:, :768]
print('data_mol_repr_flatten:', data_mol_repr_flatten.shape)  # (num_molecules, 512 * conf_size)

# Save npz file
save_path = 'data_mol_repr_1200drug_new.npz'
key_name = 'fps'
np.savez(save_path, **{key_name: data_mol_repr_flatten})
