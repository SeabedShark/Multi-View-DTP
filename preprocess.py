import os
import pandas as pd
import numpy as np
import torch
import pickle
import dgl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import random

from protein_fingerprint import generate_protein_fingerprint
import torch.nn.utils.rnn as rnn_utils

# 需要在preprocess.py文件顶部添加以下导入
from embeddings import get_embedding, train_all_embeddings, load_embeddings


# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)


# 数据保存和加载函数
def dump(fm, f):
    file = open(fm, "wb")
    pickle.dump(f, file)
    file.close()


def load(fm):
    file = open(fm, "rb")
    f = pickle.load(file)
    file.close()
    return f


# SMILES处理的token结构
class tokens_struct:
    def __init__(self):
        self.tokens = [' ', '<unk>', 'C', 'O', '(', ')', 'c', '=', '1', '2', 'N', '3', 'n', 'P', '4', '[', ']', 'S',
                       'H', '5', 'l', '-', '*', 'o', '+', '6', '#', 'M', 'F', 'g', '7', 'B', 'r', 's', 'I', 'e', 'i',
                       '8', 'Z']
        self.tokens_length = len(self.tokens)
        self.tokens_vocab = dict(zip(self.tokens, range(len(self.tokens))))
        self.reversed_tokens_vocab = {v: k for k, v in self.tokens_vocab.items()}

    @property
    def unk(self):
        return self.tokens_vocab['<unk>']

    @property
    def pad(self):
        return self.tokens_vocab[' ']

    def encode(self, char_list):
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            if char not in self.tokens:
                smiles_matrix[i] = self.unk
            else:
                smiles_matrix[i] = self.tokens_vocab[char]
        return smiles_matrix


# 蛋白质序列编码 - CNN方式
def protein_encoding(seq):
    # 定义20种氨基酸的独热编码
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_dict = {aa: i for i, aa in enumerate(amino_acids)}

    # 生成蛋白质序列的向量表示
    seq = seq.upper()
    max_seq_len = 3000  # 根据模型配置
    if len(seq) > max_seq_len:
        seq = seq[:max_seq_len]

    # 初始化序列编码矩阵
    encoding = np.zeros((max_seq_len, 20))

    # 填充序列编码
    for i, aa in enumerate(seq):
        if aa in aa_dict:
            encoding[i, aa_dict[aa]] = 1

    return encoding


# 用于RNN的蛋白质序列索引编码
class protein_tokens_struct:
    def __init__(self):
        # 20种常见氨基酸加填充和未知字符
        self.tokens = [' ', '<unk>'] + list("ACDEFGHIKLMNPQRSTVWY")
        self.tokens_length = len(self.tokens)
        self.tokens_vocab = dict(zip(self.tokens, range(len(self.tokens))))
        self.reversed_tokens_vocab = {v: k for k, v in self.tokens_vocab.items()}

    @property
    def unk(self):
        return self.tokens_vocab['<unk>']

    @property
    def pad(self):
        return self.tokens_vocab[' ']

    def encode(self, char_list):
        """将氨基酸序列转换为索引"""
        seq_indices = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            if char not in self.tokens:
                seq_indices[i] = self.unk
            else:
                seq_indices[i] = self.tokens_vocab[char]
        return seq_indices


# 药物分子转换为DGL图
def mol_to_graph(mol):
    # 获取原子特征
    def get_atom_features(atom):
        features = np.zeros(58)

        # 原子类型独热编码
        atom_type = atom.GetSymbol()
        atom_list = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Se', 'Te', 'As', 'Ge']
        if atom_type in atom_list:
            features[atom_list.index(atom_type)] = 1

        # 原子度数
        features[15 + atom.GetDegree()] = 1

        # 原子形式电荷
        features[21 + atom.GetFormalCharge() + 4] = 1

        # 原子杂化类型
        features[30 + atom.GetHybridization().real] = 1

        # 原子是否为芳香性
        features[36] = 1 if atom.GetIsAromatic() else 0

        # 原子是否在环中
        features[37] = 1 if atom.IsInRing() else 0

        # 更多原子特征...
        features[38 + atom.GetImplicitValence()] = 1

        features[43 + atom.GetNumExplicitHs()] = 1

        features[47 + atom.GetNumImplicitHs()] = 1

        features[51] = 1 if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0

        features[52:58] = [
            atom.IsInRingSize(3),
            atom.IsInRingSize(4),
            atom.IsInRingSize(5),
            atom.IsInRingSize(6),
            atom.IsInRingSize(7),
            atom.IsInRingSize(8)
        ]

        return features

    # 获取边特征
    def get_bond_features(bond):
        features = np.zeros(10)

        # 键类型
        bond_type = bond.GetBondType()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            features[0] = 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            features[1] = 1
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            features[2] = 1
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            features[3] = 1

        # 键是否为共轭
        features[4] = 1 if bond.GetIsConjugated() else 0

        # 键是否在环中
        features[5] = 1 if bond.IsInRing() else 0

        # 立体化学
        features[6] = 1 if bond.GetStereo() == Chem.rdchem.BondStereo.STEREOE else 0
        features[7] = 1 if bond.GetStereo() == Chem.rdchem.BondStereo.STEREOZ else 0
        features[8] = 1 if bond.GetStereo() == Chem.rdchem.BondStereo.STEREOCIS else 0
        features[9] = 1 if bond.GetStereo() == Chem.rdchem.BondStereo.STEREOTRANS else 0

        return features

    # 创建DGL图
    g = dgl.graph([])  # 创建空图

    # 添加节点
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)
    node_features = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        node_features.append(get_atom_features(atom))
    node_features_array = np.array(node_features, dtype=np.float32)
    g.ndata['h'] = torch.from_numpy(node_features_array)

    # 添加边
    src_list = []
    dst_list = []
    edge_features = []

    # 添加所有化学键
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        # 添加正向边
        src_list.append(i)
        dst_list.append(j)
        edge_features.append(get_bond_features(bond))

        # 添加反向边
        src_list.append(j)
        dst_list.append(i)
        edge_features.append(get_bond_features(bond))

    g.add_edges(src_list, dst_list)
    edge_features_array = np.array(edge_features, dtype=np.float32)
    g.edata['e'] = torch.from_numpy(edge_features_array)

    return g


# 生成药物的指纹特征
def get_drug_fingerprint(smiles, size=167):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(size)

    # 使用Morgan指纹
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=size)
    return np.array(fp)


# 假设的Node2Vec嵌入生成
# 替换preprocess.py中的generate_dummy_n2v函数

# 替换generate_dummy_n2v函数
def generate_embedding(entity_id, embed_type="n2v", dim=256, all_embeddings=None):
    """
    获取实体的嵌入向量

    参数:
        entity_id (str): 实体ID
        embed_type (str): 嵌入类型
        dim (int): 嵌入维度
        all_embeddings (dict): 所有嵌入向量，如果已有

    返回:
        np.array: 嵌入向量
    """
    return get_embedding(entity_id, embed_type, dim, all_embeddings)


class DrugProteinInteractionDataset(Dataset):
    def __init__(self, drug_ids, drug_smiles, protein_names, protein_seqs, labels,
                 embed_type="n2v", all_embeddings=None):
        self.drug_ids = drug_ids
        self.drug_smiles = drug_smiles
        self.protein_names = protein_names
        self.protein_seqs = protein_seqs
        self.labels = labels
        self.embed_type = embed_type  # 嵌入类型
        self.all_embeddings = all_embeddings  # 嵌入向量字典

        # 初始化token处理器
        self.token_struct = tokens_struct()
        self.protein_token_struct = protein_tokens_struct()

        # 初始化processed_data为空列表
        self.processed_data = []

        # 预处理数据
        self.preprocess()

        # 确保processed_data不为空
        if len(self.processed_data) == 0:
            print("警告: 预处理后没有有效数据，创建一个虚拟数据项")
            # 添加一个虚拟数据项，防止len(dataset)为0
            self._add_dummy_data()

    def _add_dummy_data(self):
        """添加一个虚拟数据项，防止len(dataset)为0"""
        try:
            # 使用第一个数据项的格式创建虚拟数据
            dummy_drug_id = "DUMMY_DRUG"
            dummy_protein_name = "DUMMY_PROTEIN"
            dummy_compound_graph = dgl.graph(([0], [0]))
            dummy_compound_graph.ndata['h'] = torch.zeros((1, 58))
            dummy_compound_graph.edata['e'] = torch.zeros((1, 10))

            # 创建虚拟蛋白质编码
            dummy_protein_cnn = np.zeros((3000, 20))
            dummy_protein_rnn = np.zeros(100)
            dummy_protein_len = 100

            # 创建虚拟SMILES编码
            dummy_smiles = np.zeros(50)
            dummy_smiles_len = 50

            # 创建虚拟嵌入和指纹
            dummy_d_n2v = np.zeros(256)
            dummy_p_n2v = np.zeros(256)
            dummy_drug_fp = np.zeros(167)
            dummy_protein_fp = np.zeros(343)

            # 创建虚拟标签
            dummy_label = 0

            # 添加到processed_data
            self.processed_data.append((
                dummy_drug_id,
                dummy_protein_name,
                dummy_compound_graph,
                dummy_protein_cnn,
                dummy_protein_rnn,
                dummy_protein_len,
                dummy_smiles,
                dummy_smiles_len,
                dummy_d_n2v,
                dummy_p_n2v,
                dummy_drug_fp,
                dummy_protein_fp,
                dummy_label
            ))
            print("已添加虚拟数据项")
        except Exception as e:
            print(f"添加虚拟数据项失败: {e}")
            # 如果添加虚拟数据失败，确保processed_data至少有一个空元素
            self.processed_data = [None]

    def preprocess(self):
        """处理数据并填充processed_data列表"""
        # 记录预处理成功的数据项数量
        success_count = 0

        for i in range(len(self.drug_ids)):
            try:
                # 获取基本数据
                drug_id = self.drug_ids[i]
                protein_name = self.protein_names[i]
                smiles = self.drug_smiles[i]
                seq = self.protein_seqs[i]
                label = self.labels[i]

                # 分子转DGL图
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                compound_graph = mol_to_graph(mol)

                # 蛋白质序列处理 - 同时生成两种编码
                # CNN格式的处理 - 独热编码
                protein_cnn_encoded = protein_encoding(seq)

                # RNN格式的处理 - 索引序列
                protein_rnn_encoded = self.protein_token_struct.encode(seq)
                protein_seq_len = len(protein_rnn_encoded)

                # 生成药物指纹特征
                drug_fp = get_drug_fingerprint(smiles)

                # 蛋白质指纹
                protein_fp = generate_protein_fingerprint(seq, size=343)

                # 获取真实嵌入向量
                d_n2v = generate_embedding(drug_id, self.embed_type, 256, self.all_embeddings)
                p_n2v = generate_embedding(protein_name, self.embed_type, 256, self.all_embeddings)

                # SMILES编码
                encoded_smiles = self.token_struct.encode(smiles)
                smiles_seq_len = len(encoded_smiles)

                # 存储处理后的数据
                self.processed_data.append((
                    drug_id,
                    protein_name,
                    compound_graph,
                    protein_cnn_encoded,
                    protein_rnn_encoded,
                    protein_seq_len,
                    encoded_smiles,
                    smiles_seq_len,
                    d_n2v,
                    p_n2v,
                    drug_fp,
                    protein_fp,
                    label
                ))

                success_count += 1
                if success_count % 1000 == 0:
                    print(f"已处理 {success_count} 条数据")

            except Exception as e:
                print(f"处理数据 {i} 时出错: {e}")
                continue

        print(f"预处理完成，成功处理{success_count}条数据，总数据{len(self.drug_ids)}条")

    def __len__(self):
        """返回数据集的长度"""
        # 确保即使processed_data为None也能返回合理的值
        if self.processed_data is None:
            return 0
        return len(self.processed_data)

    def __getitem__(self, idx):
        """返回指定索引的数据项"""
        if idx >= len(self.processed_data):
            raise IndexError(f"索引{idx}超出范围，数据集长度为{len(self.processed_data)}")

        try:
            (drug_id, protein_name, compound_graph, protein_cnn, protein_rnn, protein_len,
             smiles, smiles_len, d_n2v, p_n2v, drug_fp, protein_fp, label) = self.processed_data[idx]

            # 确保返回的是tensor类型
            if not isinstance(label, torch.Tensor):
                label = torch.FloatTensor([label])

            return (drug_id, protein_name, compound_graph, protein_cnn, protein_rnn, protein_len,
                    smiles, smiles_len, d_n2v, p_n2v, drug_fp, protein_fp, label)
        except Exception as e:
            print(f"获取索引{idx}的数据项时出错: {e}")
            # 如果出现错误，返回第一个项（可能是虚拟数据项）
            if len(self.processed_data) > 0:
                return self.processed_data[0]
            else:
                raise RuntimeError("数据集为空，无法返回任何数据项")

# 创建对应的collate_molgraphs函数
def collate_molgraphs(data):
    """
    统一的数据收集函数，适应不同长度的数据格式
    """
    import dgl
    import torch
    from torch.nn.utils.rnn import pad_sequence

    # 检查数据是否为空
    if len(data) == 0:
        raise ValueError("Empty batch data")

    # 数据解包 - 我们假设数据是13元素的元组
    # (drug_id, protein_name, compound_graph, protein_cnn, protein_rnn, protein_len,
    #  smiles, smiles_len, d_n2v, p_n2v, drug_fp, protein_fp, label)
    _, _, compounds, protein_cnn, protein_rnn, protein_lens, smiles, seq_lens, d_n2v, p_n2v, f_d, f_p, actions = map(
        list, zip(*data))

    # 处理图数据
    bg = dgl.batch(compounds)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    # 标签转换为张量
    actions = torch.stack([act.clone().detach() for act in actions])

    # 蛋白质数据处理 - 使用protein_rnn编码
    padded_proteins = pad_sequence([torch.FloatTensor(p).clone().detach() if not isinstance(p, torch.Tensor)
                                    else p.clone().detach() for p in protein_rnn], batch_first=True)

    # 处理其他特征
    d_n2v = torch.stack(
        [i.clone().detach().float() if isinstance(i, torch.Tensor) else torch.tensor(i, dtype=torch.float) for i in
         d_n2v], dim=0)
    p_n2v = torch.stack(
        [i.clone().detach().float() if isinstance(i, torch.Tensor) else torch.tensor(i, dtype=torch.float) for i in
         p_n2v], dim=0)
    f_d = torch.stack(
        [i.clone().detach().float() if isinstance(i, torch.Tensor) else torch.tensor(i, dtype=torch.float) for i in
         f_d], dim=0)
    f_p = torch.stack(
        [i.clone().detach().float() if isinstance(i, torch.Tensor) else torch.tensor(i, dtype=torch.float) for i in
         f_p], dim=0)

    # 填充SMILES序列
    padded_smiles = pad_sequence([torch.FloatTensor(s).clone().detach() if not isinstance(s, torch.Tensor)
                                  else s.clone().detach() for s in smiles], batch_first=True)

    # 返回统一格式 - 10元素元组
    return bg, padded_proteins, protein_lens, padded_smiles, seq_lens, d_n2v, p_n2v, f_d, f_p, actions


# 主要预处理函数
def preprocess_data(csv_file, save_path, embed_type="n2v", all_embeddings=None):
    """
    预处理数据，包括读取CSV文件、创建数据集和分割数据

    参数:
        csv_file (str): CSV文件的路径
        save_path (str): 保存处理后数据的路径
        embed_type (str): 嵌入类型，可选值有"n2v", "deepwalk", "line", "sdne"
        all_embeddings (dict): 所有嵌入向量，如果已有

    返回:
        tuple: 包含训练集、验证集和测试集的元组
    """
    print("开始读取CSV文件...")
    df = pd.read_csv(csv_file)
    print(f"CSV文件读取完成，共有 {len(df)} 条数据")

    # 提取数据
    drug_ids = df['drugbank_id'].values
    drug_smiles = df['SMILES'].values
    protein_names = df['gene_name'].values
    protein_seqs = df['Sequence'].values
    labels = df['label'].values

    print(f"创建数据集，使用嵌入类型: {embed_type}...")
    dataset = DrugProteinInteractionDataset(
        drug_ids, drug_smiles, protein_names, protein_seqs, labels,
        embed_type, all_embeddings)

    # 检查数据集是否为空
    if len(dataset) == 0:
        raise ValueError("数据集为空，无法继续处理")

    print(f"数据集创建完成，包含{len(dataset)}个数据项")

    # 分割数据集为训练、验证和测试集 - 使用8:1:1比例
    try:
        indices = list(range(len(dataset)))
        print(f"准备分割数据集，总索引数: {len(indices)}")

        # 首先分割出10%的测试集
        train_val_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)
        print(f"第一次分割完成: 训练+验证集 {len(train_val_indices)}个, 测试集 {len(test_indices)}个")

        # 然后从剩余的90%中分割出1/9（相当于整体的10%）作为验证集
        train_indices, val_indices = train_test_split(train_val_indices, test_size=1 / 9, random_state=42)
        print(f"第二次分割完成: 训练集 {len(train_indices)}个, 验证集 {len(val_indices)}个")

        # 创建数据子集
        print("创建训练子集...")
        train_subset = [dataset[i] for i in train_indices]
        print("创建验证子集...")
        val_subset = [dataset[i] for i in val_indices]
        print("创建测试子集...")
        test_subset = [dataset[i] for i in test_indices]

        print(f"训练集大小: {len(train_subset)}")
        print(f"验证集大小: {len(val_subset)}")
        print(f"测试集大小: {len(test_subset)}")

        # 保存处理后的数据
        print("保存处理后的数据...")
        # 保存嵌入类型信息以便后续加载时检查
        dump(save_path, (train_subset, val_subset, test_subset, embed_type))
        print(f"数据已保存到 {save_path}")

        return train_subset, val_subset, test_subset

    except Exception as e:
        print(f"分割数据集时出错: {e}")
        # 如果分割失败，创建一个最小的分割
        # 可能数据集太小或数据结构有问题
        print("尝试创建简单分割...")
        dataset_len = len(dataset)
        train_size = max(int(dataset_len * 0.8), 1)
        val_size = max(int(dataset_len * 0.1), 1)
        test_size = dataset_len - train_size - val_size

        # 确保测试集不为空
        if test_size <= 0:
            test_size = 1
            val_size = max(1, val_size)
            train_size = dataset_len - val_size - test_size

        print(f"简单分割: 训练集 {train_size}个, 验证集 {val_size}个, 测试集 {test_size}个")

        # 直接根据索引分割
        train_subset = [dataset[i] for i in range(train_size)]
        val_subset = [dataset[i] for i in range(train_size, train_size + val_size)]
        test_subset = [dataset[i] for i in range(train_size + val_size, dataset_len)]

        # 保存处理后的数据
        dump(save_path, (train_subset, val_subset, test_subset, embed_type))
        print(f"简单分割数据已保存到 {save_path}")

        return train_subset, val_subset, test_subset


# 执行预处理
if __name__ == "__main__":
    csv_file = "Dataset-of-activating-and-inhibiting-mechanisms.csv"
    save_path = "dataset_train_vali_test_deepWalk_with_protein_fingerprint.csv"

    print("开始数据预处理...")
    dataset_train, dataset_vali, dataset_test = preprocess_data(csv_file, save_path)
    print("数据预处理完成!")