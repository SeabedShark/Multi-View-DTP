import numpy as np
from collections import Counter
import re

def calculate_aa_composition(sequence):
    """计算氨基酸组成比例特征"""
    aa_list = "ACDEFGHIKLMNPQRSTVWY"
    aa_count = Counter(sequence)
    composition = np.zeros(20)

    seq_len = len(sequence)
    if seq_len == 0:  # 防止除零
        return composition

    for i, aa in enumerate(aa_list):
        composition[i] = aa_count.get(aa, 0) / seq_len

    return composition

def calculate_dipeptide_composition(sequence):
    """计算二肽组成特征

    参数:
        sequence (str): 蛋白质序列

    返回:
        np.array: 400个二肽组合的组成特征
    """
    aa_list = "ACDEFGHIKLMNPQRSTVWY"
    dipeptides = [a+b for a in aa_list for b in aa_list]

    # 创建二肽到索引的映射
    dipep_to_idx = {dipep: i for i, dipep in enumerate(dipeptides)}

    # 初始化特征向量
    composition = np.zeros(400)

    # 计算每个二肽出现的次数
    for i in range(len(sequence)-1):
        dipep = sequence[i:i+2]
        if all(aa in aa_list for aa in dipep):  # 确保二肽中的氨基酸是标准的
            composition[dipep_to_idx[dipep]] += 1

    # 归一化
    total_count = sum(composition)
    if total_count > 0:
        composition = composition / total_count

    return composition

def calculate_ctd_composition(sequence):
    """计算CTD (Composition, Transition, Distribution)特征的组成部分

    参数:
        sequence (str): 蛋白质序列

    返回:
        np.array: CTD组成特征
    """
    # 根据物理化学性质对氨基酸分组
    groups = {
        'hydrophobicity': {
            'group1': 'RKEDQN',  # 亲水性
            'group2': 'GASTPHY',  # 中性
            'group3': 'CLVIMFW'   # 疏水性
        },
        'normalized_vdw_volume': {
            'group1': 'GASTPD',   # 小体积
            'group2': 'NVEQIL',   # 中等体积
            'group3': 'MHKFRYW'   # 大体积
        },
        'polarity': {
            'group1': 'LIFWCMVY',  # 非极性
            'group2': 'PATGS',     # 极性中性
            'group3': 'HQRKNED'    # 极性
        },
        'charge': {
            'group1': 'KR',        # 正电荷
            'group2': 'ANCQGHILMFPSTWYV', # 中性
            'group3': 'DE'         # 负电荷
        },
        'secondary_structure': {
            'group1': 'EALMQKRH', # 螺旋（Helix）偏好
            'group2': 'VIYCWFT',  # 片层（Sheet）偏好
            'group3': 'GNPSD'     # 转角（Turn）偏好
        }
    }

    features = []
    seq_len = len(sequence)
    if seq_len == 0:  # 防止除零
        return np.zeros(15)  # 返回全零CTD特征

    # 计算各组成比例
    for prop, group in groups.items():
        g1_count = sum(1 for aa in sequence if aa in group['group1'])
        g2_count = sum(1 for aa in sequence if aa in group['group2'])
        g3_count = sum(1 for aa in sequence if aa in group['group3'])

        # 归一化
        features.extend([g1_count/seq_len, g2_count/seq_len, g3_count/seq_len])

    return np.array(features)

def calculate_pseudo_aac(sequence, lamda=30, weight=0.05):
    """计算伪氨基酸组成(PseAAC)特征
    
    参数:
        sequence (str): 蛋白质序列
        lamda (int): 序列关联因子最大滞后阶数
        weight (float): 权重因子
        
    返回:
        np.array: PseAAC特征
    """
    aa_list = "ACDEFGHIKLMNPQRSTVWY"
    aa_dict = {aa: i for i, aa in enumerate(aa_list)}
    
    # 氨基酸的物理化学性质矩阵(H1,H2,M,P1,P2,SASA,NCI,DCI)
    # H1: 疏水性[1], H2: 疏水性[2], M: 侧链质量, P1: 极性[1],
    # P2: 极性[2], SASA: 溶剂可及表面积, NCI: 正电荷指数, DCI: 疏水性分布
    aa_property = {
        'A': [0.62, -0.5, 15, 0.0, 0.0, 1.8, 0.0, 0.25],
        'C': [0.29, -1.0, 47, 1.0, 0.0, 2.5, 0.0, 0.04],
        'D': [-0.9, 3.0, 59, -1.0, 0.0, 3.0, -1.0, 0.46],
        'E': [-0.74, 3.0, 73, -1.0, 0.0, 3.0, -1.0, 0.91],
        'F': [1.19, -2.5, 91, 1.0, 0.0, 2.8, 0.0, 0.09],
        'G': [0.48, 0.0, 1, 0.0, 0.0, 0.0, 0.0, 1.0],
        'H': [-0.4, -0.5, 82, 0.0, 1.0, 3.0, 0.0, 0.47],
        'I': [1.38, -1.8, 57, 0.0, 0.0, 3.1, 0.0, 0.00],
        'K': [-1.5, 3.0, 73, 0.0, 0.0, 3.0, 1.0, 1.0],
        'L': [1.06, -1.8, 57, 0.0, 0.0, 2.8, 0.0, 0.15],
        'M': [0.64, -1.3, 75, 0.0, 0.0, 3.4, 0.0, 0.15],
        'N': [-0.78, 2.0, 58, 0.0, 0.0, 3.5, 0.0, 0.90],
        'P': [0.12, 0.0, 42, 0.0, 0.0, 2.7, 0.0, 0.25],
        'Q': [-0.85, 0.2, 72, 0.0, 0.0, 3.5, 0.0, 0.92],
        'R': [-2.53, 3.0, 101, 0.0, 1.0, 4.1, 1.0, 0.70],
        'S': [-0.18, 0.3, 31, 0.0, 0.0, 1.6, 0.0, 0.63],
        'T': [-0.05, -0.4, 45, 0.0, 0.0, 1.5, 0.0, 0.63],
        'V': [1.08, -1.5, 43, 0.0, 0.0, 2.6, 0.0, 0.21],
        'W': [0.81, -3.4, 130, 1.0, 0.0, 3.4, 0.0, 0.01],
        'Y': [0.26, -2.3, 107, 1.0, 0.0, 2.9, 0.0, 0.17]
    }
    
    # 标准化性质值
    properties = []
    for i in range(8):  # 8种物理化学性质
        prop_values = [aa_property[aa][i] for aa in aa_list]
        mean = np.mean(prop_values)
        std = np.std(prop_values)
        normalized_values = {aa: (aa_property[aa][i] - mean) / std for aa in aa_list}
        properties.append(normalized_values)
    
    # 计算氨基酸组成
    aa_count = Counter(sequence)
    composition = np.zeros(20)
    for i, aa in enumerate(aa_list):
        composition[i] = aa_count.get(aa, 0) / len(sequence)
    
    # 计算序列相关因子
    correlation_factor = np.zeros(lamda * 8)  # 8种性质，每种有lamda个序列关联因子
    
    for prop_idx in range(8):
        prop_dict = properties[prop_idx]
        for lag in range(1, lamda + 1):
            corr = 0.0
            for i in range(len(sequence) - lag):
                aa1 = sequence[i]
                aa2 = sequence[i + lag]
                if aa1 in prop_dict and aa2 in prop_dict:
                    corr += prop_dict[aa1] * prop_dict[aa2]
            correlation_factor[(prop_idx * lamda) + lag - 1] = corr / (len(sequence) - lag)
    
    # 组合特征
    denominator = 1 + weight * sum(correlation_factor)
    pse_aac = np.zeros(20 + lamda * 8)
    
    # 标准氨基酸组成部分
    for i in range(20):
        pse_aac[i] = composition[i] / denominator
    
    # 序列关联因子部分
    for j in range(lamda * 8):
        pse_aac[20 + j] = weight * correlation_factor[j] / denominator
    
    return pse_aac


# 修改 generate_protein_fingerprint 函数
def generate_protein_fingerprint(sequence, size=343):
    """生成蛋白质指纹特征

    参数:
        sequence (str): 蛋白质序列
        size (int): 输出特征向量的长度

    返回:
        np.array: 蛋白质指纹特征
    """
    # 预处理序列 - 去除非标准氨基酸
    if sequence is None or not isinstance(sequence, str):
        print(f"警告: 无效序列类型: {type(sequence)}")
        return np.zeros(size)

    sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())

    # 如果序列为空或者太短，返回零向量
    if not sequence or len(sequence) < 5:
        print(f"警告: 序列太短或为空: '{sequence}'")
        return np.zeros(size)

    try:
        # 计算各种特征，添加异常处理
        aa_comp = calculate_aa_composition(sequence)  # 20维
        dipep_comp = calculate_dipeptide_composition(sequence)  # 400维
        ctd_comp = calculate_ctd_composition(sequence)  # 15维

        try:
            pse_aac = calculate_pseudo_aac(sequence, lamda=min(30, len(sequence) - 1), weight=0.05)  # 260维
        except Exception as e:
            print(f"计算伪氨基酸组成出错: {e}")
            # 使用零向量代替
            pse_aac = np.zeros(260)

            # 合并特征
        features = np.concatenate([aa_comp, ctd_comp, pse_aac])

    except Exception as e:
        print(f"生成蛋白质指纹时出错: {e}")
        return np.zeros(size)

    # 对特征进行降维或填充，确保输出指定大小
    if len(features) > size:
        # 简单裁剪而非PCA，避免额外计算
        features = features[:size]
    elif len(features) < size:
        # 使用零填充扩展到指定大小
        padding = np.zeros(size - len(features))
        features = np.concatenate([features, padding])

    return features
