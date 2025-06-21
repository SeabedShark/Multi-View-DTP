
import os
import numpy as np
import networkx as nx
import torch
import pickle
from tqdm import tqdm
import pandas as pd
import warnings

# 抑制不必要的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# 检查目录是否存在，如果不存在则创建
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# 加载嵌入向量
def load_embeddings(embed_type, embed_dim=256, embeddings_dir="./embeddings"):
    """
    加载预训练的嵌入向量

    参数:
        embed_type (str): 嵌入类型
        embed_dim (int): 嵌入维度
        embeddings_dir (str): 嵌入向量保存目录

    返回:
        dict: 实体ID到嵌入向量的映射
    """
    ensure_dir(embeddings_dir)
    embed_file = os.path.join(embeddings_dir, f"{embed_type}_embeddings_{embed_dim}.pkl")

    if not os.path.exists(embed_file):
        print(f"嵌入文件 {embed_file} 不存在！将使用随机嵌入。")
        return {}

    try:
        with open(embed_file, "rb") as f:
            embeddings = pickle.load(f)
        print(f"成功加载{len(embeddings)}个{embed_type}嵌入向量。")
        return embeddings
    except Exception as e:
        print(f"加载嵌入向量时出错: {e}")
        return {}


# 保存嵌入向量
def save_embeddings(embeddings, embed_type, embed_dim=256, embeddings_dir="./embeddings"):
    """
    保存嵌入向量

    参数:
        embeddings (dict): 实体ID到嵌入向量的映射
        embed_type (str): 嵌入类型
        embed_dim (int): 嵌入维度
        embeddings_dir (str): 嵌入向量保存目录
    """
    ensure_dir(embeddings_dir)
    embed_file = os.path.join(embeddings_dir, f"{embed_type}_embeddings_{embed_dim}.pkl")

    try:
        with open(embed_file, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"成功保存{len(embeddings)}个{embed_type}嵌入向量。")
    except Exception as e:
        print(f"保存嵌入向量时出错: {e}")


# 构建药物-蛋白质交互网络
def build_network(csv_file, directed=False):
    """
    从CSV文件构建药物-蛋白质交互网络

    参数:
        csv_file (str): CSV文件路径
        directed (bool): 是否为有向图

    返回:
        nx.Graph or nx.DiGraph: 药物-蛋白质交互网络
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    print(f"从{csv_file}构建网络...")

    # 创建图
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # 添加药物和蛋白质节点
    drug_ids = df['drugbank_id'].unique()
    protein_names = df['gene_name'].unique()

    # 添加节点
    for drug_id in drug_ids:
        G.add_node(drug_id, node_type='drug')

    for protein_name in protein_names:
        G.add_node(protein_name, node_type='protein')

    # 添加边
    edge_count = 0
    for _, row in df.iterrows():
        drug_id = row['drugbank_id']
        protein_name = row['gene_name']
        label = row['label']

        # 添加边，使用标签作为边属性
        G.add_edge(drug_id, protein_name, weight=float(label))
        edge_count += 1

    print(
        f"网络构建完成: {len(G.nodes())}个节点 ({len(drug_ids)}个药物, {len(protein_names)}个蛋白质), {edge_count}条边")
    return G


# 将NetworkX图转换为药物-蛋白质二部图
def to_bipartite(G):
    """
    将一般图转换为二部图格式

    参数:
        G (nx.Graph): 输入图

    返回:
        nx.Graph: 二部图
    """
    B = nx.Graph()

    # 添加节点，保持原始属性
    for node, attrs in G.nodes(data=True):
        B.add_node(node, **attrs)

    # 添加边，只保留药物-蛋白质边
    for u, v, attrs in G.edges(data=True):
        u_type = G.nodes[u].get('node_type')
        v_type = G.nodes[v].get('node_type')

        if u_type != v_type:  # 不同类型的节点之间的边
            B.add_edge(u, v, **attrs)

    return B


# 增强图的连接性以改善嵌入效果
def enhance_graph(G, k_neighbors=5):
    """
    增强图的连接性以改善嵌入效果

    参数:
        G (nx.Graph): 输入图
        k_neighbors (int): 同类节点中要连接的邻居数

    返回:
        nx.Graph: 增强后的图
    """
    # 创建增强图
    G_enhanced = G.copy()

    # 将节点按类型分组
    drugs = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'drug']
    proteins = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'protein']

    # 基于共同邻居为药物-药物添加边
    for i, drug1 in enumerate(drugs):
        drug_similarities = []

        for drug2 in drugs:
            if drug1 != drug2:
                # 计算共同邻居数量作为相似度
                common_neighbors = list(nx.common_neighbors(G, drug1, drug2))
                similarity = len(common_neighbors)

                drug_similarities.append((drug2, similarity))

        # 排序并连接最相似的k个药物
        drug_similarities.sort(key=lambda x: x[1], reverse=True)
        for drug2, sim in drug_similarities[:k_neighbors]:
            if sim > 0 and not G_enhanced.has_edge(drug1, drug2):
                G_enhanced.add_edge(drug1, drug2, weight=sim / 10)  # 较弱的权重

    # 基于共同邻居为蛋白质-蛋白质添加边
    for i, prot1 in enumerate(proteins):
        prot_similarities = []

        for prot2 in proteins:
            if prot1 != prot2:
                # 计算共同邻居数量作为相似度
                common_neighbors = list(nx.common_neighbors(G, prot1, prot2))
                similarity = len(common_neighbors)

                prot_similarities.append((prot2, similarity))

        # 排序并连接最相似的k个蛋白质
        prot_similarities.sort(key=lambda x: x[1], reverse=True)
        for prot2, sim in prot_similarities[:k_neighbors]:
            if sim > 0 and not G_enhanced.has_edge(prot1, prot2):
                G_enhanced.add_edge(prot1, prot2, weight=sim / 10)  # 较弱的权重

    print(f"增强图连接性: 原始图 {len(G.edges())}条边 -> 增强图 {len(G_enhanced.edges())}条边")
    return G_enhanced


# Node2Vec实现
def train_node2vec(G, dimensions=256, walk_length=80, num_walks=10, p=1, q=1, workers=4, window=10, min_count=1):
    """
    训练Node2Vec模型

    参数:
        G (nx.Graph): 输入图
        dimensions (int): 嵌入维度
        其他参数: Node2Vec参数

    返回:
        dict: 节点到嵌入向量的映射
    """
    try:
        from node2vec import Node2Vec

        # 确保图中的节点标签是字符串
        G_str = nx.Graph()
        for node, attrs in G.nodes(data=True):
            G_str.add_node(str(node), **attrs)

        for u, v, attrs in G.edges(data=True):
            G_str.add_edge(str(u), str(v), **attrs)

        # 创建Node2Vec模型
        node2vec = Node2Vec(G_str, dimensions=dimensions, walk_length=walk_length,
                            num_walks=num_walks, p=p, q=q, workers=workers)

        # 训练模型
        print("开始训练Node2Vec模型...")
        model = node2vec.fit(window=window, min_count=min_count, batch_words=4)

        # 获取嵌入向量
        embeddings = {}
        for node in tqdm(G.nodes(), desc="获取Node2Vec嵌入向量"):
            try:
                embeddings[node] = model.wv.get_vector(str(node))
            except KeyError:
                # 如果节点不在模型中，创建随机向量
                embeddings[node] = np.random.normal(0, 1, dimensions)

        print(f"Node2Vec训练完成，获取了{len(embeddings)}个嵌入向量。")
        return embeddings
    except ImportError:
        print("无法导入node2vec。请安装: pip install node2vec")
        return generate_random_embeddings(G, dimensions)


# 在train_deepwalk函数中添加备选方案
def train_deepwalk(G, dimensions=256, walk_length=80, num_walks=10, window=10, workers=4):
    """
    训练DeepWalk模型

    参数:
        G (nx.Graph): 输入图
        dimensions (int): 嵌入维度
        walk_length (int): 随机游走的长度
        num_walks (int): 每个节点的随机游走数量
        window (int): word2vec的窗口大小
        workers (int): 并行工作器数量

    返回:
        dict: 节点到嵌入向量的映射
    """
    try:
        # 尝试导入karateclub
        try:
            from karateclub import DeepWalk

            # 将图节点转换为整数索引
            node_list = list(G.nodes())
            node_mapping = {node: i for i, node in enumerate(node_list)}
            reverse_mapping = {i: node for node, i in node_mapping.items()}

            # 创建映射后的图
            G_indexed = nx.Graph()
            for edge in G.edges():
                G_indexed.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])

            # 创建DeepWalk模型 - 使用正确的参数名称
            deepwalk = DeepWalk(dimensions=dimensions,
                                walk_length=walk_length,
                                window_size=window,
                                workers=workers,
                                walk_number=num_walks)  # 修正的参数名

            # 训练模型
            print("开始训练DeepWalk模型...")
            deepwalk.fit(G_indexed)

            # 获取嵌入向量
            embedding_matrix = deepwalk.get_embedding()

            # 将嵌入矩阵转换回原始节点ID
            embeddings = {}
            for i, node in reverse_mapping.items():
                embeddings[node] = embedding_matrix[i]

            print(f"DeepWalk训练完成，获取了{len(embeddings)}个嵌入向量。")
            return embeddings

        except ImportError:
            # 尝试使用node2vec作为备选
            raise ImportError("无法导入karateclub，尝试使用node2vec作为DeepWalk替代...")

    except ImportError:
        # 使用node2vec代替DeepWalk（它们非常相似，只是参数不同）
        try:
            from node2vec import Node2Vec

            print("使用Node2Vec (p=1, q=1) 作为DeepWalk的替代...")

            # 确保图中的节点标签是字符串
            G_str = nx.Graph()
            for node, attrs in G.nodes(data=True):
                G_str.add_node(str(node), **attrs)

            for u, v, attrs in G.edges(data=True):
                G_str.add_edge(str(u), str(v), **attrs)

            # DeepWalk相当于Node2Vec，其中p=1, q=1
            node2vec = Node2Vec(G_str, dimensions=dimensions, walk_length=walk_length,
                                num_walks=num_walks, p=1, q=1, workers=workers)

            # 训练模型
            print("开始训练替代的DeepWalk (Node2Vec p=1, q=1) 模型...")
            model = node2vec.fit(window=window, min_count=1, batch_words=4)

            # 获取嵌入向量
            embeddings = {}
            for node in tqdm(G.nodes(), desc="获取DeepWalk嵌入向量"):
                try:
                    embeddings[node] = model.wv.get_vector(str(node))
                except KeyError:
                    # 如果节点不在模型中，创建随机向量
                    embeddings[node] = np.random.normal(0, 1, dimensions)

            print(f"替代的DeepWalk训练完成，获取了{len(embeddings)}个嵌入向量。")
            return embeddings

        except ImportError:
            print("无法导入node2vec和karateclub。请安装: pip install node2vec or pip install karateclub")
            return generate_random_embeddings(G, dimensions)

    except Exception as e:
        print(f"DeepWalk训练失败: {e}")
        print("回退到随机嵌入。")
        return generate_random_embeddings(G, dimensions)


# 修改LINE实现，添加备选方案
def train_line(G, dimensions=256, order=1, negative_ratio=5, epochs=100, learning_rate=0.025):
    """
    训练LINE模型

    参数:
        G (nx.Graph): 输入图
        dimensions (int): 嵌入维度
        order (int): LINE的阶数 (1: 一阶邻近度, 2: 二阶邻近度)
        其他参数: LINE参数

    返回:
        dict: 节点到嵌入向量的映射
    """
    try:
        try:
            from karateclub import LINE

            # 将图节点转换为整数索引
            node_list = list(G.nodes())
            node_mapping = {node: i for i, node in enumerate(node_list)}
            reverse_mapping = {i: node for node, i in node_mapping.items()}

            # 创建映射后的图
            G_indexed = nx.Graph()
            for edge in G.edges():
                G_indexed.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])

            # 创建LINE模型
            line = LINE(dimensions=dimensions, order=order, negative_ratio=negative_ratio,
                        epochs=epochs, learning_rate=learning_rate)

            # 训练模型
            print("开始训练LINE模型...")
            line.fit(G_indexed)

            # 获取嵌入向量
            embedding_matrix = line.get_embedding()

            # 将嵌入矩阵转换回原始节点ID
            embeddings = {}
            for i, node in reverse_mapping.items():
                embeddings[node] = embedding_matrix[i]

            print(f"LINE训练完成，获取了{len(embeddings)}个嵌入向量。")
            return embeddings

        except ImportError:
            # 如果没有karateclub，尝试自定义的PyTorch实现
            raise ImportError("尝试使用PyTorch实现LINE...")

    except ImportError:
        # 使用PyTorch实现简化版的LINE
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, Dataset

            print("使用自定义PyTorch实现的LINE...")

            # 定义LINE模型和训练逻辑
            class LineDataset(Dataset):
                def __init__(self, G, negative_samples=5):
                    self.G = G
                    self.nodes = list(G.nodes())
                    self.node_map = {node: i for i, node in enumerate(self.nodes)}
                    self.edges = list(G.edges())
                    self.neg_samples = negative_samples

                def __len__(self):
                    return len(self.edges)

                def __getitem__(self, idx):
                    # 正样本
                    u, v = self.edges[idx]
                    u_idx, v_idx = self.node_map[u], self.node_map[v]

                    # 负样本
                    neg_v_idxs = []
                    for _ in range(self.neg_samples):
                        neg_v = np.random.choice(self.nodes)
                        while self.G.has_edge(u, neg_v):
                            neg_v = np.random.choice(self.nodes)
                        neg_v_idxs.append(self.node_map[neg_v])

                    return u_idx, v_idx, neg_v_idxs

            class LINE(nn.Module):
                def __init__(self, n_nodes, embedding_dim):
                    super(LINE, self).__init__()
                    self.embedding = nn.Embedding(n_nodes, embedding_dim)
                    self.context_embedding = nn.Embedding(n_nodes, embedding_dim)

                    # 初始化
                    nn.init.xavier_uniform_(self.embedding.weight)
                    nn.init.xavier_uniform_(self.context_embedding.weight)

                def forward(self, u_idx, v_idx, neg_v_idxs):
                    # 获取嵌入向量
                    u = self.embedding(u_idx)
                    v = self.context_embedding(v_idx)

                    # 计算正样本分数
                    pos_score = torch.sum(u * v, dim=1)
                    pos_loss = -torch.mean(nn.functional.logsigmoid(pos_score))

                    # 计算负样本分数
                    neg_loss = 0
                    for neg_idx in range(len(neg_v_idxs[0])):
                        neg_v = self.context_embedding(torch.tensor([neg[neg_idx] for neg in neg_v_idxs]))
                        neg_score = torch.sum(u * neg_v, dim=1)
                        neg_loss += -torch.mean(nn.functional.logsigmoid(-neg_score))

                    return pos_loss + neg_loss / len(neg_v_idxs[0])

                def get_embeddings(self):
                    return self.embedding.weight.detach().cpu().numpy()

            # 创建数据集和数据加载器
            dataset = LineDataset(G, negative_samples=negative_ratio)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

            # 创建模型
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = LINE(len(G.nodes()), dimensions).to(device)

            # 定义优化器
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # 训练模型
            print("开始训练PyTorch版LINE模型...")
            model.train()

            for epoch in range(1, epochs + 1):
                total_loss = 0
                for batch in dataloader:
                    u_idx, v_idx, neg_v_idxs = [item.to(device) for item in batch]

                    # 计算损失
                    loss = model(u_idx, v_idx, neg_v_idxs)

                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                if epoch % max(1, epochs // 10) == 0:
                    print(f"Epoch {epoch}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

            # 获取嵌入向量
            embedding_matrix = model.get_embeddings()

            # 将嵌入向量映射回原始节点
            embeddings = {}
            for i, node in enumerate(G.nodes()):
                embeddings[node] = embedding_matrix[i]

            print(f"PyTorch版LINE训练完成，获取了{len(embeddings)}个嵌入向量。")
            return embeddings

        except Exception as e:
            print(f"PyTorch版LINE训练失败: {e}")
            return generate_random_embeddings(G, dimensions)

    except Exception as e:
        print(f"LINE训练失败: {e}")
        print("回退到随机嵌入。")
        return generate_random_embeddings(G, dimensions)


# 使用PyTorch实现的SDNE模型
def train_sdne(G, dimensions=256, hidden_layers=[512, 256], epochs=10, batch_size=32, learning_rate=0.01):
    """
    使用PyTorch训练SDNE模型

    参数:
        G (nx.Graph): 输入图
        dimensions (int): 嵌入维度
        hidden_layers (list): 隐藏层大小
        epochs (int): 训练轮数
        batch_size (int): 批次大小
        learning_rate (float): 学习率

    返回:
        dict: 节点到嵌入向量的映射
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        # 将图转换为邻接矩阵
        node_list = list(G.nodes())
        node_mapping = {node: i for i, node in enumerate(node_list)}
        n = len(node_list)

        # 创建邻接矩阵
        adj_matrix = np.zeros((n, n), dtype=np.float32)
        for edge in G.edges(data=True):
            i, j = node_mapping[edge[0]], node_mapping[edge[1]]
            weight = edge[2].get('weight', 1.0)
            adj_matrix[i, j] = weight
            adj_matrix[j, i] = weight  # 对于无向图

        # 定义SDNE自编码器模型
        class SDNE(nn.Module):
            def __init__(self, input_dim, hidden_dims, embedding_dim):
                super(SDNE, self).__init__()

                # 构建编码器
                encoder_layers = []
                in_dim = input_dim

                for hidden_dim in hidden_dims:
                    encoder_layers.append(nn.Linear(in_dim, hidden_dim))
                    encoder_layers.append(nn.ReLU())
                    in_dim = hidden_dim

                encoder_layers.append(nn.Linear(in_dim, embedding_dim))
                self.encoder = nn.Sequential(*encoder_layers)

                # 构建解码器
                decoder_layers = []
                in_dim = embedding_dim

                for hidden_dim in reversed(hidden_dims):
                    decoder_layers.append(nn.Linear(in_dim, hidden_dim))
                    decoder_layers.append(nn.ReLU())
                    in_dim = hidden_dim

                decoder_layers.append(nn.Linear(in_dim, input_dim))
                decoder_layers.append(nn.Sigmoid())
                self.decoder = nn.Sequential(*decoder_layers)

            def forward(self, x):
                # 编码
                embedding = self.encoder(x)
                # 解码
                reconstructed = self.decoder(embedding)
                return embedding, reconstructed

            def get_embedding(self, x):
                return self.encoder(x)

        # 创建数据集和数据加载器
        adj_tensor = torch.FloatTensor(adj_matrix)
        dataset = TensorDataset(adj_tensor, adj_tensor)
        dataloader = DataLoader(dataset, batch_size=min(batch_size, n), shuffle=True)

        # 初始化模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SDNE(n, hidden_layers, dimensions).to(device)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练模型
        print("开始训练SDNE模型...")
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                _, reconstructed = model(batch_x)

                # 计算损失
                loss = criterion(reconstructed, batch_y)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # 打印每个epoch的损失
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

        # 获取嵌入向量
        model.eval()
        with torch.no_grad():
            embeddings = {}
            embedding_matrix = model.get_embedding(adj_tensor.to(device)).cpu().numpy()

            for i, node in enumerate(node_list):
                embeddings[node] = embedding_matrix[i]

        print(f"SDNE训练完成，获取了{len(embeddings)}个嵌入向量。")
        return embeddings
    except Exception as e:
        print(f"SDNE训练失败: {e}")
        print("回退到随机嵌入。")
        return generate_random_embeddings(G, dimensions)


# 生成随机嵌入（fallback方法）
def generate_random_embeddings(G, dimensions=256):
    """
    为图中的节点生成随机嵌入向量

    参数:
        G (nx.Graph): 输入图
        dimensions (int): 嵌入维度

    返回:
        dict: 节点到嵌入向量的映射
    """
    print("使用随机嵌入作为后备方法。")
    embeddings = {}
    for node in G.nodes():
        # 为每个节点生成随机向量
        np.random.seed(hash(str(node)) % 10000)
        embeddings[node] = np.random.normal(0, 1, dimensions)

    return embeddings


# 修改训练所有嵌入的函数以添加更多错误处理
def train_all_embeddings(csv_file, embed_dim=256, embeddings_dir="./embeddings", force_retrain=False):
    """
    训练并保存所有类型的嵌入向量

    参数:
        csv_file (str): CSV文件路径
        embed_dim (int): 嵌入维度
        embeddings_dir (str): 嵌入向量保存目录
        force_retrain (bool): 是否强制重新训练

    返回:
        dict: 嵌入类型到嵌入向量的映射
    """
    ensure_dir(embeddings_dir)

    # 检查是否已有预训练的嵌入
    all_embeddings = {}
    if not force_retrain:
        for embed_type in ['n2v', 'deepwalk', 'line', 'sdne']:
            embed_file = os.path.join(embeddings_dir, f"{embed_type}_embeddings_{embed_dim}.pkl")
            if os.path.exists(embed_file):
                try:
                    with open(embed_file, "rb") as f:
                        all_embeddings[embed_type] = pickle.load(f)
                    print(f"已加载现有的{embed_type}嵌入向量，包含{len(all_embeddings[embed_type])}个实体。")
                except Exception as e:
                    print(f"加载{embed_type}嵌入向量失败: {e}")

    # 如果需要训练，构建网络
    if len(all_embeddings) < 4 or force_retrain:
        try:
            # 构建基本网络
            G = build_network(csv_file)

            # 增强图的连接性
            G_enhanced = enhance_graph(G)

            # 训练每种嵌入方法
            for embed_type in ['n2v', 'deepwalk', 'line', 'sdne']:
                if embed_type not in all_embeddings or force_retrain:
                    try:
                        print(f"\n训练{embed_type}嵌入...")

                        if embed_type == 'n2v':
                            embeddings = train_node2vec(G_enhanced, dimensions=embed_dim)
                        elif embed_type == 'deepwalk':
                            embeddings = train_deepwalk(G_enhanced, dimensions=embed_dim)
                        elif embed_type == 'line':
                            embeddings = train_line(G_enhanced, dimensions=embed_dim)
                        elif embed_type == 'sdne':
                            embeddings = train_sdne(G_enhanced, dimensions=embed_dim)

                        save_embeddings(embeddings, embed_type, embed_dim, embeddings_dir)
                        all_embeddings[embed_type] = embeddings
                    except Exception as e:
                        print(f"训练{embed_type}嵌入失败: {e}")
                        print(f"生成随机{embed_type}嵌入作为后备方案...")
                        random_embeddings = generate_random_embeddings(G_enhanced, embed_dim)
                        save_embeddings(random_embeddings, embed_type, embed_dim, embeddings_dir)
                        all_embeddings[embed_type] = random_embeddings

        except Exception as e:
            print(f"构建网络失败: {e}")
            # 如果网络构建失败，为所有方法创建随机嵌入
            for embed_type in ['n2v', 'deepwalk', 'line', 'sdne']:
                if embed_type not in all_embeddings:
                    print(f"生成随机{embed_type}嵌入...")
                    # 创建一些随机节点以生成嵌入
                    dummy_G = nx.Graph()
                    for i in range(1000):
                        dummy_G.add_node(f"dummy_node_{i}")
                    for i in range(1000):
                        dummy_G.add_edge(f"dummy_node_{i}", f"dummy_node_{(i + 1) % 1000}")

                    random_embeddings = generate_random_embeddings(dummy_G, embed_dim)
                    save_embeddings(random_embeddings, embed_type, embed_dim, embeddings_dir)
                    all_embeddings[embed_type] = random_embeddings

    # 确保所有必要的嵌入类型都已加载
    for embed_type in ['n2v', 'deepwalk', 'line', 'sdne']:
        if embed_type not in all_embeddings:
            print(f"警告: {embed_type}嵌入不可用。创建随机嵌入作为后备方案。")
            dummy_embeddings = {}
            for i in range(1000):
                np.random.seed(i)
                dummy_embeddings[f"dummy_node_{i}"] = np.random.normal(0, 1, embed_dim)
            all_embeddings[embed_type] = dummy_embeddings

            # 保存以便下次使用
            save_embeddings(dummy_embeddings, embed_type, embed_dim, embeddings_dir)

    return all_embeddings

# 辅助函数 - 获取嵌入向量
def get_embedding(entity_id, embed_type="n2v", embed_dim=256, all_embeddings=None, embeddings_dir="./embeddings"):
    """
    获取给定实体的嵌入向量

    参数:
        entity_id (str): 实体ID
        embed_type (str): 嵌入类型
        embed_dim (int): 嵌入维度
        all_embeddings (dict): 所有嵌入向量，如果已有
        embeddings_dir (str): 嵌入向量保存目录

    返回:
        np.array: 嵌入向量
    """
    # 如果提供了嵌入向量集合，直接使用
    if all_embeddings is not None and embed_type in all_embeddings:
        embeddings = all_embeddings[embed_type]
        if entity_id in embeddings:
            return embeddings[entity_id]

    # 否则尝试加载嵌入向量
    embeddings = load_embeddings(embed_type, embed_dim, embeddings_dir)
    if entity_id in embeddings:
        return embeddings[entity_id]

    # 如果找不到嵌入向量，生成随机向量
    print(f"未找到实体 {entity_id} 的{embed_type}嵌入向量，使用随机向量。")
    np.random.seed(hash(str(entity_id)) % 10000)
    return np.random.normal(0, 1, embed_dim)


# 使用示例
if __name__ == "__main__":
    csv_file = "Dataset-of-activating-and-inhibiting-mechanisms.csv"

    # 训练所有嵌入方法
    all_embeddings = train_all_embeddings(csv_file, embed_dim=256, force_retrain=True)

    # 测试获取嵌入向量
    test_nodes = list(all_embeddings['n2v'].keys())[:5]
    for entity_id in test_nodes:
        for embed_type in ['n2v', 'deepwalk', 'line', 'sdne']:
            embedding = get_embedding(entity_id, embed_type, all_embeddings=all_embeddings)
            print(f"{embed_type}嵌入向量样本 {entity_id}: {embedding[:5]}...")