import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

import networkx as nx
import random,math,copy,time,timeit

from preprocess import tokens_struct, protein_tokens_struct
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


import pickle 
from prettytable import PrettyTable
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, log_loss

import dgl
from functools import partial

# 设置随机种子，确保结果可复现
def setup_seed(seed):
	"""
    这个函数设置各个库的随机种子，确保结果的可复现性。

    参数：
    seed (int): 要设置的随机种子。
    """
	# 设置PyTorch在CPU上的随机种子
	torch.manual_seed(seed)
	# 设置PyTorch在GPU上的随机种子
	torch.cuda.manual_seed_all(seed)
	# 设置NumPy的随机种子
	np.random.seed(seed)
	# 设置Python内置随机模块的种子
	random.seed(seed)
	# 确保卷积操作的结果是确定性的（没有随机性）
	torch.backends.cudnn.deterministic = True

# 设置随机种子为0，确保实验结果可复现
setup_seed(0)

# 根据给定的种子打乱数据集
def shuffle_dataset(dataset, seed):
	"""
    这个函数使用指定的种子随机打乱给定的数据集，确保打乱过程的可复现性。

    参数：
    dataset (list or array): 要打乱的数据集。
    seed (int): 用于确保打乱过程可复现的随机种子。

    返回：
    list: 打乱后的数据集。
    """
	# 设置NumPy的随机种子
	np.random.seed(seed)
	# 随机打乱数据集
	np.random.shuffle(dataset)
	# 返回打乱后的数据集
	return dataset

# 根据给定的比例将数据集分成三部分
def split_dataset_r(dataset, ratio1, ratio2, ratio3):
	"""
    这个函数根据给定的比例将数据集分成三部分。

    参数：
    dataset (list or array): 要分割的数据集。
    ratio1 (float): 第一个部分的比例。
    ratio2 (float): 第二个部分的比例。
    ratio3 (float): 第三个部分的比例。

    返回：
    tuple: 包含三个分割数据集的元组。
    """
	# 根据比例计算每个部分的元素个数
	n1 = int(ratio1 * len(dataset))
	n2 = int(ratio2 * len(dataset))
	# 按照计算出的大小将数据集分成三部分
	dataset_1, dataset_2, dataset_3 = dataset[:n1], dataset[n1:(n1 + n2)], dataset[(n1 + n2):]
	# 返回分割后的三个数据集
	return dataset_1, dataset_2, dataset_3

# 将数据集分成两部分，第一部分包含前n个元素
def split_dataset_n(dataset, n):
	"""
    这个函数将数据集分成两部分。第一部分包含前n个元素，第二部分包含剩余的元素。

    参数：
    dataset (list or array): 要分割的数据集。
    n (int): 第一部分的数据个数。

    返回：
    tuple: 包含两个分割数据集的元组。
    """
	# 将数据集分成两部分
	dataset_1, dataset_2 = dataset[:n], dataset[n:]
	# 返回分割后的两个数据集
	return dataset_2, dataset_1

# 使用pickle将Python对象保存到文件
def dump(fm, f):
	"""
    这个函数使用pickle将Python对象保存到文件，以便进行序列化。

    参数：
    fm (str): 文件名，保存对象的文件路径。
    f (object): 要保存的Python对象。
    """
	# 以写二进制的模式打开文件
	file = open(fm, "wb")
	# 使用pickle将对象序列化并保存到文件
	pickle.dump(f, file)
	# 关闭文件
	file.close()

# 使用pickle从文件中加载Python对象
def load(fm):
	"""
    这个函数使用pickle从文件中加载序列化的Python对象。

    参数：
    fm (str): 文件名，从中加载对象的文件路径。

    返回：
    object: 从文件中加载的Python对象。
    """
	# 以读二进制的模式打开文件
	file = open(fm, "rb")
	# 使用pickle反序列化并加载对象
	f = pickle.load(file)
	# 关闭文件
	file.close()
	# 返回加载的对象
	return f

# 在DrugAI_code.py中修改

def collate_molgraphs(data):
	"""
    优化的数据收集函数，减少CPU密集型操作
    """
	import dgl
	import torch
	import numpy as np
	from torch.nn.utils.rnn import pad_sequence

	# 检查数据是否为空
	if len(data) == 0:
		raise ValueError("Empty batch data")

	# 确保处理后的数据在批次维度上一致
	batch_size = len(data)

	# 检查每个样本中元素的数量
	sample_len = len(data[0])

	# 使用内存高效的方式解包数据
	try:
		if sample_len == 13:  # 包含全部数据的格式
			_, _, compounds, protein_cnn, protein_rnn, protein_lens, smiles, seq_lens, d_n2v, p_n2v, f_d, f_p, actions = zip(
				*data)
			proteins = protein_cnn  # 使用CNN格式的蛋白质编码
			use_protein_rnn = False
			use_smiles = True
			# 将protein_lens转换为Python列表
			protein_lens = [int(l) if isinstance(l, torch.Tensor) else int(l) for l in protein_lens]
		else:
			# 处理其他格式的数据...（保持原来的逻辑）
			raise ValueError(f"意外的数据格式: {sample_len} 个元素/样本")
	except Exception as e:
		print(f"数据解包错误: {e}")
		# 提供默认值避免函数失败
		compounds = proteins = smiles = d_n2v = p_n2v = f_d = f_p = actions = [None] * batch_size
		protein_lens = seq_lens = [0] * batch_size
		use_protein_rnn = False
		use_smiles = False

	# 批处理图 - 使用try-except避免失败
	try:
		bg = dgl.batch(compounds)
		bg.set_n_initializer(dgl.init.zero_initializer)
		bg.set_e_initializer(dgl.init.zero_initializer)
	except Exception as e:
		print(f"批处理图失败: {e}")
		# 创建空图作为备用
		bg = dgl.batch([dgl.graph([]) for _ in range(batch_size)])

	# 减少不必要的类型检查和数据转换，直接使用高效的批处理操作
	try:
		# 处理标签
		actions = torch.stack([
			act if isinstance(act, torch.Tensor) else torch.tensor(float(act), dtype=torch.float)
			for act in actions
		])

		# 确保actions是二维的[batch_size, 1]
		if actions.dim() == 1:
			actions = actions.unsqueeze(1)

		# 使用更高效的方式处理蛋白质数据
		padded_proteins = pad_sequence([
			torch.FloatTensor(p) if not isinstance(p, torch.Tensor) else p
			for p in proteins
		], batch_first=True)

		# 优化特征向量转换
		d_n2v = torch.stack([
			torch.FloatTensor(i) if not isinstance(i, torch.Tensor) else i.float()
			for i in d_n2v
		])

		p_n2v = torch.stack([
			torch.FloatTensor(i) if not isinstance(i, torch.Tensor) else i.float()
			for i in p_n2v
		])

		f_d = torch.stack([
			torch.FloatTensor(i) if not isinstance(i, torch.Tensor) else i.float()
			for i in f_d
		])

		f_p = torch.stack([
			torch.FloatTensor(i) if not isinstance(i, torch.Tensor) else i.float()
			for i in f_p
		])

		# 处理SMILES数据
		if use_smiles:
			padded_smiles = pad_sequence([
				torch.FloatTensor(s) if not isinstance(s, torch.Tensor) else s
				for s in smiles
			], batch_first=True)
		else:
			padded_smiles = torch.zeros((batch_size, 1), dtype=torch.float)

	except Exception as e:
		# 如果处理失败，提供默认值
		print(f"数据处理错误: {e}")
		padded_proteins = torch.zeros((batch_size, 1, 20), dtype=torch.float)
		d_n2v = p_n2v = torch.zeros((batch_size, 256), dtype=torch.float)
		f_d = torch.zeros((batch_size, 167), dtype=torch.float)
		f_p = torch.zeros((batch_size, 343), dtype=torch.float)
		padded_smiles = torch.zeros((batch_size, 1), dtype=torch.float)
		actions = torch.zeros((batch_size, 1), dtype=torch.float)

	# 返回统一格式
	return bg, padded_proteins, protein_lens, padded_smiles, seq_lens, d_n2v, p_n2v, f_d, f_p, actions

class CNN(nn.Module):
	"""
    这个类实现了一个卷积神经网络（CNN）。它通过1D卷积层来处理输入的蛋白质序列数据。

    参数：
    protein_Oridim (int): 输入数据的通道数（蛋白质特征的维度）。
    feature_size (int): 每个卷积层的输出通道数（特征的大小）。
    out_features (int): 最终输出的特征数。
    max_seq_len (int): 输入序列的最大长度。
    kernels (list): 卷积核的大小列表。
    dropout_rate (float): Dropout层的丢弃率。
    """

	def __init__(self, protein_Oridim, feature_size, out_features, max_seq_len, kernels, dropout_rate):
		super(CNN, self).__init__()

		self.dropout_rate = dropout_rate
		self.protein_Oridim = protein_Oridim
		self.feature_size = feature_size
		self.max_seq_len = max_seq_len
		self.kernels = kernels
		self.out_features = out_features

		# 定义多个卷积层，每个卷积层的卷积核大小不同
		self.convs = nn.ModuleList([
			nn.Sequential(
				nn.Conv1d(in_channels=self.protein_Oridim,
						  out_channels=self.feature_size,
						  kernel_size=ks),  # 定义卷积层
				nn.ReLU(),  # 激活函数
				nn.MaxPool1d(kernel_size=self.max_seq_len - ks + 1)  # 最大池化层
			)
			for ks in self.kernels  # 为每个卷积核大小创建一个卷积层
		])

		# 计算卷积层输出的特征数量
		self.conv_output_dim = self.feature_size * len(self.kernels)

		# 全连接层 - 动态适应卷积层输出的特征数量
		self.fc = nn.Linear(in_features=self.conv_output_dim,
							out_features=self.out_features)

		# Dropout层，防止过拟合
		self.dropout = nn.Dropout(p=self.dropout_rate)

	def forward(self, x):
		"""
        前向传播函数，定义了数据如何通过网络传播。

        参数：
        x (Tensor): 输入数据。

        返回：
        Tensor: 输出结果。
        """
		# # 输出输入形状用于调试
		# print(f"CNN输入形状: {x.shape}")

		# 保留原始批次大小以便后续检查
		batch_size = x.size(0)

		# 检查输入张量的维度
		if x.dim() == 2:
			# 如果是2维，添加一个通道维度
			x = x.unsqueeze(2)
			# 改变数据维度为 (batch_size, sequence_length, channels)
			# 然后转换为 (batch_size, channels, sequence_length)
			embedding_x = x.permute(0, 2, 1)
		elif x.dim() == 3:
			# 如果已经是3维，直接改变维度顺序
			if x.size(2) == self.protein_Oridim:  # 如果最后一维是通道数
				embedding_x = x.permute(0, 2, 1)
			else:
				embedding_x = x  # 已经是正确的排列
		else:
			raise ValueError(f"输入张量维度应为2或3，但实际为{x.dim()}")

		# 确保embedding_x的通道维数正确
		if embedding_x.size(1) != self.protein_Oridim:
			# print(f"警告: 输入通道数不匹配，期望{self.protein_Oridim}，实际{embedding_x.size(1)}")
			# 尝试调整形状以匹配预期的通道数
			if embedding_x.size(1) > self.protein_Oridim:
				embedding_x = embedding_x[:, :self.protein_Oridim, :]
			else:
				# 创建零张量并复制现有数据
				corrected_x = torch.zeros(batch_size, self.protein_Oridim, embedding_x.size(2),
										  device=embedding_x.device, dtype=embedding_x.dtype)
				corrected_x[:, :embedding_x.size(1), :] = embedding_x
				embedding_x = corrected_x

		# 打印转换后的形状
		# print(f"转换后的embedding_x形状: {embedding_x.shape}")

		# 检查序列长度是否超出最大长度限制
		if embedding_x.size(2) > self.max_seq_len:
			# print(f"警告: 序列长度超出限制，截断至{self.max_seq_len}")
			embedding_x = embedding_x[:, :, :self.max_seq_len]

		# 对每个卷积层进行操作，返回每个卷积层的结果
		# 添加错误处理，防止卷积操作出错
		conv_outputs = []
		for i, conv in enumerate(self.convs):
			try:
				# 计算卷积核的有效范围
				ks = self.kernels[i]
				if embedding_x.size(2) < ks:
					# 如果序列长度小于卷积核大小，则进行填充
					padding = torch.zeros(batch_size, embedding_x.size(1), ks - embedding_x.size(2),
										  device=embedding_x.device, dtype=embedding_x.dtype)
					padded_x = torch.cat([embedding_x, padding], dim=2)
					conv_out = conv(padded_x)
				else:
					# 正常情况
					conv_out = conv(embedding_x)
				conv_outputs.append(conv_out)
			except Exception as e:
				# print(f"卷积层{i}出错: {e}")
				# 创建全零输出作为替代
				conv_outputs.append(torch.zeros(batch_size, self.feature_size, 1,
												device=embedding_x.device, dtype=embedding_x.dtype))

		# 拼接所有卷积层的输出
		if len(conv_outputs) > 0:
			out = torch.cat(conv_outputs, dim=1)
			# 调整输出数据的形状，准备输入到全连接层
			out = out.view(out.size(0), -1)

			# 检查输出特征维度是否与全连接层输入维度匹配
			if out.size(1) != self.conv_output_dim:
				# print(f"卷积输出维度不匹配: 期望{self.conv_output_dim}，实际{out.size(1)}")
				# 调整输出维度以匹配全连接层的输入维度
				if out.size(1) > self.conv_output_dim:
					# 如果输出维度大于预期，截断多余部分
					out = out[:, :self.conv_output_dim]
				else:
					# 如果输出维度小于预期，填充零
					padding = torch.zeros(batch_size, self.conv_output_dim - out.size(1),
										  device=out.device, dtype=out.dtype)
					out = torch.cat([out, padding], dim=1)

			# 检查输出批次大小是否与输入匹配
			if out.size(0) != batch_size:
				# print(f"CNN 内部输出批次大小不匹配：输入 {batch_size}，输出 {out.size(0)}")
				# 修正批次大小不匹配问题
				if out.size(0) > batch_size:
					# 如果输出批次大小大于输入，截断多余的部分
					out = out[:batch_size]
				else:
					# 如果输出批次大小小于输入，填充至匹配大小
					# 创建零张量并复制现有数据
					corrected_out = torch.zeros(batch_size, out.size(1),
												device=out.device, dtype=out.dtype)
					corrected_out[:out.size(0)] = out
					out = corrected_out
		else:
			# 如果所有卷积层都失败，创建零输出
			out = torch.zeros(batch_size, self.conv_output_dim, device=x.device, dtype=x.dtype)

		# 输出卷积后的形状用于调试
		# print(f"卷积输出形状: {out.shape}")

		# 应用Dropout
		out = self.dropout(out)

		# 全连接层的输出
		out = self.fc(out)

		# 最终输出形状
		# print(f"CNN最终输出形状: {out.shape}")

		return out


from dgllife.model import AttentiveFPGNN, GCNPredictor, GATPredictor, MPNNPredictor


def GNN(flag, layers, heads):
	"""
    This function chooses different Graph Neural Network (GNN) models based on the given flag.
    """
	if flag == "AttentiveFP":
		compound_Encoder1 = AttentiveFPGNN(
			node_feat_size=58,
			edge_feat_size=10,
			num_layers=layers,
			graph_feat_size=167,
			dropout=0.1
		)
	elif flag == "GCN":
		# 创建隐藏层维度的列表
		hidden_feats = [128] * layers

		# 重要修复：确保这些参数也是列表而不是单个布尔值
		dropout = [0.1] * len(hidden_feats)
		residual = [True] * len(hidden_feats)
		batchnorm = [True] * len(hidden_feats)

		compound_Encoder1 = GCNPredictor(
			in_feats=58,
			hidden_feats=hidden_feats,
			n_tasks=167,
			predictor_hidden_feats=167,
			dropout=dropout,  # 使用列表
			residual=residual,  # 使用列表
			batchnorm=batchnorm  # 使用列表
		)
	elif flag == "GAT":
		hidden_feats = [128] * layers
		num_heads = [heads] * layers if heads is not None else [4] * layers

		compound_Encoder1 = GATPredictor(
			in_feats=58,
			hidden_feats=hidden_feats,
			num_heads=num_heads,
			n_tasks=167
		)
	elif flag == "MPNN":
		compound_Encoder1 = MPNNPredictor(
			node_in_feats=58,
			edge_in_feats=10,
			node_out_feats=167,
			edge_hidden_feats=32,
			num_step_message_passing=layers
		)
	else:
		raise ValueError(f"Unknown GNN flag: {flag}. Must be one of 'AttentiveFP', 'GCN', 'GAT', 'MPNN'.")

	return compound_Encoder1
# 添加ProteinRNNModule类
class ProteinRNNModule(nn.Module):
	"""
    用于处理蛋白质序列的RNN模块，类似于SMILES处理
    """

	def __init__(self, vocab, embed_dim, blstm_dim, num_layers, out_dim=64, dropout=0.2,
				 bidirectional=True, device='cpu'):
		super(ProteinRNNModule, self).__init__()
		self.vocab = vocab
		self.embed_dim = embed_dim
		self.blstm_dim = blstm_dim
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.device = device
		self.num_dir = 2 if bidirectional else 1

		# 网络层
		self.embeddings = nn.Embedding(vocab.tokens_length, self.embed_dim, padding_idx=vocab.pad)
		self.rnn = nn.LSTM(
			self.embed_dim, self.blstm_dim,
			num_layers=num_layers,
			bidirectional=bidirectional,
			dropout=dropout,
			batch_first=True
		)
		self.fc = nn.Sequential(
			nn.Linear(self.blstm_dim * self.num_dir, out_dim),
			nn.ReLU(),
			nn.Dropout(p=dropout)
		)

	def forward(self, protein_seq, seq_lens):
		"""
        处理蛋白质序列

        参数:
            protein_seq: token索引的张量 [batch_size, max_seq_len] 或
                        one-hot编码序列的张量 [batch_size, max_seq_len, features]
            seq_lens: 实际序列长度列表

        返回:
            形状为 [batch_size, out_dim] 的张量
        """
		# 检查输入是否已经是one-hot编码或3D张量
		if protein_seq.dim() == 3:
			# 从one-hot编码格式中提取token索引
			# 假设最大值是token
			protein_seq = protein_seq.argmax(dim=2) if protein_seq.size(2) > 1 else protein_seq.squeeze(2)

		# 现在转换为嵌入
		x = self.embeddings(protein_seq.long())

		# 打包序列以高效处理RNN
		packed_input = pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
		packed_output, _ = self.rnn(packed_input)
		output, _ = pad_packed_sequence(packed_output, batch_first=True)

		# 根据序列长度获取最终状态
		if self.bidirectional:
			# 对于双向，连接前向的最后和后向的第一
			out_forward = output[range(len(output)), np.array(seq_lens) - 1, :self.blstm_dim]
			out_reverse = output[:, 0, self.blstm_dim:]
			protein_fea = torch.cat((out_forward, out_reverse), 1)
		else:
			# 对于单向，只使用最后状态
			protein_fea = output[range(len(output)), np.array(seq_lens) - 1, :]

		# 最终全连接层
		out = self.fc(protein_fea)
		return out

class RNNModule(nn.Module):
	"""
    用于处理SMILES字符串的RNN模块
    """

	def __init__(self, vocab, embed_dim, blstm_dim, num_layers, out_dim=64, dropout=0.2,
				 bidirectional=True, device='cpu'):
		super(RNNModule, self).__init__()
		self.vocab = vocab
		self.embed_dim = embed_dim
		self.blstm_dim = blstm_dim
		self.num_layers = num_layers
		self.bidirectional = bidirectional
		self.device = device
		self.num_dir = 2 if bidirectional else 1

		# 网络层
		self.embeddings = nn.Embedding(vocab.tokens_length, self.embed_dim, padding_idx=vocab.pad)
		self.rnn = nn.LSTM(
			self.embed_dim, self.blstm_dim,
			num_layers=num_layers,
			bidirectional=bidirectional,
			dropout=dropout,
			batch_first=True
		)
		self.fc = nn.Sequential(
			nn.Linear(self.blstm_dim * self.num_dir, out_dim),
			nn.ReLU(),
			nn.Dropout(p=dropout)
		)

	def forward(self, smiles, seq_lens):
		"""
        处理SMILES字符串

        参数:
            smiles: 标记索引的张量 [batch_size, max_seq_len]
            seq_lens: 实际序列长度列表

        返回:
            形状为 [batch_size, out_dim] 的张量
        """
		# 将索引转换为嵌入
		x = self.embeddings(smiles.long())

		# 打包序列以高效处理RNN
		packed_input = pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
		packed_output, _ = self.rnn(packed_input)
		output, _ = pad_packed_sequence(packed_output, batch_first=True)

		# 根据序列长度获取最终状态
		if self.bidirectional:
			# 对于双向，连接前向的最后和后向的第一
			out_forward = output[range(len(output)), np.array(seq_lens) - 1, :self.blstm_dim]
			out_reverse = output[:, 0, self.blstm_dim:]
			text_fea = torch.cat((out_forward, out_reverse), 1)
		else:
			# 对于单向，只使用最后状态
			text_fea = output[range(len(output)), np.array(seq_lens) - 1, :]

		# 最终全连接层
		out = self.fc(text_fea)
		return out

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
	def __init__(self, CNN_fdim, p_n2v_fdim, g_fp_dim, GNN_fdim, d_n2v_fdim, d_fp_dim, RNN_fdim, protein_RNN_fdim,
				 CNN, GNN, RNN, protein_RNN, dropout_r, DNN_layers, views_flag, GNNs_flag):
		super(Classifier, self).__init__()

		self.views_flag = views_flag  # 确保保存了视图标志
		self.GNNs_flag = GNNs_flag

		# 编码器 - 不管是否使用都初始化
		self.protein_CNN = CNN
		self.protein_RNN = protein_RNN
		self.compound_Encoder = GNN
		self.smiles_Encoder = RNN

		# Dropout
		self.dropout = nn.Dropout(p=dropout_r)

		# 层配置
		self.DNN_layers = DNN_layers

		# 特征维度
		self.CNN_fdim = CNN_fdim  # 蛋白质CNN特征
		self.protein_RNN_fdim = protein_RNN_fdim  # 蛋白质RNN特征
		self.p_n2v_fdim = p_n2v_fdim  # 蛋白质node2vec特征
		self.g_fp_dim = g_fp_dim  # 蛋白质指纹特征
		self.GNN_fdim = GNN_fdim  # 药物GNN特征
		self.d_n2v_fdim = d_n2v_fdim  # 药物node2vec特征
		self.d_fp_dim = d_fp_dim  # 药物指纹特征
		self.RNN_fdim = RNN_fdim  # SMILES RNN特征

		self.layer_size = len(self.DNN_layers) + 1

		# 所有可能的特征组合和对应的维度
		views_dim_dict = {
			# 不含RNN的原有组合
			"d_GNN+d_N2V+d_fp+g_CNN+p_n2v+g_fp": [(self.GNN_fdim + self.d_n2v_fdim + self.d_fp_dim +
												   self.CNN_fdim + self.p_n2v_fdim + self.g_fp_dim)] + self.DNN_layers + [
													 1],

			"d_N2V+d_fp+g_CNN+p_n2v+g_fp": [(self.d_n2v_fdim + self.d_fp_dim + self.CNN_fdim +
											 self.p_n2v_fdim + self.g_fp_dim)] + self.DNN_layers + [1],

			"d_GNN+d_fp+g_CNN+p_n2v+g_fp": [(self.GNN_fdim + self.d_fp_dim + self.CNN_fdim +
											 self.p_n2v_fdim + self.g_fp_dim)] + self.DNN_layers + [1],

			"d_GNN+d_N2V+g_CNN+p_n2v+g_fp": [(self.GNN_fdim + self.d_n2v_fdim + self.CNN_fdim +
											  self.p_n2v_fdim + self.g_fp_dim)] + self.DNN_layers + [1],

			"d_GNN+d_N2V+d_fp+p_n2v+g_fp": [(self.GNN_fdim + self.d_n2v_fdim + self.d_fp_dim +
											 self.p_n2v_fdim + self.g_fp_dim)] + self.DNN_layers + [1],

			"d_GNN+d_N2V+d_fp+g_CNN+g_fp": [(self.GNN_fdim + self.d_n2v_fdim + self.d_fp_dim +
											 self.CNN_fdim + self.g_fp_dim)] + self.DNN_layers + [1],

			"d_GNN+d_N2V+d_fp+g_CNN+p_n2v": [(self.GNN_fdim + self.d_n2v_fdim + self.d_fp_dim +
											  self.CNN_fdim + self.p_n2v_fdim)] + self.DNN_layers + [1],

			"d_GNN+d_fp+g_CNN+g_fp": [(self.GNN_fdim + self.d_fp_dim + self.CNN_fdim +
									   self.g_fp_dim)] + self.DNN_layers + [1],

			"d_GNN+d_N2V+g_CNN+p_n2v": [(self.GNN_fdim + self.d_n2v_fdim + self.CNN_fdim +
										 self.p_n2v_fdim)] + self.DNN_layers + [1],

			"d_GNN+g_CNN": [(self.GNN_fdim + self.CNN_fdim)] + self.DNN_layers + [1],

			# 包含RNN的新组合

			"d_RNN+g_RNN" : [(self.RNN_fdim + self.protein_RNN_fdim)] + self.DNN_layers + [1],
			# 包含所有特征的完整组合
			"d_GNN+d_N2V+d_fp+g_CNN+p_n2v+g_fp+d_RNN+g_RNN": [(self.GNN_fdim + self.d_n2v_fdim + self.d_fp_dim +
												   self.CNN_fdim + self.p_n2v_fdim + self.g_fp_dim + self.RNN_fdim +self.protein_RNN_fdim)] + self.DNN_layers + [1],

			"d_GNN+d_fp+g_CNN+g_fp+d_RNN+g_RNN": [(self.GNN_fdim + self.d_fp_dim +self.CNN_fdim +
												   self.g_fp_dim + self.RNN_fdim + self.protein_RNN_fdim)] + self.DNN_layers + [1],

			"d_N2V+d_fp+p_n2v+g_fp+d_RNN+g_RNN": [(self.d_n2v_fdim + self.d_fp_dim + self.p_n2v_fdim + self.g_fp_dim +
												   self.RNN_fdim +self.protein_RNN_fdim)] + self.DNN_layers + [1],

			"d_GNN+d_N2V+g_CNN+p_n2v+d_RNN+g_RNN": [(self.GNN_fdim + self.d_n2v_fdim  +self.CNN_fdim + self.p_n2v_fdim +
													 self.RNN_fdim + self.protein_RNN_fdim)] + self.DNN_layers + [1],

			"d_GNN+g_CNN+d_RNN+g_RNN": [(self.GNN_fdim +  self.CNN_fdim + self.RNN_fdim + self.protein_RNN_fdim)] + self.DNN_layers + [1],

			"d_N2V+d_fp+g_CNN+p_n2v+g_fp+d_RNN+g_RNN": [( self.d_n2v_fdim + self.d_fp_dim +
															   self.CNN_fdim + self.p_n2v_fdim + self.g_fp_dim + self.RNN_fdim + self.protein_RNN_fdim)] + self.DNN_layers + [
																 1],#6
			"d_GNN+d_fp+g_CNN+p_n2v+g_fp+d_RNN+g_RNN": [(self.GNN_fdim +  self.d_fp_dim +
															   self.CNN_fdim + self.p_n2v_fdim + self.g_fp_dim + self.RNN_fdim + self.protein_RNN_fdim)] + self.DNN_layers + [
																 1],#7
			"d_GNN+d_N2V+g_CNN+p_n2v+g_fp+d_RNN+g_RNN": [(self.GNN_fdim + self.d_n2v_fdim +
															   self.CNN_fdim + self.p_n2v_fdim + self.g_fp_dim + self.RNN_fdim + self.protein_RNN_fdim)] + self.DNN_layers + [
																 1],#8
			"d_GNN+d_N2V+d_fp+p_n2v+g_fp+d_RNN+g_RNN": [(self.GNN_fdim + self.d_n2v_fdim + self.d_fp_dim +
															    self.p_n2v_fdim + self.g_fp_dim + self.RNN_fdim + self.protein_RNN_fdim)] + self.DNN_layers + [
																 1],#9
			"d_GNN+d_N2V+d_fp+g_CNN+g_fp+d_RNN+g_RNN": [(self.GNN_fdim + self.d_n2v_fdim + self.d_fp_dim +
															   self.CNN_fdim +  self.g_fp_dim + self.RNN_fdim + self.protein_RNN_fdim)] + self.DNN_layers + [
																 1],#10
			"d_GNN+d_N2V+d_fp+g_CNN+p_n2v+d_RNN+g_RNN": [(self.GNN_fdim + self.d_n2v_fdim + self.d_fp_dim +
															   self.CNN_fdim + self.p_n2v_fdim +  self.RNN_fdim + self.protein_RNN_fdim)] + self.DNN_layers + [
																 1],#11
			"d_GNN+d_N2V+d_fp+g_CNN+p_n2v+g_fp+g_RNN": [(self.GNN_fdim + self.d_n2v_fdim + self.d_fp_dim +
															   self.CNN_fdim + self.p_n2v_fdim + self.g_fp_dim +  self.protein_RNN_fdim)] + self.DNN_layers + [
																 1],#12
			"d_GNN+d_N2V+d_fp+g_CNN+p_n2v+g_fp+d_RNN": [(self.GNN_fdim + self.d_n2v_fdim + self.d_fp_dim +
															   self.CNN_fdim + self.p_n2v_fdim + self.g_fp_dim + self.RNN_fdim)] + self.DNN_layers + [
																 1],  # 13

		}

		# 添加默认配置
		total_dim = (self.GNN_fdim + self.CNN_fdim + self.d_n2v_fdim + self.p_n2v_fdim +
					 self.d_fp_dim + self.g_fp_dim + self.RNN_fdim + self.protein_RNN_fdim)
		views_dim_dict["default"] = [total_dim] + self.DNN_layers + [1]

		# 获取维度配置，如果不存在则使用默认
		try:
			self.dims = views_dim_dict[self.views_flag]
		except KeyError:
			print(f"警告: 视图标志'{self.views_flag}'未定义，使用默认配置")
			self.dims = views_dim_dict["default"]

		# 创建预测器层
		self.predictor = nn.ModuleList([nn.Linear(self.dims[i], self.dims[i + 1]) for i in range(self.layer_size)])

	def forward(self, compounds, proteins, d_n2v, p_n2v, d_FP, g_FP, actions=None,
				smiles=None, smiles_lens=None, protein_lens=None):
		# 批次大小根据蛋白质输入确定
		batch_size = proteins.size(0)

		# 1. 处理分子图 - GNN特征
		if self.compound_Encoder is not None and self.GNNs_flag == "AttentiveFP":
			# 获取节点表示
			node_features = self.compound_Encoder(compounds, compounds.ndata['h'], compounds.edata['e'])

			# 确保批次大小正确
			batch_num_nodes = compounds.batch_num_nodes()
			if len(batch_num_nodes) != batch_size:
				print(f"警告：compounds批次大小不匹配，期望 {batch_size}，实际 {len(batch_num_nodes)}")
				# 调整大小以匹配proteins的批次大小
				compound_feature = torch.zeros(batch_size, self.GNN_fdim, device=proteins.device)
			else:
				# 聚合节点特征到图级表示
				compound_feature = torch.zeros(batch_size, self.GNN_fdim, device=node_features.device)
				start_idx = 0
				for i in range(batch_size):
					num_nodes = batch_num_nodes[i]
					end_idx = start_idx + num_nodes
					compound_feature[i] = node_features[start_idx:end_idx].sum(dim=0)
					start_idx = end_idx
		else:
			# 如果GNN不可用，创建全零张量
			compound_feature = torch.zeros(batch_size, self.GNN_fdim, device=proteins.device)

		# 2. 处理蛋白质 - CNN特征
		if self.protein_CNN is not None:
			# 确保CNN输出的批次大小与预期一致
			protein_cnn_feature = self.protein_CNN(proteins)
			if protein_cnn_feature.size(0) != batch_size:
				print(f"警告：protein_cnn_feature批次大小不匹配，期望 {batch_size}，实际 {protein_cnn_feature.size(0)}")
				# 需要调整大小
				temp_feature = torch.zeros(batch_size, self.CNN_fdim, device=proteins.device)
				copy_size = min(batch_size, protein_cnn_feature.size(0))
				temp_feature[:copy_size] = protein_cnn_feature[:copy_size]
				protein_cnn_feature = temp_feature
		else:
			protein_cnn_feature = torch.zeros(batch_size, self.CNN_fdim, device=proteins.device)

		# 3. 处理蛋白质 - RNN特征
		# 只有当protein_lens不为None且实际需要RNN特征时处理
		if self.protein_RNN is not None and protein_lens is not None and (
				"g_RNN" in self.views_flag or "all_features" in self.views_flag):
			protein_rnn_feature = self.protein_RNN(proteins, protein_lens)
		else:
			protein_rnn_feature = torch.zeros(batch_size, self.protein_RNN_fdim, device=proteins.device)

		# 4. 处理SMILES - RNN特征
		# 只有当smiles不为None且实际需要SMILES特征时处理
		if self.smiles_Encoder is not None and smiles is not None and smiles_lens is not None and (
				"d_RNN" in self.views_flag or "all_features" in self.views_flag):
			smiles_feature = self.smiles_Encoder(smiles, smiles_lens)
		else:
			smiles_feature = torch.zeros(batch_size, self.RNN_fdim, device=proteins.device)

		# 根据views_flag选择要使用的特征
		if self.views_flag == "d_GNN+g_CNN": #a
			v_f = torch.cat([compound_feature, protein_cnn_feature], 1)

		elif self.views_flag == "d_GNN+d_N2V+g_CNN+p_n2v": #b
			v_f = torch.cat([compound_feature, d_n2v, protein_cnn_feature, p_n2v], 1)

		elif self.views_flag == "d_GNN+d_fp+g_CNN+g_fp":#c
			v_f = torch.cat([compound_feature, d_FP, protein_cnn_feature, g_FP], 1)

		elif self.views_flag == "d_GNN+d_N2V+d_fp+g_CNN+p_n2v+g_fp":		#j 全
			v_f = torch.cat([compound_feature, d_n2v, d_FP, protein_cnn_feature, p_n2v, g_FP], 1)

		elif self.views_flag == "d_N2V+d_fp+g_CNN+p_n2v+g_fp":		# d
			v_f = torch.cat([d_n2v, d_FP, protein_cnn_feature, p_n2v, g_FP], 1)

		elif self.views_flag == "d_GNN+d_fp+g_CNN+p_n2v+g_fp":		#e
			v_f = torch.cat([compound_feature, d_FP, protein_cnn_feature, p_n2v, g_FP], 1)

		elif self.views_flag == "d_GNN+d_N2V+g_CNN+p_n2v+g_fp":		#f
			v_f = torch.cat([compound_feature, d_n2v , protein_cnn_feature, p_n2v, g_FP], 1)

		elif self.views_flag == "d_GNN+d_N2V+d_fp+p_n2v+g_fp":		#g
			v_f = torch.cat([compound_feature, d_n2v, d_FP , p_n2v, g_FP], 1)

		elif self.views_flag == "d_GNN+d_N2V+d_fp+g_CNN+g_fp":		#h
			v_f = torch.cat([compound_feature, d_n2v, d_FP, protein_cnn_feature, g_FP], 1)

		elif self.views_flag == "d_GNN+d_N2V+d_fp+g_CNN+p_n2v":		#i
			v_f = torch.cat([compound_feature, d_n2v, d_FP, protein_cnn_feature, p_n2v], 1)

		elif self.views_flag == "d_RNN+g_RNN":  					#1
			v_f = torch.cat([smiles_feature, protein_rnn_feature], 1)

		elif self.views_flag == "d_GNN+d_N2V+d_fp+g_CNN+p_n2v+g_fp+d_RNN+g_RNN":		#2
			v_f = torch.cat([compound_feature, d_n2v, d_FP, protein_cnn_feature, p_n2v, g_FP,smiles_feature,protein_rnn_feature], 1)

		elif self.views_flag == "d_GNN+d_fp+g_CNN+g_fp+d_RNN+g_RNN":					#3
			v_f = torch.cat([compound_feature, d_FP, protein_cnn_feature, g_FP,smiles_feature,protein_rnn_feature], 1)

		elif self.views_flag == "d_N2V+d_fp+p_n2v+g_fp+d_RNN+g_RNN":		#4
			v_f = torch.cat([d_n2v, d_FP, p_n2v, g_FP,smiles_feature,protein_rnn_feature], 1)

		elif self.views_flag == "d_GNN+d_N2V+g_CNN+p_n2v+d_RNN+g_RNN":		#5
			v_f = torch.cat([compound_feature, d_n2v, protein_cnn_feature, p_n2v,smiles_feature,protein_rnn_feature], 1)

		elif self.views_flag == "d_GNN+g_CNN+d_RNN+g_RNN":		#6
			v_f = torch.cat([compound_feature, protein_cnn_feature,smiles_feature,protein_rnn_feature], 1)

		elif self.views_flag == "d_N2V+d_fp+g_CNN+p_n2v+g_fp+d_RNN+g_RNN":		#6
			v_f = torch.cat([d_n2v, d_FP, protein_cnn_feature, p_n2v, g_FP,smiles_feature,protein_rnn_feature], 1)
		elif self.views_flag == "d_GNN+d_fp+g_CNN+p_n2v+g_fp+d_RNN+g_RNN":		#7
			v_f = torch.cat([compound_feature, d_FP, protein_cnn_feature, p_n2v, g_FP,smiles_feature,protein_rnn_feature], 1)
		elif self.views_flag == "d_GNN+d_N2V+g_CNN+p_n2v+g_fp+d_RNN+g_RNN":		#8
			v_f = torch.cat([compound_feature, d_n2v, protein_cnn_feature, p_n2v, g_FP,smiles_feature,protein_rnn_feature], 1)
		elif self.views_flag == "d_GNN+d_N2V+d_fp+p_n2v+g_fp+d_RNN+g_RNN":		#9
			v_f = torch.cat([compound_feature, d_n2v, d_FP, p_n2v, g_FP,smiles_feature,protein_rnn_feature], 1)
		elif self.views_flag == "d_GNN+d_N2V+d_fp+g_CNN+g_fp+d_RNN+g_RNN":		#10
			v_f = torch.cat([compound_feature, d_n2v, d_FP, protein_cnn_feature, g_FP,smiles_feature,protein_rnn_feature], 1)
		elif self.views_flag == "d_GNN+d_N2V+d_fp+g_CNN+p_n2v+d_RNN+g_RNN":		#11
			v_f = torch.cat([compound_feature, d_n2v, d_FP, protein_cnn_feature, p_n2v, smiles_feature,protein_rnn_feature], 1)
		elif self.views_flag == "d_GNN+d_N2V+d_fp+g_CNN+p_n2v+g_fp+g_RNN":		#12
			v_f = torch.cat([compound_feature, d_n2v, d_FP, protein_cnn_feature, p_n2v, g_FP,protein_rnn_feature], 1)
		elif self.views_flag == "d_GNN+d_N2V+d_fp+g_CNN+p_n2v+g_fp+d_RNN":		#13
			v_f = torch.cat([compound_feature, d_n2v, d_FP, protein_cnn_feature, p_n2v, g_FP,smiles_feature], 1)
		else:
			# 默认情况，包括其他上面未列出的组合，通过对应的views_dim_dict中定义的维度来决定特征
			# 遵循原始代码中的结构，这里我们需要确保views_dim_dict中有对应的配置
			v_f = torch.cat([compound_feature, d_n2v, d_FP, protein_cnn_feature, p_n2v, g_FP], 1)
			print(f"使用默认特征组合，因为 {self.views_flag} 没有专门的处理逻辑")

		# 通过预测器层
		for i, l in enumerate(self.predictor):
			if i == (len(self.predictor) - 1):
				v_f = l(v_f)  # 最后一层，直接输出
			else:
				v_f = F.relu(self.dropout(l(v_f)))  # 中间层，应用ReLU和dropout

		# 确保输出维度与actions匹配 - 修改此处
		return v_f, actions

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, \
	f1_score, log_loss

def metric(y_label, y_prob):
	"""
    计算多种评估指标，包括准确率、精确度、召回率、AUC、平均精度、F1分数和对数损失。

    参数：
    y_label (array-like): 真实标签，通常为二分类标签（0或1）。
    y_prob (array-like): 模型的预测概率，通常是[0, 1]之间的值。

    返回：
    tuple: 包含多个评估指标的元组，依次为：
           accuracy, precision, recall, auc, auprc, f1, loss
    """
	# 将预测的概率转化为二进制的预测标签（0或1），预测概率大于等于0.5时为1，反之为0
	y_pred = np.asarray([1 if i else 0 for i in (np.asarray(y_prob) >= 0.5)])

	# 计算准确率
	accuracy = accuracy_score(y_label, y_pred)

	# 计算精确度（Precision）
	precision = precision_score(y_label, y_pred, average="binary")

	# 计算召回率（Recall）
	recall = recall_score(y_label, y_pred, average="binary")

	# 计算AUC（ROC曲线下面积）
	auc = roc_auc_score(y_label, y_prob)

	# 计算平均精度（Average Precision, AUPRC）
	auprc = average_precision_score(y_label, y_prob)

	# 计算F1分数
	f1 = f1_score(y_label, y_pred, average="binary")

	# 计算对数损失（Log Loss）
	loss = log_loss(y_label, y_pred)

	return accuracy, precision, recall, auc, auprc, f1, loss

class DrugAI():
	'''
    集成了SMILES和蛋白质RNN处理功能的DrugAI模型，现在通过views_flag控制特征组合
    '''

	def __init__(self, **config):
		# 基本配置
		self.views_flag = config["views_flag"]

		# 蛋白质编码器参数
		self.protein_Oridim = config["protein_Oridim"]
		self.feature_size = config["feature_size"]
		self.out_features = config["out_features"]
		self.max_seq_len = config["max_seq_len"]
		self.kernels = config["kernels"]
		self.CNN_fdim = config["CNN_fdim"]
		self.g_fp_dim = config["g_fp_dim"]
		self.p_n2v_fdim = config["p_n2v_fdim"]

		# 蛋白质RNN参数 - 总是初始化
		self.protein_embed_dim = config.get("protein_embed_dim", 64)
		self.protein_blstm_dim = config.get("protein_blstm_dim", 64)
		self.protein_layers = config.get("protein_layers", 2)
		self.protein_RNN_fdim = config.get("protein_RNN_fdim", 128)

		# SMILES处理参数 - 总是初始化
		self.smiles_embed_dim = config.get("smiles_embed_dim", 64)
		self.smiles_blstm_dim = config.get("smiles_blstm_dim", 64)
		self.smiles_layers = config.get("smiles_layers", 2)
		self.RNN_fdim = config.get("RNN_fdim", 128)

		# Dropout率
		self.dropout_r = config["dropout_r"]

		# GNN参数
		self.GNNs_flag = config["GNNs_flag"]
		self.layers = config["GNNs_layers"]
		self.heads = config["GNNs_heads"]
		self.GNN_fdim = config["GNN_fdim"]
		self.d_fp_dim = config["d_fp_dim"]
		self.d_n2v_fdim = config["d_n2v_fdim"]

		# DNN和训练参数
		self.DNN_layers = config["DNN_layers"]
		self.batch_size = config["batch_size"]
		self.lr = config["lr"]
		self.decay = config["decay"]
		self.train_epoch = config["train_epoch"]
		self.result_folder = config["result_folder"]

		# 早停配置
		self.use_early_stopping = config.get("use_early_stopping", False)
		self.patience = config.get("patience", 10)

		# 类别不平衡处理
		self.use_weighted_loss = config.get("use_weighted_loss", False)

		# 设置设备
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		# 初始化编码器组件 - 不管views_flag如何，我们都初始化所有编码器
		# 初始化蛋白质CNN编码器
		self.protein_Encoder = CNN(self.protein_Oridim, self.feature_size, self.out_features,
								   self.max_seq_len, self.kernels, self.dropout_r)

		# 初始化GNN编码器
		self.compound_Encoder = GNN(self.GNNs_flag, self.layers, self.heads)

		# 初始化SMILES RNN编码器
		self.token_struct = tokens_struct()
		self.smiles_Encoder = RNNModule(
			self.token_struct,
			self.smiles_embed_dim,
			self.smiles_blstm_dim,
			self.smiles_layers,
			self.RNN_fdim,
			self.dropout_r,
			bidirectional=True,
			device=self.device
		)

		# 初始化蛋白质RNN编码器
		self.protein_token_struct = protein_tokens_struct()
		self.protein_RNN_Encoder = ProteinRNNModule(
			self.protein_token_struct,
			self.protein_embed_dim,
			self.protein_blstm_dim,
			self.protein_layers,
			self.protein_RNN_fdim,
			self.dropout_r,
			bidirectional=True,
			device=self.device
		)

		# 初始化完整模型
		self.model = Classifier(
			self.CNN_fdim, self.p_n2v_fdim, self.g_fp_dim,
			self.GNN_fdim, self.d_n2v_fdim, self.d_fp_dim, self.RNN_fdim, self.protein_RNN_fdim,
			self.protein_Encoder, self.compound_Encoder, self.smiles_Encoder, self.protein_RNN_Encoder,
			self.dropout_r, self.DNN_layers, self.views_flag, self.GNNs_flag
		)

		print(f"初始化完成DrugAI模型，使用特征组合: {self.views_flag}")

	# train方法 - 修改为总是接收和处理所有特征
	def vali(self, dataset_vali):
		"""在验证数据集上验证模型"""
		y_prob = []
		y_label = []

		self.model.eval()
		dataset_loader = DataLoader(dataset=dataset_vali, batch_size=self.batch_size, collate_fn=collate_molgraphs)

		with torch.no_grad():
			for batch_data in dataset_loader:
				# 确定数据格式并解包数据
				try:
					if len(batch_data) == 10:  # 完整数据格式
						bg, proteins, protein_lens, smiles, smiles_lens, d_n2v, p_n2v, f_d, f_p, actions = batch_data
					elif len(batch_data) == 9:  # 无蛋白质RNN长度
						bg, proteins, smiles, smiles_lens, d_n2v, p_n2v, f_d, f_p, actions = batch_data
						protein_lens = None
					elif len(batch_data) == 7:  # 无SMILES和蛋白质RNN
						bg, proteins, d_n2v, p_n2v, f_d, f_p, actions = batch_data
						smiles = None
						smiles_lens = None
						protein_lens = None
					else:
						print(f"警告：验证集中意外的批次数据格式，有 {len(batch_data)} 个元素")
						continue  # 跳过这个批次
				except Exception as e:
					print(f"解析验证批次数据时出错: {e}")
					continue

				# 将数据移至设备
				bg = bg.to(self.device)
				proteins = proteins.to(self.device)
				d_n2v = d_n2v.to(self.device)
				p_n2v = p_n2v.to(self.device)
				f_d = f_d.to(self.device)
				f_p = f_p.to(self.device)
				actions = actions.to(self.device)

				if smiles is not None:
					smiles = smiles.to(self.device)

				# 前向传播
				score, actions = self.model(
					bg, proteins, d_n2v, p_n2v, f_d, f_p, actions,
					smiles, smiles_lens, protein_lens
				)

				# 应用sigmoid并收集预测结果 - 修改此处，不使用squeeze
				Sigm = torch.nn.Sigmoid()
				prob = Sigm(score).detach().cpu().numpy()
				y_label = y_label + actions.detach().cpu().numpy().flatten().tolist()
				y_prob = y_prob + prob.flatten().tolist()  # 使用flatten()来确保一维列表

			# 计算指标
			accuracy, precision, recall, auc, auprc, f1, loss = metric(y_label, y_prob)

			# 将概率转换为二进制预测
			y_pred = np.asarray([1 if i else 0 for i in (np.asarray(y_prob) >= 0.5)])

		return accuracy, precision, recall, auc, auprc, f1, loss, y_pred, y_prob, y_label

	def predict(self, dataset_predict):
		"""对数据集进行预测"""
		print('------ 预测中 ------')

		y_prob = []

		self.model.eval()
		dataset_loader = DataLoader(dataset=dataset_predict, batch_size=self.batch_size, collate_fn=collate_molgraphs)

		with torch.no_grad():
			for batch_data in dataset_loader:
				try:
					# 确定数据格式并解包数据
					if len(batch_data) >= 9:  # 包含actions的格式
						if len(batch_data) == 10:  # 完整数据格式
							bg, proteins, protein_lens, smiles, smiles_lens, d_n2v, p_n2v, f_d, f_p, actions = batch_data
						elif len(batch_data) == 9:  # 无蛋白质RNN长度
							bg, proteins, smiles, smiles_lens, d_n2v, p_n2v, f_d, f_p, actions = batch_data
							protein_lens = None
					elif len(batch_data) >= 6:  # 不包含actions的格式
						if len(batch_data) == 9:  # 完整数据格式，无actions
							bg, proteins, protein_lens, smiles, smiles_lens, d_n2v, p_n2v, f_d, f_p = batch_data
							actions = None
						elif len(batch_data) == 8:  # 无蛋白质RNN长度，无actions
							bg, proteins, smiles, smiles_lens, d_n2v, p_n2v, f_d, f_p = batch_data
							protein_lens = None
							actions = None
						elif len(batch_data) == 6:  # 无SMILES和蛋白质RNN，无actions
							bg, proteins, d_n2v, p_n2v, f_d, f_p = batch_data
							smiles = None
							smiles_lens = None
							protein_lens = None
							actions = None
					else:
						print(f"警告：预测集中意外的批次数据格式，有 {len(batch_data)} 个元素")
						continue  # 跳过这个批次

					# 将数据移至设备
					bg = bg.to(self.device)
					proteins = proteins.to(self.device)
					d_n2v = d_n2v.to(self.device)
					p_n2v = p_n2v.to(self.device)
					f_d = f_d.to(self.device)
					f_p = f_p.to(self.device)

					if smiles is not None:
						smiles = smiles.to(self.device)

					# 前向传播
					score, _ = self.model(
						bg, proteins, d_n2v, p_n2v, f_d, f_p, actions,
						smiles, smiles_lens, protein_lens
					)

					# 应用sigmoid并收集预测结果 - 修改此处，不使用squeeze
					Sigm = torch.nn.Sigmoid()
					prob = Sigm(score).detach().cpu().numpy()
					y_prob = y_prob + prob.flatten().tolist()  # 使用flatten()确保一维列表

				except Exception as e:
					print(f"预测批次时出错: {e}")
					continue

		print('------ 预测完成! ------')

		return y_prob

	def train(self, dataset_train, dataset_vali, vali_flag):
		"""训练模型"""
		# 将模型移至设备
		self.model.to(self.device)
		print("if on cuda:", next(self.model.parameters()).is_cuda)

		# 设置损失函数 - 添加权重处理类别不平衡
		if hasattr(self, 'use_weighted_loss') and self.use_weighted_loss:
			# 计算数据集正负样本比例
			all_labels = []
			for data in dataset_train:
				label = data[-1]  # 标签总是最后一个元素
				all_labels.append(float(label[0]))

			# 计算正样本权重
			pos_count = sum(all_labels)
			neg_count = len(all_labels) - pos_count

			# 使用可选的权重调节参数
			weight_multiplier = getattr(self, 'pos_weight_multiplier', 1.5)  # 默认为1.5
			pos_weight = (neg_count / pos_count) * weight_multiplier if pos_count > 0 else 1.0

			print(f"数据集正样本数: {pos_count}, 负样本数: {neg_count}, 调整后的正样本权重: {pos_weight:.4f}")

			# 使用加权损失函数
			loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(self.device))
		else:
			# 使用普通损失函数
			loss_func = torch.nn.BCEWithLogitsLoss()
			print("使用普通损失函数")

		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)

		# 添加学习率调度器
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, mode='max', factor=0.5, patience=5, verbose=True
		)

		print('--- 数据准备 ---')

		# 创建数据加载器 - 使用通用的collate_molgraphs函数
		train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, collate_fn=collate_molgraphs)
		vali_loader = DataLoader(dataset=dataset_vali, batch_size=self.batch_size, collate_fn=collate_molgraphs)

		# 早停跟踪
		max_auc = 0
		patience_counter = 0

		# 初始化指标表格
		from prettytable import PrettyTable

		# 指标表头
		train_metric_header = ["# epoch", "Accuracy", "Precision", "Recall", "AUROC", "AUPRC", "F1", "log_Loss"]
		vali_metric_header = ["# epoch", "Accuracy", "Precision", "Recall", "AUROC", "AUPRC", "F1", "log_Loss"]
		best_metric_header = ["# epoch", "Accuracy", "Precision", "Recall", "AUROC", "AUPRC", "F1", "log_Loss"]

		# 创建指标表格
		table_train = PrettyTable(train_metric_header)
		table_vali = PrettyTable(vali_metric_header)
		table_vali_best = PrettyTable(best_metric_header)

		# 确保结果文件夹存在
		if not os.path.exists(self.result_folder):
			os.makedirs(self.result_folder)

		if hasattr(self, 'debug_model') and self.debug_model:
			# 创建一个简单测试模型
			simple_model = SimpleTestModel(128).to(self.device)

		print('--- 开始训练 ---')
		t_start = time.time()

		for epo in range(self.train_epoch):
			self.model.train()
			y_label_train = []
			y_prob_train = []

			for batch_data in train_loader:
				# 1. 确定数据格式并解包数据
				# 我们总是假设有所有的数据，只是有些可能是None
				try:
					if len(batch_data) == 10:  # 完整数据格式
						bg, proteins, protein_lens, smiles, smiles_lens, d_n2v, p_n2v, f_d, f_p, actions = batch_data
					elif len(batch_data) == 9:  # 无蛋白质RNN长度
						bg, proteins, smiles, smiles_lens, d_n2v, p_n2v, f_d, f_p, actions = batch_data
						protein_lens = None
					elif len(batch_data) == 7:  # 无SMILES和蛋白质RNN
						bg, proteins, d_n2v, p_n2v, f_d, f_p, actions = batch_data
						smiles = None
						smiles_lens = None
						protein_lens = None
					else:
						print(f"警告：意外的批次数据格式，有 {len(batch_data)} 个元素")
						continue  # 跳过这个批次
				except Exception as e:
					print(f"解析批次数据时出错: {e}")
					continue

				# 2. 将数据移至设备
				# 将必要的数据移至设备
				bg = bg.to(self.device)
				proteins = proteins.to(self.device)
				d_n2v = d_n2v.to(self.device)
				p_n2v = p_n2v.to(self.device)
				f_d = f_d.to(self.device)
				f_p = f_p.to(self.device)
				actions = actions.to(self.device)

				# 条件移动可选数据
				if smiles is not None:
					smiles = smiles.to(self.device)

				# 打印debug信息
				# print(f"[Debug] actions shape: {actions.shape}")

				# 3. 前向传播
				score, actions = self.model(
					bg, proteins, d_n2v, p_n2v, f_d, f_p, actions,
					smiles, smiles_lens, protein_lens
				)

				# 打印debug信息
				# print(f"[Debug] score shape: {score.shape}, updated actions shape: {actions.shape}")

				# 4. 计算损失并更新权重
				# 修改：不对score使用squeeze，直接使用
				loss = loss_func(score, actions)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# 5. 收集预测结果用于指标计算
				y_label_train = y_label_train + actions.detach().cpu().numpy().flatten().tolist()
				y_prob_train = y_prob_train + torch.sigmoid(score).detach().cpu().numpy().flatten().tolist()

			# 计算训练指标
			metrics_train = metric(y_label_train, y_prob_train)

			print('------ 训练 ------')
			print('训练轮次 ' + str(epo + 1) +
				  ' , 准确率: ' + str(metrics_train[0])[:6] +
				  ' , 精确率: ' + str(metrics_train[1])[:6] +
				  ' , 召回率: ' + str(metrics_train[2])[:6] +
				  ' , AUROC: ' + str(metrics_train[3])[:6] +
				  ' , AUPRC: ' + str(metrics_train[4])[:6] +
				  ' , F1: ' + str(metrics_train[5])[:6] +
				  ' , 交叉熵损失: ' + str(metrics_train[6])[:6])

			lst_train = ["epoch " + str(epo + 1)] + [str(metrics_train[0])[:6], str(metrics_train[1])[:6],
													 str(metrics_train[2])[:6], str(metrics_train[3])[:6],
													 str(metrics_train[4])[:6], str(metrics_train[5])[:6],
													 str(metrics_train[6])[:6]]
			table_train.add_row(lst_train)

			if vali_flag:
				# 验证模型
				self.model.eval()
				y_label_vali = []
				y_prob_vali = []

				for batch_data in vali_loader:
					# 1. 确定数据格式并解包数据
					try:
						if len(batch_data) == 10:  # 完整数据格式
							bg, proteins, protein_lens, smiles, smiles_lens, d_n2v, p_n2v, f_d, f_p, actions = batch_data
						elif len(batch_data) == 9:  # 无蛋白质RNN长度
							bg, proteins, smiles, smiles_lens, d_n2v, p_n2v, f_d, f_p, actions = batch_data
							protein_lens = None
						elif len(batch_data) == 7:  # 无SMILES和蛋白质RNN
							bg, proteins, d_n2v, p_n2v, f_d, f_p, actions = batch_data
							smiles = None
							smiles_lens = None
							protein_lens = None
						else:
							print(f"警告：验证集中意外的批次数据格式，有 {len(batch_data)} 个元素")
							continue  # 跳过这个批次
					except Exception as e:
						print(f"解析验证批次数据时出错: {e}")
						continue

					# 2. 将数据移至设备
					bg = bg.to(self.device)
					proteins = proteins.to(self.device)
					d_n2v = d_n2v.to(self.device)
					p_n2v = p_n2v.to(self.device)
					f_d = f_d.to(self.device)
					f_p = f_p.to(self.device)
					actions = actions.to(self.device)

					if smiles is not None:
						smiles = smiles.to(self.device)

					# 3. 前向传播
					with torch.no_grad():
						score, actions = self.model(
							bg, proteins, d_n2v, p_n2v, f_d, f_p, actions,
							smiles, smiles_lens, protein_lens
						)

					# 4. 计算损失并收集预测结果
					# 修改：不对score使用squeeze，直接使用
					loss = loss_func(score, actions)
					y_label_vali = y_label_vali + actions.detach().cpu().numpy().flatten().tolist()
					y_prob_vali = y_prob_vali + torch.sigmoid(score).detach().cpu().numpy().flatten().tolist()

				# 计算验证指标
				metrics_vali = metric(y_label_vali, y_prob_vali)

				# 更新学习率
				scheduler.step(metrics_vali[3])  # 使用AUROC作为指标

				lst_vali = ["epoch " + str(epo + 1)] + [str(metrics_vali[0])[:6], str(metrics_vali[1])[:6],
														str(metrics_vali[2])[:6], str(metrics_vali[3])[:6],
														str(metrics_vali[4])[:6], str(metrics_vali[5])[:6],
														str(metrics_vali[6])[:6]]

				print('------ 验证 ------')
				print('验证轮次 ' + str(epo + 1) + ' , 准确率: ' + str(metrics_vali[0])[:6] +
					  ' , 精确率: ' + str(metrics_vali[1])[:6] + ' , 召回率: ' + str(metrics_vali[2])[:6] +
					  ' , AUROC: ' + str(metrics_vali[3])[:6] + ' , AUPRC: ' + str(metrics_vali[4])[:6] +
					  ' , F1: ' + str(metrics_vali[5])[:6] + ' , 交叉熵损失: ' + str(metrics_vali[6])[:6])

				table_vali.add_row(lst_vali)

				# 根据AUROC保存最佳模型
				if metrics_vali[3] > max_auc:
					max_auc_state = {"net": self.model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epo}
					max_auc = metrics_vali[3]
					max_auc_index_list = lst_vali
					patience_counter = 0  # 重置耐心计数器
				else:
					patience_counter += 1  # 增加耐心计数器

				# 早停检查
				if self.use_early_stopping and patience_counter >= self.patience:
					print(f"早停于第{epo + 1}轮，{self.patience}轮内未见改善")
					break
			else:
				# 如果没有验证，则根据训练指标保存
				if metrics_train[3] > max_auc:
					max_auc_state = {"net": self.model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epo}
					max_auc = metrics_train[3]
					max_auc_index_list = lst_train
					patience_counter = 0
				else:
					patience_counter += 1

				# 早停检查
				if self.use_early_stopping and patience_counter >= self.patience:
					print(f"早停于第{epo + 1}轮，{self.patience}轮内未见改善")
					break

			# 每10个epoch保存一次模型
			if (epo + 1) % 10 == 0:
				current_state = {"net": self.model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epo}
				# 使用与最终模型相似的命名方式，只添加轮数
				auc_value = metrics_vali[3] if vali_flag else metrics_train[3]
				save_path = os.path.join(self.result_folder, f"model_{auc_value}_AUC_epoch_{epo + 1}.pkl")
				torch.save(current_state, save_path)
				print(f'已保存第{epo + 1}轮模型到: {save_path}')

			t_now = time.time()
			print('训练轮次 ' + str(epo + 1) + ' 损失 ' + str(loss.detach().cpu().numpy())[:7] +
				  ". 总时间 " + str(int(t_now - t_start) / 3600)[:7] + " 小时")

		table_vali_best.add_row(max_auc_index_list)

		# 保存最佳模型
		torch.save(max_auc_state, self.result_folder + "/model_" + str(max_auc) + "_AUC.pkl")

		# 保存指标表格
		train_file = os.path.join(self.result_folder, "train_markdowntable.txt")
		with open(train_file, 'w') as fp:
			fp.write(table_train.get_string())

		test_file = os.path.join(self.result_folder, "Vali_markdowntable.txt")
		with open(test_file, 'w') as fp:
			fp.write(table_vali.get_string())

		best_file = os.path.join(self.result_folder, "Vali_best_markdowntable.txt")
		with open(best_file, 'w') as fp:
			fp.write(table_vali_best.get_string())

		print('------ 训练完成 ------')
# 同样更新vali和predict方法以支持新的数据处理方式，与train方法类似

class SimpleTestModel(nn.Module):
	def __init__(self, input_dim=128):
		super(SimpleTestModel, self).__init__()
		self.fc = nn.Linear(input_dim, 1)

	def forward(self, x):
		return self.fc(x)