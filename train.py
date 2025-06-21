import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# 需要在preprocess.py文件顶部添加以下导入
from embeddings import get_embedding, train_all_embeddings, load_embeddings
# 从preprocess.py中导入必要的函数
from preprocess import preprocess_data, load, dump, protein_tokens_struct
# 确保导入蛋白质指纹模块
from protein_fingerprint import generate_protein_fingerprint

import warnings

warnings.filterwarnings("ignore", message="dropout option adds dropout after all but last recurrent layer")

# 导入修改后的模块
from DrugAI_code import DrugAI, tokens_struct


def main():
    # 配置模型参数
    config = {
        # 嵌入配置
        "embed_type": "deepwalk",  # 嵌入类型选择: "n2v", "deepwalk", "line", "sdne"

        # 特征组合配置
        "views_flag": "d_GNN+d_N2V+d_fp+g_CNN+p_n2v+g_fp",

        # GNN类型配置
        "GNNs_flag": "GCN",  # 可选: "AttentiveFP", "GCN", "GAT", "MPNN"

        # 蛋白质编码器参数
        "protein_Oridim": 20,
        "feature_size": 128,
        "out_features": 128,
        "max_seq_len": 3000,
        "kernels": [3, 5, 7],
        # [3, 5, 7]
        "CNN_fdim": 128,
        "g_fp_dim": 343,  # 蛋白质指纹特征维度
        "p_n2v_fdim": 256,

        # 蛋白质RNN参数
        "protein_embed_dim": 64,
        "protein_blstm_dim": 64,
        "protein_layers": 2,
        "protein_RNN_fdim": 128,

        # 药物编码器参数
        "GNNs_layers": 1,
        #1，2，3，4
        "GNNs_heads": None,
        "GNN_fdim": 167,
        "d_fp_dim": 167,  # 药物指纹特征维度
        "d_n2v_fdim": 256,

        # SMILES处理参数
        "smiles_embed_dim": 64,
        "smiles_blstm_dim": 64,
        "smiles_layers": 2,
        "RNN_fdim": 128,

        # 通用模型参数
        "dropout_r": 0.2,
        "DNN_layers": [512, 256, 128, 64],
        # [512, 256, 128, 64]
        # 训练参数
        "batch_size": 128,
        "lr": 4e-4,
        "decay": 1e-5,
        "train_epoch": 40,

        # 早停和类别不平衡处理
        "use_early_stopping": True,
        "patience": 10,
        "use_weighted_loss": True,

        # 嵌入训练参数
        "force_retrain_embeddings": False  # 是否强制重新训练嵌入
    }

    # 根据配置设置结果文件夹
    config["result_folder"] = f"./results_{config['GNNs_flag']}_{config['embed_type']}"

    # 数据文件路径
    csv_file = "Dataset-of-activating-and-inhibiting-mechanisms.csv"

    try:
        # 先训练或加载嵌入向量
        print(f"准备{config['embed_type']}嵌入向量...")
        all_embeddings = train_all_embeddings(
            csv_file,
            embed_dim=config['d_n2v_fdim'],
            force_retrain=config['force_retrain_embeddings']
        )

        # 根据嵌入类型创建保存路径
        save_path = f"dataset_train_vali_test_{config['embed_type']}_with_protein_fingerprint.csv"

        # 检查是否有预处理好的数据
        if os.path.exists(save_path):
            print(f"检测到已处理的数据文件 {save_path}，直接加载...")
            try:
                data = load(save_path)

                # 适应不同版本的保存格式
                if isinstance(data, tuple) and len(data) >= 3:
                    if len(data) == 4:  # 新格式包含embed_type
                        dataset_train, dataset_vali, dataset_test, saved_embed_type = data
                        if saved_embed_type != config["embed_type"]:
                            print(
                                f"警告: 已保存的嵌入类型({saved_embed_type})与配置的类型({config['embed_type']})不匹配!")
                            print("重新开始数据预处理...")
                            dataset_train, dataset_vali, dataset_test = preprocess_data(
                                csv_file, save_path, config["embed_type"], all_embeddings)
                        else:
                            print("数据加载成功!")
                    else:
                        dataset_train, dataset_vali, dataset_test = data
                        print("数据加载成功，但未检测到嵌入类型信息。假设为n2v。")
                        if config["embed_type"] != "n2v":
                            print(f"配置的嵌入类型是{config['embed_type']}，需要重新预处理数据...")
                            dataset_train, dataset_vali, dataset_test = preprocess_data(
                                csv_file, save_path, config["embed_type"], all_embeddings)
                else:
                    raise ValueError("数据格式不正确")
            except Exception as e:
                print(f"加载数据失败: {e}")
                print("重新开始数据预处理...")
                dataset_train, dataset_vali, dataset_test = preprocess_data(
                    csv_file, save_path, config["embed_type"], all_embeddings)
        else:
            print(f"未检测到预处理数据文件 {save_path}，开始数据预处理...")
            dataset_train, dataset_vali, dataset_test = preprocess_data(
                csv_file, save_path, config["embed_type"], all_embeddings)

        # 检查数据集大小
        if not dataset_train or len(dataset_train) == 0:
            raise ValueError("训练集为空")
        if not dataset_vali or len(dataset_vali) == 0:
            raise ValueError("验证集为空")
        if not dataset_test or len(dataset_test) == 0:
            raise ValueError("测试集为空")

        print('数据加载完成!!!')
        print(f"训练集大小: {len(dataset_train)}")
        print(f"验证集大小: {len(dataset_vali)}")
        print(f"测试集大小: {len(dataset_test)}")

        # 确保结果文件夹存在
        if not os.path.exists(config["result_folder"]):
            os.makedirs(config["result_folder"])

        # 创建模型
        print(
            f"开始创建DrugAI模型，使用GNN: {config['GNNs_flag']}, 嵌入: {config['embed_type']}, 特征组合: {config['views_flag']}...")

        # 实例化模型
        model = DrugAI(**config)

        # 确保模型和张量明确地移动到CUDA
        use_cuda = torch.cuda.is_available()
        model.device = torch.device('cuda' if use_cuda else 'cpu')
        model.model.to(model.device)

        print(f"模型创建完成，开始训练... 使用设备: {model.device}")
        if use_cuda:
            print(f"模型是否在CUDA上: {next(model.model.parameters()).is_cuda}")
        else:
            print("CUDA不可用，使用CPU")

        # 训练模型
        model.train(dataset_train, dataset_vali, vali_flag=True)

        # 在测试集上评估模型
        print("训练完成，开始在测试集上评估模型...")
        accuracy, precision, recall, auc, auprc, f1, loss, y_pred, y_prob, y_label = model.vali(dataset_test)

        # 打印结果
        print('------ 测试结果 ------')
        print(f'嵌入类型: {config["embed_type"]}')
        print('准确率: ', accuracy)
        print('精确率: ', precision)
        print('召回率: ', recall)
        print('AUROC: ', auc)
        print('AUPRC: ', auprc)
        print('F1: ', f1)
        print('损失: ', loss)

        # 构建结果文件路径
        result_file_path = os.path.join(config["result_folder"], "test.txt")

        # 将结果写入文件
        with open(result_file_path, 'w', encoding='utf-8') as f:
            f.write('------ 测试结果 ------\n')
            f.write(f'嵌入类型: {config["embed_type"]}\n')
            f.write(f'特征组合: {config["views_flag"]}\n')
            f.write(f'GNN类型: {config["GNNs_flag"]}\n')
            f.write(f'准确率: {accuracy}\n')
            f.write(f'精确率: {precision}\n')
            f.write(f'召回率: {recall}\n')
            f.write(f'AUROC: {auc}\n')
            f.write(f'AUPRC: {auprc}\n')
            f.write(f'F1: {f1}\n')
            f.write(f'损失: {loss}\n')

        print(f"测试结果已保存到 {result_file_path}")

        # 绘制ROC曲线和PR曲线
        from sklearn.metrics import roc_curve, precision_recall_curve
        import matplotlib.pyplot as plt

        # ROC曲线
        fpr, tpr, _ = roc_curve(y_label, y_prob)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {config["embed_type"]}')
        plt.legend()
        plt.savefig(os.path.join(config["result_folder"], 'roc_curve.png'))

        # PR曲线
        precision_curve, recall_curve, _ = precision_recall_curve(y_label, y_prob)
        plt.figure(figsize=(10, 8))
        plt.plot(recall_curve, precision_curve, label=f'PR (AUPRC = {auprc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {config["embed_type"]}')
        plt.legend()
        plt.savefig(os.path.join(config["result_folder"], 'pr_curve.png'))

        print("训练和评估完成!")

    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("尝试使用更简单的配置重新运行，或检查数据文件和依赖项。")


if __name__ == "__main__":
    main()

