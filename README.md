# NKU-NLP

本仓库实现《作业一：文本分类实现》的三个实验任务。统一将 NYT 随机打乱并按类别分层划分为训练集 9,215 条、验证集 1,152 条、测试集 1,152 条（80%/10%/10%，随机种子 42）。三个任务共享同一函数及同一组有序行号。`results/split.json` 保存数据文件 SHA-256、准确划分和各类别数量，便于核对实验可比性。

请将作业提供的原始 `nyt.csv`、`ag.csv` 放在 `HW-1/`，保持原始内容和顺序。原始数据、下载缓存、训练模型保存在本地；完整测试指标和逐条预测保存在可提交的 `results/`。本阶段只完成实验，实验报告另按提供的模板编写。

## 环境

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13.2 为本次运行环境。依赖版本已固定。使用支持 CUDA 12.6 的 NVIDIA 显卡时，先在虚拟环境中安装 GPU 版 PyTorch，再安装其他依赖：

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --cache-dir artifacts/pip-cache
python -m pip install -r requirements.txt --cache-dir artifacts/pip-cache
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

运行命令均在仓库根目录执行。以下单行命令同时适用于 PowerShell 和 bash。

## Task 1：Bag of Words

```bash
python -u -m scripts.task1_bow
```

脚本依次训练 Binary BoW、Word Frequency BoW 和 TF-IDF 三个 Logistic Regression 模型。词表只在训练集拟合，验证集和测试集仅执行转换。默认使用完整词表（66,180 个 token），`C=1`、`solver=liblinear`、`max_iter=1000`，并保存三组向量器和分类器到 `models/task1_bow/`。

原始作业文档写明“三种” BoW，却只展开说明 Binary BoW 和 Word Frequency 两种；保留这两项原要求，并补充 TF-IDF 作为第三种比较，文档没有将 TF-IDF 单独列出。

完整 NYT 测试集运行结果：

| 表示方法 | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Binary BoW | 0.989583 | 0.974058 |
| Word Frequency | 0.992188 | 0.980103 |
| TF-IDF | 0.986111 | 0.969559 |

## Task 2：Word Embedding

从 Stanford 下载 `glove.6B.zip` 后解压，使用其中的 `glove.6B.100d.txt`：

```bash
python -u -m scripts.task2_embeddings --glove-path artifacts/glove.6B.100d.txt
```

脚本完成三组实验：预训练 GloVe、在完整 90,000 条 AG News 上训练的 Word2Vec、在 NYT 的全部 9,215 条训练文本上训练的 Word2Vec。NYT 验证集和测试集不参与词向量训练。所有向量严格为 100 维，每篇文档对已知词的每次出现取平均，未知词不参与分母；全部未知的文本使用零向量，再训练 Logistic Regression。

Word2Vec 使用 Skip-gram、window=5、min_count=2、epochs=5、seed=42。默认单线程和稳定哈希，便于重复运行。作业未指定 Word2Vec 训练轮次，这里选择 5 轮；BERT 的 3 轮要求则严格固定。词向量模型、分类器保存到 `models/task2_embeddings/`。也可用 `--method glove`、`--method ag-word2vec` 或 `--method nyt-word2vec` 单独运行。

## Task 3：BERT

首次运行会从 Hugging Face 下载 `google-bert/bert-base-uncased`（需要网络和足够磁盘空间）：

```bash
python -u -m scripts.task3_bert
```

正式脚本仅接受指定的 `google-bert/bert-base-uncased`、`max_length=64`、`epochs=3`，不提供缩小样本的参数。全部编码长度为 64（含特殊 token），训练完整 3 轮；分类头为 NYT 的 3 类重新初始化，整个模型参与微调。使用 AdamW、学习率 2e-5、batch size 16、梯度裁剪 1.0、10% warmup 和线性学习率衰减。CUDA 下默认 fp16 自动混合精度，也可通过 `--fp32` 禁用。

每轮验证结束保存训练历史和可恢复检查点；可用 `python -u -m scripts.task3_bert --resume` 从已完成的 epoch 继续。完成三轮后，根据验证集 Macro-F1 选择模型，再在完整测试集上评价。最佳模型和 tokenizer 保存到 `models/task3_bert/`；所有 Hugging Face 缓存位于工作区 `artifacts/huggingface/`。

三个脚本都会输出 Accuracy、Macro-F1、各类别 Precision/Recall/F1、混淆矩阵、真实参数和依赖版本。对应 JSON 为 `results/task1_bow.json`、`results/task2_embeddings.json`、`results/task3_bert.json`；`results/predictions/` 保存有序的验证集和测试集逐条预测（行号、真实标签、预测标签，不含新闻正文）。
