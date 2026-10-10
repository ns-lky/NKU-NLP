# NKU-NLP

本仓库实现《作业一：文本分类实现》的三个实验任务。实验统一使用 `HW-1/nyt.csv` 的分层 80%/10%/10% 划分（随机种子默认为 42），并在测试集上输出 Accuracy 与 Macro-F1。数据文件不会提交到 GitHub；运行时请将它们放在 `HW-1/` 目录。

## 环境

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Task 1：Bag of Words

```bash
python -m scripts.task1_bow --nyt HW-1/nyt.csv --output outputs/task1_bow.json
```

脚本依次训练 Binary BoW、Word Frequency BoW 和 TF-IDF 三个 Logistic Regression 模型。词表只在训练集拟合，验证集和测试集仅执行转换。

## Task 2：Word Embedding

从 Stanford 下载 `glove.6B.zip` 后解压，使用其中的 `glove.6B.100d.txt`：

```bash
python -m scripts.task2_embeddings \
  --nyt HW-1/nyt.csv --ag HW-1/ag.csv \
  --glove-path path/to/glove.6B.100d.txt \
  --output outputs/task2_embeddings.json
```

脚本完成三组实验：预训练 GloVe、在 AG News 上训练的 Word2Vec、在 NYT 训练集上训练的 Word2Vec。每篇文档取已知词向量的平均作为 100 维表示，再训练 Logistic Regression。也可用 `--method glove`、`--method ag-word2vec` 或 `--method nyt-word2vec` 单独运行。

## Task 3：BERT

首次运行会从 Hugging Face 下载 `google-bert/bert-base-uncased`（需要网络和足够磁盘空间）：

```bash
python -m scripts.task3_bert \
  --nyt HW-1/nyt.csv \
  --output outputs/task3_bert.json \
  --output-dir models/task3_bert
```

默认严格使用 `max_length=64`、`epochs=3`。可用 `--device cpu` 或 `--device cuda` 指定设备；`--max-train-samples` 和 `--max-eval-samples` 仅用于快速冒烟测试，正式结果不要设置。

三个脚本都会打印验证集和测试集指标，并将完整配置及结果写入 `outputs/`。
