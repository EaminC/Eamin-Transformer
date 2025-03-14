import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 设定随机种子，保证结果可复现
torch.manual_seed(42)

# 创建一个词典大小为 10，每个词的向量维度为 2 的嵌入层
embedding = nn.Embedding(10, 2)  

# 生成索引 0~9
indices = torch.arange(10)  # [0, 1, 2, ..., 9]
embeddings = embedding(indices).detach().numpy()  # 转换为 NumPy 以便可视化

# 绘制点
plt.figure(figsize=(6, 6))
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=range(10), cmap='tab10', s=100)

# 添加文本标签
for i, (x, y) in enumerate(embeddings):
    plt.text(x, y, str(i), fontsize=12, ha='right', va='bottom')

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("2D Visualization of nn.Embedding (10, 2)")
plt.grid()
plt.show()