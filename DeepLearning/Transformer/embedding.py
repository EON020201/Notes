import torch
import torch.nn as nn

# 初始化一个嵌入层，假设词汇表大小为10，每个单词的嵌入维度为3
embedding = nn.Embedding(5, 3)

# 创建一批输入数据，这里我们有两个句子，每个句子有4个单词，单词索引分别为[1, 2, 4, 5]和[4, 3, 2, 9]
x = torch.LongTensor([[1, 2, 4, 3], [4, 3, 2, 1]])

# 使用嵌入层获取对应的词向量
y = embedding(x)

print('权重:\n', embedding.weight)
print('输出:')
print(y)