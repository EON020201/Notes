# Transformer

## 注意力机制
**自注意力的输出与输入的张量形状相同，均为（批量大小batch_size，时间步的数目或词元序列的长度num_step，隐藏层大小hidden_size）。**

自注意力机制（Self-Attention）未显式引入非线性操作（如ReLU、Sigmoid等）的主要原因可以归结为以下几点：

---

### 1. **核心设计目标：关联建模而非非线性变换**
   自注意力机制的核心功能是**建模序列中元素之间的关联性**。它通过计算输入元素之间的相似度（点积注意力）来分配注意力权重，再通过加权求和整合全局信息。这一过程本质上是**线性组合**（线性变换+加权求和），目的是更高效地捕捉长距离依赖和上下文关系。  
   - **非线性并非主要需求**：自注意力层的目标是建立元素间的关系，而非对输入进行复杂的非线性映射。如果在此处引入非线性，可能会干扰对全局关系的直接建模。

---

### 2. **后续模块（FFN）负责非线性表达**
   Transformer模型通过**前馈神经网络（FFN）**显式引入非线性（如ReLU）。FFN通常由两个线性层和一个激活函数构成，紧跟在自注意力层之后，负责对自注意力输出的特征进行非线性变换和增强。  
   - **分工明确**：自注意力层专注于全局关联建模，FFN负责特征的非线性转换。这种分工使模型结构更清晰，各模块各司其职。

---

### 3. **Softmax的隐式非线性**
   自注意力机制虽然不显式使用激活函数，但其计算过程中包含**Softmax归一化**操作。Softmax本身是一个非线性函数，能够将注意力分数映射为概率分布。  
   - **有限的非线性**：Softmax提供了局部非线性，但整体自注意力层的表达能力仍以线性组合为主。这种设计平衡了计算效率和表达能力。

---

### 4. **多头注意力的隐式增强**
   多头注意力（Multi-Head Attention）通过并行多个独立的注意力头，将输入映射到不同的子空间，再拼接结果。虽然每个头的计算是线性的，但多头的组合相当于隐式引入了**多组线性变换的联合表达**。  
   - **表达能力提升**：多头机制通过不同子空间的投影，增强了模型的表达能力，部分弥补了线性操作的局限性。

---

### 5. **简化梯度传播**
   自注意力层的线性操作（矩阵乘法）具有简单的梯度计算规则，若加入非线性激活函数，可能会增加梯度消失或爆炸的风险（如Sigmoid的饱和区）。  
   - **稳定性考量**：保持线性操作有助于优化过程的稳定性，尤其是在深层Transformer模型中。

---

### 6. **参数效率与计算效率**
   自注意力层本身已包含大量参数（如Q/K/V的投影矩阵）。若加入非线性激活函数，可能需要额外增加参数（如全连接层的偏置项），导致模型复杂度上升。  
   - **轻量化设计**：省略非线性操作减少了计算量和内存占用，使模型更适合处理长序列。

---

### 总结
自注意力机制未显式引入非线性操作，是为了**更高效地建模全局关联性**，而将非线性表达能力交给后续的FFN模块。这种设计实现了以下平衡：
1. **全局关联建模**（自注意力）与**局部非线性变换**（FFN）的分工；
2. **计算效率**与**表达能力**的权衡；
3. **梯度传播稳定性**与**参数效率**的优化。

这种分工明确的架构使得Transformer模型在多种任务中表现出色，同时保持了较高的可扩展性。


### 缩放点积注意力
```
#@save
class DotProductAttention(nn.Block):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = npx.batch_dot(queries, keys, transpose_b=True) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```
### 掩蔽softmax操作
```
#@save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

## 多头注意力
多头注意力融合了来自于多个注意力汇聚的不同知识，这些知识的不同来源于相同的查询、键和值的不同的子空间表示。
```
#@save
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

#@save
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


#@save
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```
## 自注意力与位置编码
### 位置编码
```
#@save
class PositionalEncoding(nn.Block):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P, P的长度max_len对应输入X的第2维度时间步数或序列长度
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X) # [0::2]表示数组切片操作，从0开始，未设置结束索引，步长为2
        self.P[:, :, 1::2] = np.cos(X) # 与上同理

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```
## Transformer
![1](transformer.jpg)
### nn.embedding
`nn.Embedding` 是 PyTorch 中用于处理序列数据中的词嵌入（word embeddings）的核心模块。它本质上是一个查找表，将输入的离散型数据（通常是整数形式的单词索引）映射为连续型的数据表示（即词向量）。这种转换在自然语言处理（NLP）、推荐系统等领域中非常常见。

### 函数解释

当你调用 `nn.Embedding(vocab_size, num_hiddens)` 时，你正在初始化一个嵌入层，其中：

- `vocab_size`：这是你的词汇表大小，也就是你希望这个嵌入层能够支持的最大单词索引值加一（因为索引是从0开始的）。例如，如果你有一个包含10,000个不同单词的词汇表，那么 `vocab_size` 应该设置为10,000。
- `num_hiddens`：这是每个单词对应的嵌入向量的维度。这个值决定了每个单词被表示为一个多维空间中的点，其坐标数量就是 `num_hiddens` 的值。通常，这个值可以根据具体任务和模型的需求来选择，比如50、100或300等。

除了这两个主要参数之外，`nn.Embedding` 还接受其他一些可选参数，如 `padding_idx`、`max_norm` 等，这些参数可以用来控制嵌入层的行为，比如指定填充标记的索引，或者限制嵌入向量的最大范数等。

### 示例代码

下面是一个简单的例子，展示了如何使用 `nn.Embedding` 来创建一个嵌入层，并将一批单词索引转换为对应的词向量：

```python
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
```
上述代码输出
```
权重:
 Parameter containing:
tensor([[ 1.2475,  0.2461, -0.1228],
        [ 0.5988, -2.0277, -1.4456],
        [ 1.2011,  0.2131, -0.9624],
        [ 1.2717,  1.7339,  1.2558],
        [-0.3740,  2.0479,  0.6131]], requires_grad=True)
输出:
tensor([[[ 0.5988, -2.0277, -1.4456],
         [ 1.2011,  0.2131, -0.9624],
         [-0.3740,  2.0479,  0.6131],
         [ 1.2717,  1.7339,  1.2558]],

        [[-0.3740,  2.0479,  0.6131],
         [ 1.2717,  1.7339,  1.2558],
         [ 1.2011,  0.2131, -0.9624],
         [ 0.5988, -2.0277, -1.4456]]], grad_fn=<EmbeddingBackward0>)
```
在这个例子中，`embedding` 层被初始化为一个10x3的矩阵，意味着它可以表示最多10个不同的单词，每个单词由一个三维向量表示。输入 `x` 是一个二维张量，包含了两个句子的单词索引，形状为 `[2, 4]`，即每句话有4个单词。通过 `embedding(x)` 操作，我们可以得到一个新的张量 `y`，其形状为 `[2, 4, 3]`，表示每个句子中的每个单词都被替换成了相应的3维词向量。

值得注意的是，在实际应用中，`nn.Embedding` 层通常作为神经网络的一部分，与其他层（如RNN、LSTM或Transformer等）一起训练，以学习到更有效的词表示。此外，有时我们会使用预训练的词向量（如Word2Vec或GloVe），并通过设置 `_weight` 参数将其加载到 `nn.Embedding` 中，同时设置 `requires_grad=False` 来固定这些预训练的词向量不参与后续的训练过程。

总之，`nn.Embedding` 是构建深度学习模型特别是涉及文本处理任务的重要组件之一。正确地配置和利用它可以极大地提升模型对文本数据的理解能力。
### 嵌入表示缩放
由于这里使用的是值范围在-1和1之间的固定位置编码，因此通过学习得到的输入的嵌入表示的值需要先乘以嵌入维度的平方根进行重新缩放，然后再与位置编码相加。
```
# 因为位置编码值在-1和1之间，
# 因此嵌入值乘以嵌入维度的平方根进行缩放，
# 然后再与位置编码相加。
X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
```
### 解码器
Transformer解码器也是由多个相同的层组成。在DecoderBlock类中实现的每个层包含了三个子层：解码器自注意力、“编码器-解码器”注意力和基于位置的前馈网络。这些子层也都被残差连接和紧随的层规范化围绕。

在掩蔽多头解码器自注意力层（第一个子层）中，查询、键和值都来自上一个解码器层的输出。关于序列到序列模型（sequence-to-sequence model），在训练阶段，其输出序列的所有位置（时间步）的词元都是已知的；然而，在预测阶段，其输出序列的词元是逐个生成的。因此，在任何解码器时间步中，只有生成的词元才能用于解码器的自注意力计算中。**为了在解码器中保留自回归的属性，其掩蔽自注意力设定了参数dec_valid_lens，以便任何查询都只会与解码器中所有已经生成词元的位置（即直到该查询位置为止）进行注意力计算。（只在训练时生效，推理时无需使用）**

#### dec_valid_lens
```
if self.training:
    batch_size, num_steps, _ = X.shape
    # dec_valid_lens的开头:(batch_size,num_steps),
    # 其中每一行是[1,2,...,num_steps]
    dec_valid_lens = torch.arange(
        1, num_steps + 1, device=X.device).repeat(batch_size, 1)
else:
    dec_valid_lens = None
```

#### state
state共有3个值，第1个和第2个值分别为编码器输出和编码器有效长度，训练和推理时一致；第3个值为解码器块中第1个掩码多头注意力层的key和value，在训练时，由于输出序列的所有词元同时处理，因此其在训练过程中始终与解码器输入X保持一致；在推理预测阶段，输出序列是通过词元一个接着一个解码的，因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示，具体表现为其第2维度与当前时间步保持一致
```
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
```