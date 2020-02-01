# batch_size = 64          # 训练
batch_size = 1          # 测试
max_length = 32         # 句子的最大长度
vocab_size = 20000      # 词袋大小(保留词频最高的词语数量)

vector_size = 200       # 词向量维度
embedding_size = 25     # 位置标识的向量维度
hidden_size = vector_size + embedding_size  # LSTM维度
signal_size = 5         # 位置标识的数量

'''训练参数'''
learningRateBase = 5e-4             # 初始学习率
learningRateDecayStep = 10000       # 衰减步长
learningRateDecayRate = 0.98        # 衰减率
total_epoch = 20
