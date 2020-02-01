# -*- coding: utf-8 -*-

import json
import random
import numpy as np
from gensim import models
import sys
import os
sys.path.append(os.path.dirname(__file__))


SONGS_PATH = os.path.join(os.path.dirname(__file__), "data/songs_cut.json")
W2V_PATH = os.path.join(os.path.dirname(__file__), "data/lyricist_w2v.bin")
WORDS_PATH = os.path.join(os.path.dirname(__file__), "data/words.json")


class AttnDataSet(object):
    def __init__(self, batch_size=32, max_length=16, topn=None,
                 corpus_path=SONGS_PATH, w2v_path=W2V_PATH, words_path=WORDS_PATH):
        """
        :param batch_size: 每个batch的语料数量大小
        :param max_length: 单个句子的最大长度
        :param topn: 保留频率最高的topn个词
        """
        self.max_length = max_length
        self.batch_size = batch_size
        self.topn = topn
        self.cursor = 0     # 用于标记当前的batch处在语料库中的位置
        self.epoch = 0      # 用于记录当前的循环次数epoch
        self.corpus = self.load_corpus(corpus_path)     # 包含所有句子的语料库
        self.w2v_model = models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)     # Word2Vec模型
        self.word_bag, self.word_dict = self.load_word_bags(words_path)         # 词袋

    def get_next_batch(self):
        """
        获取下一个batch
        :returns:
            X_encoder 前句, attention_machine的输入, shape为(batch_size, max_length, vector_size)
            X_decoder 后句, decoder的输入, shape为(batch_size, max_length, vector_size)
            encoder_length 前句的长度, shape为(batch_size)
            decoder_length 后句的长度, shape为(batch_size)
            y 神经网络的目标, shape为(batch_size, max_length)
            z_encoder 词语在前句中的位置, shape为(batch_size, max_length)
            z_decoder 词语在后句中的位置, shape为(batch_size, max_length)
                1:当前词是句首,
                2:当前词是倒数第三个词,
                3:当前词是倒数第二的词
                4:当前词是句末, 其后应接句末标识符<EOS>
                0:其他
            mask 有效词语mask, 在计算seq_loss时可以更精确, 让模型更快收敛, shape为(batch_size, max_length)
        """
        if self.cursor + self.batch_size > len(self.corpus):
            self.cursor = 0
            self.epoch += 1
        sentences = self.corpus[self.cursor:self.cursor + self.batch_size]  # 取batch_size组句子
        self.cursor += self.batch_size
        encoder_length = [len(s[0]) for s in sentences]    # 统计每个句子的长度
        decoder_length = [len(s[1]) for s in sentences]

        X_encoder, X_decoder, y = [], [], []
        for s in sentences:
            X_encoder.append(self.sentence2seq(s[0]))      # 用word2vec编码为词向量
            X_decoder.append(self.sentence2seq(s[1], go=True))      # 用word2vec编码为词向量
            y.append(np.expand_dims(self.sentence_encode(s[1]), 0))  # 编码为其在词袋中编号的格式

        X_encoder = np.concatenate(X_encoder)
        X_decoder = np.concatenate(X_decoder)
        y = np.concatenate(y)
        y = y.astype(np.int32)  # 转换格式, 否则后续计算seq_loss会报错

        z_encoder = np.zeros((self.batch_size, self.max_length), dtype=np.int32)    # 根据词语在句中的位置生成
        z_decoder = np.zeros((self.batch_size, self.max_length), dtype=np.int32)    # 根据词语在句中的位置生成
        for k, v in enumerate(encoder_length):
            if v == 1:
                z_encoder[k, 0] = 4
            elif v == 2:
                z_encoder[k, 0], z_encoder[k, 1] = 3, 4
            elif v == 3:
                z_encoder[k, 0], z_encoder[k, 1], z_encoder[k, 2] = 2, 3, 4
            elif v >= 4:
                z_encoder[k, 0], z_encoder[k, v-3], z_encoder[k, v-2], z_encoder[k, v-1] = 1, 2, 3, 4

        for k, v in enumerate(decoder_length):
            if v == 1:
                z_decoder[k, 1] = 4
            elif v == 2:
                z_decoder[k, 1], z_decoder[k, 2] = 3, 4
            elif v == 3:
                z_decoder[k, 1], z_decoder[k, 2], z_decoder[k, 3] = 2, 3, 4
            elif v >= 4:
                z_decoder[k, 1], z_decoder[k, v-2], z_decoder[k, v-1], z_decoder[k, v] = 1, 2, 3, 4

        encoder_length = np.array(encoder_length, dtype=np.int32)
        decoder_length = np.array(decoder_length, dtype=np.int32) + 1

        mask = np.zeros_like(y, dtype=np.float32)  # 生成mask, 用于
        for i in range(self.batch_size):
            mask[i, :decoder_length[i]] = 1.0
        # 因为<PAD>和<UKN>在句中出现的频率很高
        # 如果不把它们的权重置零, 得到的模型会倾向于生成更多的<PAD>和<UKN>, 这显然不是我们想要的
        mask[np.where(y == 2)] = 0  # <PAD>不参与loss计算
        mask[np.where(y == 3)] = 0  # <UKN>不参与loss计算

        return X_encoder, X_decoder, encoder_length, decoder_length, y, z_encoder, z_decoder, mask

    def sentence2seq(self, sentence, go=None, eos=True, pad=True, ukn=True):
        """
        将句子中的词语依次用word2vec模型编码为Vector
        :param sentence: 分词后的list格式, 如 ["故事", "的", "小黄花"]
        :return result: np.Array格式, shape为(1, max_length, vector_size)
        """
        sentence = self.fill(sentence, go, eos, pad, ukn)
        result = []
        for w in sentence:
            result.append(self.w2v_model[w])
        result = np.expand_dims(np.concatenate([np.expand_dims(w, 0) for w in result]), 0)
        return result

    def fill(self, sentence, go=None, eos=True, pad=True, ukn=True):
        """
        将句子填充上各种标志位, 并且补全为长度为max_length的句子
        :param sentence: 分词后的list格式, 如 ["故事", "的", "小黄花"]
        :param go: 是否添加<GO>标识
        :param eos: 是否添加<EOS>标识
        :param pad: 是否添加<PAD>标识
        :param ukn: 是否添加<UKN>标识
        """
        sentence = sentence.copy()
        if go:
            sentence = (['<GO>'] + sentence)
        if eos:
            sentence.append('<EOS>')
        if ukn:
            for key, w in enumerate(sentence.copy()):
                if w not in self.w2v_model:     # 所有不在word2vec模型中的词语都会被替换为<UKN>, 否则会报错
                    sentence[key] = '<UKN>'
        if pad and len(sentence) < self.max_length:
            sentence += ['<PAD>'] * (self.max_length - len(sentence))
        elif len(sentence) > self.max_length:
            sentence = sentence[:self.max_length]
        return sentence

    def sentence_encode(self, sentence):
        """
        句子编码, 将词语依次编码为其在词袋中编号的格式, 会将句子添加<EOS>标识, 并自动补齐为max_length
        :param sentence: 分词后的list格式,
                    如 ["故事", "的", "小黄花"]
        :return sentence: list格式,
                    如 [295, 4, 2, 1, 3, 3, 3, 3......]
        """
        sentence = sentence.copy()
        sentence = self.fill(sentence)
        for key, w in enumerate(sentence.copy()):
            sentence[key] = self.word_dict.get(w, self.word_dict['<UKN>'])  # 不在词袋中的词语一律填充为<UKN>
        return sentence

    def sentence_decode(self, embed):
        """
        句子解码, 将编码后的词语依次解码成中文
        :param embed: list格式,
                如 [6, 7, 4, 1, 3, 3]
        :return sentence: list格式,
                如 ['你', '在', '的', '<EOS>']

        """
        sentence = []
        for key in embed:
            word = self.word_bag[key] if key < len(self.word_bag) else '<UKN>'
            sentence.append(word)
            if word == '<EOS>':     # 遇到结尾标识<EOS>自动停止解码
                break
        return sentence

    def load_word_bags(self, file_path):
        """
        加载词典, 保留词频最高的前topn个词语, 并添加如下几个特殊标识符:
            <GO>:句子开始标识
            <EOS>:句子结尾标识
            <UKN>:未知词标识(词语不在词袋里)
            <PAD>:将句子填充为固定长度max_length
        :return word_bag: 词袋列表, 用于根据编号索引词语, 用于句子的解码,
                    如 word_bag[72] == "真"
                word_dict: 词袋列表的字典化, 用于依据词语反向相应的编号, 用于训练模型时的句子编码,
                    如 word_dict["真"] == 72

        """
        with open(file_path, 'r', encoding='utf8') as f:
            word_bag = json.loads(f.read())
        word_bag = ["<GO>", "<EOS>", "<UKN>", "<PAD>"] + word_bag
        if self.topn and isinstance(self.topn, int):
            # word_bag中的词语已按词频进行排序
            word_bag = word_bag[:self.topn]
        word_dict = dict(zip(word_bag, range(len(word_bag))))
        return word_bag, word_dict

    def load_corpus(self, file_path):
        """
        加载语料库, 只保留长度小于max_length的句子, 并对所有句子进行乱序
        :return corpus: 一个包含所有句子的列表
        """
        with open(file_path, 'r', encoding='utf8') as f:
            data = json.loads(f.read())
        corpus = []
        for name, lyric in data.items():
            for i in range(len(lyric) - 1):
                if len(lyric[i]) < self.max_length and len(lyric[i+1]) < self.max_length:
                    corpus.append((lyric[i], lyric[i+1]))
        random.shuffle(corpus)  # 乱序
        return corpus
