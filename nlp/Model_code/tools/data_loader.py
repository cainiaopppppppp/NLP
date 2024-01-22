import random
import numpy as np
import os
import sys
import re

import torch
from torch.autograd import Variable

import tools.utils as utils
from download import *

class DataLoader(object):
    def __init__(self, data_dir, embedding_file, word_emb_dim, max_len=100, pos_dis_limit=50,
                 pad_word='<pad>', unk_word='<unk>', other_label='Other', gpu=0):
        self.data_dir = data_dir  #数据集根目录
        self.embedding_file = embedding_file  #词向量文件目录
        self.max_len = max_len  #可接受的最大句子长度
        self.word_emb_dim = word_emb_dim
        self.limit = pos_dis_limit  #为避免超长时记忆，所设定的最大窗口，窗口中心是目标实体
        self.pad_word = pad_word  #填充词
        self.unk_word = unk_word  #UNK
        self.other_label = other_label  # 范围以外的关系类型
        self.gpu = gpu

        self.word2idx = dict() #词转化为id的字典
        self.label2idx = dict()#关系类型转化为id的字典

        self.embedding_vectors = list()  #词向量的id与word2idx的id一致
        self.unique_words = list()  # 数据集中所有出现的词

        self.original_words_num = 0
        self.lowercase_words_num = 0
        self.zero_digits_replaced_num = 0
        self.zero_digits_replaced_lowercase_num = 0
        
        if pad_word is not None:
            self.pad_idx = len(self.word2idx)  #默认从0开始
            self.word2idx[pad_word] = self.pad_idx
            self.embedding_vectors.append(utils.generate_zero_vector(self.word_emb_dim))

        if unk_word is not None:
            self.unk_idx = len(self.word2idx)  #默认从1开始
            self.word2idx[unk_word] = self.unk_idx
            self.embedding_vectors.append(utils.generate_random_vector(self.word_emb_dim))
        
        vocab_path = os.path.join(self.data_dir, 'words.txt')
        with open(vocab_path, 'r') as f:
            for line in f:
                self.unique_words.append(line.strip())
        
        labels_path = os.path.join(data_dir, 'labels.txt')
        with open(labels_path, 'r') as f:
            for i, line in enumerate(f):
                self.label2idx[line.strip()] = i

        other_label_idx = self.label2idx[self.other_label]
        self.metric_labels = list(self.label2idx.values())
        self.metric_labels.remove(other_label_idx)

    def tensor_ensure_gpu(self, tensor):
        """如果使用GPU的话，将Tensor切换到GPU模式"""
        if self.gpu >= 0:
            return tensor.cuda(device=self.gpu)
        else:
            return tensor

    def get_loaded_embedding_vectors(self):
        """将词向量读入为tensor形式"""
        return torch.FloatTensor(np.asarray(self.embedding_vectors))

    def load_embeddings_from_file_and_unique_words(self, emb_path, emb_delimiter=' ', verbose=True):
        embedding_words = [emb_word for emb_word, _ in self.load_embeddings_from_file(emb_path=self.embedding_file, 
                                                                                      emb_delimiter=emb_delimiter)]
        emb_word2unique_word = dict()  # 一个emb_word对应一个[unique_word_1, unique_word_2, ...]
        out_of_vocab_words = list()  # 不在预训练的词汇中，即OOV
        for unique_word in self.unique_words:
            emb_word = self.get_embedding_word(unique_word, embedding_words)

            if emb_word is None: # 说明这个词的各种形式都不在预训练的词表中
                out_of_vocab_words.append(unique_word)
            else:
                if emb_word not in emb_word2unique_word:
                    emb_word2unique_word[emb_word] = [unique_word] # 注意这里的list形式，方便下面append（）
                else:
                    emb_word2unique_word[emb_word].append(unique_word) # emb_word是忽略大小写的词，这里是把词的各种形式放到统一的键下

        for emb_word, emb_vector in self.load_embeddings_from_file(emb_path=self.embedding_file,
                                                                   emb_delimiter=emb_delimiter):
            if emb_word in emb_word2unique_word:
                for unique_word in emb_word2unique_word[emb_word]:
                    self.word2idx[unique_word] = len(self.word2idx)
                    self.embedding_vectors.append(emb_vector) # 只有大小写区别的词使用同一词向量
        if verbose:
            print('\nloading vocabulary from embedding file and unique words:')
            print('    First 20 OOV words:')
            for i, oov_word in enumerate(out_of_vocab_words):
                print('        out_of_vocab_words[%d] = %s' % (i, oov_word))
                if i > 20:
                    break
            print(' -- len(out_of_vocab_words) = %d' % len(out_of_vocab_words))
            print(' -- original_words_num = %d' % self.original_words_num)
            print(' -- lowercase_words_num = %d' % self.lowercase_words_num)
            print(' -- zero_digits_replaced_num = %d' % self.zero_digits_replaced_num)
            print(' -- zero_digits_replaced_lowercase_num = %d' % self.zero_digits_replaced_lowercase_num)

    def load_embeddings_from_file(self, emb_path, emb_delimiter):
        """从文件中读入预训练词向量"""
        #如果相应路径下没有词向量文件，就自动下载
        if not os.path.isfile(emb_path):
            print("Pre-trained embedding file {} not found. Downloading...".format(emb_path))
            data_url = 'https://raw.githubusercontent.com/cybercsc/Relation-Extraction-Data/main/vector_50d.txt'
            save_url=emb_path.rsplit('/',1)[0]
            file_name=emb_path.rsplit('/',1)[-1]
            Download(data_url,save_url,file_name)
        with open(emb_path, 'r') as f:
            for line in f:
                values = line.strip().split(emb_delimiter)
                word = values[0]
                # filter过滤空白字符，然后map把str类型转换为float类型
                emb_vector = list(map(lambda emb: float(emb),
                                  filter(lambda val: val and not val.isspace(), values[1:])))
                yield word, emb_vector

    def get_embedding_word(self, word, embedding_words):
        """将数据集中的词映射为词向量中的词"""
        if word in embedding_words:
            self.original_words_num += 1
            return word
        elif word.lower() in embedding_words:
            self.lowercase_words_num += 1
            return word.lower()
        elif re.sub(r'\d', '0', word) in embedding_words:
            self.zero_digits_replaced_num += 1
            return re.sub(r'\d', '0', word)
        elif re.sub(r'\d', '0', word.lower()) in embedding_words:
            self.zero_digits_replaced_lowercase_num += 1
            return re.sub(r'\d', '0', word.lower())
        return None

    def load_sentences_labels(self, sentences_file, labels_file, d):
        """导入加工好的句子和对应的关系标签. 
            将所有词和关系标签转化为id，并存入字典d.
        """
        sents, pos1s, pos2s = list(), list(), list()
        labels = list()

        # 如果词在词表中，就映射为id，而OOv则映射为UNK
        with open(sentences_file, 'r') as f:
            for i, line in enumerate(f):
                e1, e2, sent = line.strip().split('\t')
                words = sent.split(' ')
                e1_start = e1.split(' ')[0] if ' ' in e1 else e1
                e2_start = e2.split(' ')[0] if ' ' in e2 else e2
                #实体e1/e2在其所在句子中的下标顺序
                e1_idx = words.index(e1_start) 
                e2_idx = words.index(e2_start)  
                sent, pos1, pos2 = list(), list(), list()
                for idx, word in enumerate(words):
                    emb_word = self.get_embedding_word(word, self.word2idx)
                    if emb_word:
                        sent.append(self.word2idx[word])
                    else:
                        sent.append(self.unk_idx)
                    pos1.append(self.get_pos_feature(idx - e1_idx))
                    pos2.append(self.get_pos_feature(idx - e2_idx))
                sents.append(sent)
                pos1s.append(pos1)
                pos2s.append(pos2)
        
        with open(labels_file, 'r') as f:
            for line in f:
                idx = self.label2idx[line.strip()]
                labels.append(idx)

        # 确保关系标签和句子对应
        assert len(labels) == len(sents)

        # 存入字典d
        d['data'] = {'sents': sents, 'pos1s': pos1s, 'pos2s': pos2s}
        d['labels'] = labels
        d['size'] = len(sents)

    def get_pos_feature(self, x):
        """用窗口截出最大考虑范围:
            -limit ~ limit => 0 ~ limit * 2+2
        """
        if x < -self.limit:
            return 0
        elif x >= -self.limit and x <= self.limit:
            return x + self.limit + 1
        else:
            return self.limit * 2 + 2

    def load_data(self, data_type):
        data = dict()
        
        if data_type in ['train', 'val', 'test']:
            sentences_file = os.path.join(self.data_dir, data_type, 'sentences.txt')
            labels_file = os.path.join(self.data_dir, data_type, 'labels.txt')
            self.load_sentences_labels(sentences_file, labels_file, data)
        else:
            raise ValueError("data type not in ['train', 'val', 'test']")
        return data

    def data_iterator(self, data, batch_size, shuffle='False'):
        """用于生成batch的迭代器."""
        order = list(range(data['size']))
        if shuffle:
            random.seed(230) # 保证结果可以复现
            random.shuffle(order)
        for i in range((data['size'])//batch_size):
            batch_sents = [data['data']['sents'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_pos1s = [data['data']['pos1s'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_pos2s = [data['data']['pos2s'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]
            batch_labels = [data['labels'][idx] for idx in order[i*batch_size:(i+1)*batch_size]]

            """每个batch填充的最大长度不一样"""
            batch_max_len = self.max_len

            """分为词特征和位置特征，词特征填充pad，位置特征填self.limit*2+2"""
            batch_data_sents = self.pad_idx * np.ones((batch_size, batch_max_len))
            batch_data_pos1s = (self.limit * 2 + 2) * np.ones((batch_size, batch_max_len))
            batch_data_pos2s = (self.limit * 2 + 2) * np.ones((batch_size, batch_max_len))
            for j in range(batch_size):
                cur_len = len(batch_sents[j])
                min_len = min(cur_len, batch_max_len)
                batch_data_sents[j][:min_len] = batch_sents[j][:min_len]
                batch_data_pos1s[j][:min_len] = batch_pos1s[j][:min_len]
                batch_data_pos2s[j][:min_len] = batch_pos2s[j][:min_len]
            batch_data_sents = self.tensor_ensure_gpu(torch.LongTensor(batch_data_sents))
            batch_data_pos1s = self.tensor_ensure_gpu(torch.LongTensor(batch_data_pos1s))
            batch_data_pos2s = self.tensor_ensure_gpu(torch.LongTensor(batch_data_pos2s))
            batch_labels = self.tensor_ensure_gpu(torch.LongTensor(batch_labels))

            batch_data = {'sents': batch_data_sents, 'pos1s': batch_data_pos1s, 'pos2s': batch_data_pos2s}
            yield batch_data, batch_labels

