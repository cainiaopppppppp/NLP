"""Build vocabularies of words and labels from datasets"""

import argparse
from collections import Counter

import json
import os


parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, help="Minimum count for words in the dataset", type=int)
parser.add_argument('--min_count_tag', default=1, help="Minimum count for labels in the dataset", type=int)
parser.add_argument('--data_dir', default='data/SemEval2010_task8', help="Directory containing the dataset")


def save_to_txt(vocab, txt_path):
    with open(txt_path, 'w') as f:
        for token in vocab:
            f.write(token + '\n')
            
def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def update_vocab(txt_path, vocab):
    """从数据集中更新词表"""
    size = 0
    with open(txt_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.endswith('...'):
                line = line.rstrip('...')
            word_seq = line.split('\t')[-1].split(' ')
            vocab.update(word_seq)
            size = i
    return size + 1

def update_labels(txt_path, labels):
    """从数据集中更新关系类型字典"""
    size = 0
    with open(txt_path) as f:
        for i, line in enumerate(f):
            line = line.strip()  #一行一个标签
            labels.update([line])
            size = i
    return size + 1


if __name__ == '__main__':
    args = parser.parse_args()

    #建立测试集和训练集的词表
    print("Building word vocabulary...")
    words = Counter()
    size_train_sentences = update_vocab(os.path.join(args.data_dir, 'train/sentences.txt'), words)
    size_test_sentences = update_vocab(os.path.join(args.data_dir, 'test/sentences.txt'), words)
    print("- done.")

    # 建立测试集和训练集的关系类型标签字典
    print("Building label vocabulary...")
    labels = Counter()
    size_train_tags = update_labels(os.path.join(args.data_dir, 'train/labels.txt'), labels)
    size_test_tags = update_labels(os.path.join(args.data_dir, 'test/labels.txt'), labels)
    print("- done.")

    # 确保语句和关系标签对应
    assert size_train_sentences == size_train_tags
    assert size_test_sentences == size_test_tags

    #去掉低词频的词和标签
    words = sorted([tok for tok, count in words.items() if count >= args.min_count_word])
    labels = sorted([tok for tok, count in labels.items() if count >= args.min_count_tag])

    #保存
    print("Saving vocabularies to file...")
    save_to_txt(words, os.path.join(args.data_dir, 'words.txt'))
    save_to_txt(labels, os.path.join(args.data_dir, 'labels.txt'))
    print("- done.")

    # 保存数据集属性
    sizes = {
        'train_size': size_train_sentences,
        'test_size': size_test_sentences,
        'vocab_size': len(words),
        'num_tags': len(labels)
    }
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'dataset_params.json'))

    to_print = "\n".join("-- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the dataset:\n{}".format(to_print))

