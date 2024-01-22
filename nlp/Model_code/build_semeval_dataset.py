"""读取并处理semval数据集"""

import os
import re
from download import *

pattern_normalwords = re.compile('(<e1>)|(</e1>)|(<e2>)|(</e2>)|(\'s)')
pattern_e1 = re.compile('<e1>(.*)</e1>')
pattern_e2 = re.compile('<e2>(.*)</e2>')
pattern_del = re.compile('^[!"#$%&\\\'()*+,-./:;<=>?@[\\]^_`{|}~]|[!"#$%&\\\'()*+,-./:;<=>?@[\\]^_`{|}~]$')


def load_dataset(path_dataset):
    """加载语料数据，包括实体、关系、完整句子"""
    dataset = []
    with open(path_dataset) as f:
        piece = list()  #根据语句存储形式设置piece,一个piece就是一个<标注语句,关系,comment>的组合
        for line in f:
            line = line.strip()
            if line:
                piece.append(line)
            elif piece:
                #sentence即标注语句
                sentence = piece[0].split('\t')[1].strip('"')
                #提取出不带标注符号的两个实体
                e1 = delete_symbol(pattern_e1.findall(sentence)[0])
                e2 = delete_symbol(pattern_e2.findall(sentence)[0])
                sentence_nosymbol = list()
                #提取出不带标注符号，并且不带标点符号的原始语句
                for word in pattern_normalwords.sub('', sentence).split(' '):
                    new_word = delete_symbol(word)
                    if new_word:
                        sentence_nosymbol.append(new_word)
                #语句中包含的关系是piece的第二行
                relation = piece[1]
                #重组成<实体1,实体2,原语句,实体关系>
                dataset.append(((e1, e2, ' '.join(sentence_nosymbol)), relation))
                piece = list()
    return dataset

def delete_symbol(text):
    if pattern_del.search(text):
        return pattern_del.sub('', text)
    return text

def save_dataset(dataset, save_dir):
    """将加工后的语料存入sentences.txt和labels.txt"""
    # 创建文件夹
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 导出数据集
    # 大致形式是：
    """words : ('garbage bag', 'clothes', 'I have a large garbage bag full of clothes for a teen or preteen girl')"""
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences, \
        open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
        for words, labels in dataset:
            file_sentences.write('{}\n'.format('\t'.join(words)))
            file_labels.write('{}\n'.format(labels))
    print("- done.")


if __name__ == '__main__':
    path_train = 'data/SemEval2010_task8/TRAIN_FILE.TXT'
    path_test = 'data/SemEval2010_task8/TEST_FILE.TXT'
    msg = "{} or {} file not found. Downloading...".format(path_train, path_test)
    if not (os.path.isfile(path_train) and os.path.isfile(path_test)):
        print(msg)
            # data源地址url
        data_url = 'https://raw.githubusercontent.com/cybercsc/Relation-Extraction-Data/main/original.zip'
        # data保存文件夹
        save_url='./data/SemEval2010_task8'
        path_train=os.path.join(save_url,'TRAIN_FILE.TXT')
        path_test = os.path.join(save_url,'TEST_FILE.TXT')
        # data文件名
        file_name = 'original.zip'
        Download(data_url,save_url, file_name)
        move(save_url,file_name,path_test,path_train)
    

    # 导入数据集
    print("Loading SemEval2010_task8 dataset into memory...")
    train_dataset = load_dataset(path_train)
    test_dataset = load_dataset(path_test)
    print("- done.")

    # 提取关键信息后存入文件
    save_dataset(train_dataset, 'data/SemEval2010_task8/train')
    save_dataset(test_dataset, 'data/SemEval2010_task8/test')