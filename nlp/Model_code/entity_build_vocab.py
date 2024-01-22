import argparse
from collections import Counter
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--min_count_entity', default=1, help="Minimum count for entities in the dataset", type=int)
parser.add_argument('--data_dir', default='data/SemEval2010_task8', help="Directory containing the dataset")

def save_to_txt(vocab, txt_path):
    with open(txt_path, 'w') as f:
        for token in vocab:
            f.write(token + '\n')

def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4)

def update_entity_vocab(txt_path, vocab):
    """从数据集中更新实体词表"""
    size = 0
    with open(txt_path) as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if len(parts) > 2:
                entity1, entity2, _ = parts
                vocab.update([entity1, entity2])
                size = i
    return size + 1

if __name__ == '__main__':
    args = parser.parse_args()

    # 建立实体词表
    print("Building entity vocabulary...")
    entities = Counter()
    size_train_entities = update_entity_vocab(os.path.join(args.data_dir, 'train/sentences.txt'), entities)
    size_test_entities = update_entity_vocab(os.path.join(args.data_dir, 'test/sentences.txt'), entities)
    print("- done.")

    # 去掉低频实体
    entities = sorted([tok for tok, count in entities.items() if count >= args.min_count_entity])

    # 保存实体词汇表
    print("Saving entity vocabularies to file...")
    save_to_txt(entities, os.path.join(args.data_dir, 'entities.txt'))
    print("- done.")

    # 保存数据集属性
    sizes = {
        'train_size': size_train_entities,
        'test_size': size_test_entities,
        'entity_vocab_size': len(entities)
    }
    save_dict_to_json(sizes, os.path.join(args.data_dir, 'entity_dataset_params.json'))

    to_print = "\n".join("-- {}: {}".format(k, v) for k, v in sizes.items())
    print("Characteristics of the entity dataset:\n{}".format(to_print))
