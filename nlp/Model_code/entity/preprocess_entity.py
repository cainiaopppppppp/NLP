# preprocess_entity.py
import os

def process_data(input_file, output_file):
    with open(input_file, 'r') as file, open(output_file, 'w') as out:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            entity1, entity2, sentence = parts
            words = sentence.split(' ')
            entity_labels = [0] * len(words)  # 初始化为非实体
            for entity in [entity1, entity2]:
                if entity in words:
                    index = words.index(entity)
                    entity_labels[index] = 1  # 标记实体位置
            out.write(f"{' '.join(map(str, entity_labels))}\t{' '.join(words)}\n")

if __name__ == "__main__":
    data_dir = 'data/SemEval2010_task8'
    process_data(os.path.join(data_dir, 'train/sentences.txt'), os.path.join(data_dir, 'train/processed.txt'))
    process_data(os.path.join(data_dir, 'test/sentences.txt'), os.path.join(data_dir, 'test/processed.txt'))
