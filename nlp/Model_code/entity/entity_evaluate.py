# evaluate.py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from entity import EntityRecognitionModel
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# 假设的参数
vocab_size = 20000
embedding_dim = 100
hidden_dim = 256
num_tags = 2
batch_size = 32
dropout = 0.5

# 加载word2idx
def build_word2idx(file_path):
    word2idx = {}
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            word = line.strip()
            word2idx[word] = index
    return word2idx

word2idx = build_word2idx('./data/SemEval2010_task8/entities.txt')
word2idx['<pad>'] = len(word2idx)  # 添加 <pad> 符号
word2idx['UNK'] = len(word2idx)    # 添加 UNK 符号

pad_idx = word2idx['<pad>']  # 获取 <pad> 符号的索引

# 定义数据集
class EntityDataset(Dataset):
    def __init__(self, sentences_file, word2idx, pad_idx):
        self.data = []
        with open(sentences_file, 'r') as f:
            for line in f:
                entities, sentence = line.strip().split('\t')
                sentence_idx = [word2idx.get(word, word2idx['UNK']) for word in sentence.split(' ')]
                entities_idx = [int(e) for e in entities.split(' ')]  # 直接将标签转换为整数
                self.data.append((torch.tensor(sentence_idx, dtype=torch.long),
                                  torch.tensor(entities_idx, dtype=torch.long)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    sentences, entities = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=word2idx['<pad>'])
    entities_padded = pad_sequence(entities, batch_first=True, padding_value=word2idx['<pad>'])
    return sentences_padded, entities_padded

# 加载模型
model = EntityRecognitionModel(vocab_size, embedding_dim, hidden_dim, 2, dropout)  # 注意：输出维度应为2
model.load_state_dict(torch.load('./experiments/entity_model/best_model.pt'))
model.eval()

# 加载测试数据集
test_dataset = EntityDataset('./data/SemEval2010_task8/test/processed.txt', word2idx, pad_idx)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 评估模型
y_true = []
y_pred = []
with torch.no_grad():
    for sentences, entities in test_loader:
        predictions = model(sentences)
        predicted_tags = torch.argmax(predictions, dim=2)
        y_true.extend(entities.numpy().flatten())
        y_pred.extend(predicted_tags.numpy().flatten())

# 移除填充标签的影响
y_true_filtered = [tag for tag, pred in zip(y_true, y_pred) if tag != pad_idx]
y_pred_filtered = [pred for tag, pred in zip(y_true, y_pred) if tag != pad_idx]

precision, recall, f1, _ = precision_recall_fscore_support(y_true_filtered, y_pred_filtered, average='binary')  # 改为'binary'
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")