# train.py
import torch
import os
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.nn as nn
from entity import EntityRecognitionModel
from torch.utils.data import DataLoader, Dataset


# 假设的参数
vocab_size = 20000
embedding_dim = 100
hidden_dim = 256
num_tags = 2
num_epochs = 20
batch_size = 64
dropout = 0.5


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

def build_word2idx(file_path):
    word2idx = {}
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            word = line.strip()
            word2idx[word] = index
    return word2idx

# 检查GPU是否可用，如果可用则使用它
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

word2idx = build_word2idx('./data/SemEval2010_task8/entities.txt')
word2idx['<pad>'] = len(word2idx)
word2idx['UNK'] = len(word2idx)

model_save_dir = './experiments/entity_model'
os.makedirs(model_save_dir, exist_ok=True)

# 初始化模型、损失函数和优化器
model = EntityRecognitionModel(vocab_size, embedding_dim, hidden_dim, num_tags, dropout)
model.to(device)
ignore_index = word2idx['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
optimizer = optim.Adam(model.parameters())

pad_idx = word2idx['<pad>']

# 加载数据集
train_dataset = EntityDataset('./data/SemEval2010_task8/train/processed.txt', word2idx, pad_idx)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

best_loss = float('inf')

# 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for sentences, entities in train_loader:
        sentences, entities = sentences.to(device), entities.to(device)

        optimizer.zero_grad()
        predictions = model(sentences)
        loss = criterion(predictions.view(-1, 2), entities.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss}")

    if average_loss < best_loss:
        best_loss = average_loss
        best_model_path = os.path.join(model_save_dir, 'best_model.pt')
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved at {best_model_path}")

    if epoch == num_epochs - 1:
        final_model_path = os.path.join(model_save_dir, 'final_model.pt')
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved at {final_model_path}")