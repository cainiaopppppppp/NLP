# entity.py
import torch
import torch.nn as nn

class EntityRecognitionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, dropout):
        super(EntityRecognitionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(dropout)  # 添加 Dropout 层
        self.fc = nn.Linear(hidden_dim, num_tags)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)
        outputs = self.dropout(outputs)  # 应用 Dropout
        predictions = self.fc(outputs)
        return predictions

