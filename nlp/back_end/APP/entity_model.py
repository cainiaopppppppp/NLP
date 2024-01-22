import torch
import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def load_glove_embeddings(glove_file, word2idx, embedding_dim):
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector

    weights_matrix = np.zeros((len(word2idx), embedding_dim))
    for word, idx in word2idx.items():
        weights_matrix[idx] = embeddings.get(word, np.random.normal(scale=0.6, size=(embedding_dim, )))

    return torch.tensor(weights_matrix, dtype=torch.float32)

class EntityRecognitionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, dropout, weights_matrix):
        super(EntityRecognitionModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weights_matrix, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_tags)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)
        outputs = self.dropout(outputs)
        predictions = self.fc(outputs)
        return predictions

class EntityDataset(Dataset):
    def __init__(self, sentences_file, word2idx, pad_idx, oversample=True):
        self.data = []
        self.oversample_data = []
        with open(sentences_file, 'r') as f:
            for line in f:
                entities, sentence = line.strip().split('\t')
                sentence_idx = [word2idx.get(word, word2idx['UNK']) for word in sentence.split(' ')]
                entities_idx = [int(e) for e in entities.split(' ')]
                data_point = (torch.tensor(sentence_idx, dtype=torch.long), torch.tensor(entities_idx, dtype=torch.long))
                self.data.append(data_point)
                if any(entities_idx):
                    self.oversample_data.append(data_point)

        if oversample:
            self.oversample()

    def oversample(self):
        num_samples = len(self.data)
        num_entity_samples = len(self.oversample_data)
        oversample_times = max(1, int((num_samples - num_entity_samples) / num_entity_samples))
        for _ in range(oversample_times):
            self.data.extend(self.oversample_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, word2idx):
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
    word2idx['<pad>'] = len(word2idx)
    word2idx['UNK'] = len(word2idx)
    return word2idx

def split_dataset(dataset, split_ratio=0.1):
    valid_size = int(len(dataset) * split_ratio)
    train_size = len(dataset) - valid_size
    return random_split(dataset, [train_size, valid_size])

def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, model_save_dir, patience=3):
    best_loss = float('inf')
    patience_counter = 0

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

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for sentences, entities in valid_loader:
                sentences, entities = sentences.to(device), entities.to(device)
                predictions = model(sentences)
                valid_loss = criterion(predictions.view(-1, 2), entities.view(-1))
                total_valid_loss += valid_loss.item()

        average_valid_loss = total_valid_loss / len(valid_loader)
        print(f"Validation Loss: {average_valid_loss}")

        if average_valid_loss < best_loss:
            best_loss = average_valid_loss
            best_model_path = os.path.join(model_save_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break

    if epoch == num_epochs - 1 and patience_counter < patience:
        final_model_path = os.path.join(model_save_dir, 'final_model.pt')
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved at {final_model_path}")

def evaluate(model, test_loader, device, word2idx):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for sentences, entities in test_loader:
            sentences = sentences.to(device)
            predictions = model(sentences)
            predicted_tags = torch.argmax(predictions, dim=2)
            y_true.extend(entities.numpy().flatten())
            y_pred.extend(predicted_tags.cpu().numpy().flatten())

    y_true_filtered = [tag for tag, pred in zip(y_true, y_pred) if tag != word2idx['<pad>']]
    y_pred_filtered = [pred for tag, pred in zip(y_true, y_pred) if tag != word2idx['<pad>']]

    precision, recall, f1, _ = precision_recall_fscore_support(y_true_filtered, y_pred_filtered, average='binary')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

def main():
    vocab_size = 7860
    embedding_dim = 50
    hidden_dim = 256
    num_tags = 2
    num_epochs = 50
    batch_size = 128
    dropout = 0.5
    lr = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    word2idx = build_word2idx('../data/SemEval2010_task8/entities.txt')

    glove_embeddings = load_glove_embeddings('../data/embeddings/glove.6B.50d.txt', word2idx, embedding_dim)

    model_save_dir = '../experiments/entity_model'
    os.makedirs(model_save_dir, exist_ok=True)

    model = EntityRecognitionModel(vocab_size, embedding_dim, hidden_dim, num_tags, dropout, glove_embeddings)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr)

    train_dataset = EntityDataset('../data/SemEval2010_task8/train/processed.txt', word2idx, word2idx['<pad>'])
    train_dataset, valid_dataset = split_dataset(train_dataset, split_ratio=0.1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, word2idx))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, word2idx))

    train(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, model_save_dir)

    test_dataset = EntityDataset('../data/SemEval2010_task8/test/processed.txt', word2idx, word2idx['<pad>'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, word2idx))

    best_model_path = os.path.join(model_save_dir, 'best_model.pt')
    model.load_state_dict(torch.load(best_model_path))
    evaluate(model, test_loader, device, word2idx)

if __name__ == '__main__':
    main()
