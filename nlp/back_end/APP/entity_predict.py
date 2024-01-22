import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

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

def build_word2idx(file_path):
    word2idx = {}
    with open(file_path, 'r') as file:
        for index, line in enumerate(file):
            word = line.strip()
            word2idx[word] = index

    word2idx['<pad>'] = len(word2idx)  # 添加 <pad> 符号
    word2idx['UNK'] = len(word2idx)    # 添加 UNK 符号
    return word2idx


def load_model(model_path, vocab_size, embedding_dim, hidden_dim, num_tags, dropout):
    model = EntityRecognitionModel(vocab_size, embedding_dim, hidden_dim, num_tags, dropout)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def prepare_sentence(sentence, word2idx):
    sentence_idx = [word2idx.get(word, word2idx['UNK']) for word in sentence.split()]
    return torch.tensor(sentence_idx, dtype=torch.long).unsqueeze(0)  # 增加一个批处理维度


def extract_entities(words, predictions):
    """
    从预测结果中提取两个最可能的实体。
    """
    # 获取每个词的实体概率
    entity_probs = torch.softmax(predictions, dim=2)[:, :, 1].squeeze(0)

    # 对概率进行排序，选择前两个最高的作为实体
    top_two_probs, top_two_indices = torch.topk(entity_probs, 2)

    # 按照在原始句子中的顺序提取实体
    ordered_indices = torch.sort(top_two_indices)[0]
    # 提取实体
    entities = [words[idx] for idx in ordered_indices]

    return entities

def predict(model, sentence, word2idx, device):
    model.to(device)
    tensor_sentence = prepare_sentence(sentence, word2idx).to(device)
    with torch.no_grad():
        predictions = model(tensor_sentence)

    words = sentence.split()
    entities = extract_entities(words, predictions)

    return entities


def main():
    # 参数设置
    vocab_size = 7860
    embedding_dim = 50
    hidden_dim = 256
    num_tags = 2
    dropout = 0.2
    model_path = '../experiments/entity_model/best_model.pt'
    sentence = "The child was carefully wrapped and bound into the cradle by means of a cord."

    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建 word2idx
    word2idx = build_word2idx('../data/SemEval2010_task8/entities.txt')

    # 加载模型
    model = load_model(model_path, vocab_size, embedding_dim, hidden_dim, num_tags, dropout)

    # 预测
    try:
        entities = predict(model, sentence, word2idx, device)
        print(f"Predicted entities in the sentence: {entities}")
    except ValueError as e:
        print(e)


if __name__ == '__main__':
    main()
