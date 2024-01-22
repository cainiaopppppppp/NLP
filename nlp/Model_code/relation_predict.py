import torch
import os
import json
import re
import model.net as net
from tools.data_loader import DataLoader
import tools.utils as utils
import torch.nn as nn

# 定义模型类型
MODEL_TYPES = ['CNN', 'BiLSTM_Att', 'BiLSTM_MaxPooling']

# 定义模型加载函数
def load_model(model_dir, model_type, data_loader, params):
    """Load the model from a .pth.tar file."""
    if model_type == 'CNN':
        model = net.CNN(data_loader, params)
    elif model_type == 'BiLSTM_Att':
        model = net.BiLSTM_Att(data_loader, params)
    elif model_type == 'BiLSTM_MaxPooling':
        model = net.BiLSTM_MaxPooling(data_loader, params)
    else:
        raise ValueError("Unknown model type")

    model_path = os.path.join(model_dir, model_type, 'best.pth.tar')

    # 确保在加载模型状态之前，模型的词嵌入层已正确初始化
    embedding_vectors = data_loader.get_loaded_embedding_vectors()
    model.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_vectors, freeze=False)

    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
    return model

def process_sentence(sentence, data_loader):
    """处理输入句子，去除标记并转换为单词索引列表"""
    # 移除实体标记
    sentence = re.sub(r'</?e[12]>', '', sentence)
    words = sentence.split()
    indexed_sentence = [data_loader.word2idx.get(word, data_loader.unk_idx) for word in words]

    # 计算实体位置
    pos1, pos2 = find_entity_positions(words, sentence)
    indexed_pos1 = [data_loader.get_pos_feature(i - pos1) for i in range(len(words))]
    indexed_pos2 = [data_loader.get_pos_feature(i - pos2) for i in range(len(words))]

    return indexed_sentence, indexed_pos1, indexed_pos2

def find_entity_positions(words, original_sentence):
    """在去除标记的句子中找到实体的位置"""
    # 使用原始句子中的标记来定位实体
    e1_start = original_sentence.find("<e1>")
    e1_end = original_sentence.find("</e1>")
    e2_start = original_sentence.find("<e2>")
    e2_end = original_sentence.find("</e2>")

    # 计算实体位置
    pos1 = len(original_sentence[:e1_start].split())
    pos2 = len(original_sentence[:e2_start].split())

    return pos1, pos2

def create_batch_data(data_loader, processed_sentences):
    """将处理过的句子列表转换为模型输入所需的batch格式"""
    batch_size = len(processed_sentences)
    batch_data_sents, batch_data_pos1s, batch_data_pos2s = [], [], []

    for sent, pos1, pos2 in processed_sentences:
        # 确保句子长度不超过最大长度限制
        if len(sent) > data_loader.max_len:
            sent = sent[:data_loader.max_len]
            pos1 = pos1[:data_loader.max_len]
            pos2 = pos2[:data_loader.max_len]

        # 填充句子和位置信息，以满足最大长度
        while len(sent) < data_loader.max_len:
            sent.append(data_loader.pad_idx)
            pos1.append(data_loader.limit * 2 + 2)
            pos2.append(data_loader.limit * 2 + 2)

        batch_data_sents.append(sent)
        batch_data_pos1s.append(pos1)
        batch_data_pos2s.append(pos2)

    # 将列表转换为Tensor
    batch_data_sents = torch.LongTensor(batch_data_sents)
    batch_data_pos1s = torch.LongTensor(batch_data_pos1s)
    batch_data_pos2s = torch.LongTensor(batch_data_pos2s)

    if data_loader.gpu >= 0:
        batch_data_sents = batch_data_sents.cuda(device=data_loader.gpu)
        batch_data_pos1s = batch_data_pos1s.cuda(device=data_loader.gpu)
        batch_data_pos2s = batch_data_pos2s.cuda(device=data_loader.gpu)

    return {'sents': batch_data_sents, 'pos1s': batch_data_pos1s, 'pos2s': batch_data_pos2s}


def idx_to_label(label_idx, label_dict):
    """将标签索引转换为标签名"""
    label_names = list(label_dict.keys())
    label_values = list(label_dict.values())

    # 如果索引在label_values中，则返回对应的标签名，否则返回 None
    if label_idx in label_values:
        return label_names[label_values.index(label_idx)]
    return None

# 定义预测函数
def predict(batch_data, model, label_dict):
    """Predict the relation for the given batch data."""
    model.eval()
    with torch.no_grad():
        output = model(batch_data)
        if len(output.shape) == 1:
            output = output.unsqueeze(0)
        prediction = torch.argmax(output, dim=1)
        print(prediction.item())
        relation = idx_to_label(prediction.item(), label_dict)
    return relation

def load_labels_to_dict(labels_file):
    """从文本文件中加载标签，并创建一个标签名称到索引的映射字典."""
    label_dict = {}
    with open(labels_file, 'r') as f:
        for idx, label in enumerate(f):
            label = label.strip()
            label_dict[label] = idx
    return label_dict

# 主函数
def main():
    sentence_file = './sentence.txt'  # 句子文件的路径
    model_dir = './experiments/base_model'  # 模型文件夹的根目录
    data_dir = './data/SemEval2010_task8'  # 数据集的目录
    embedding_file = './data/embeddings/vector_50d.txt'  # 嵌入文件的路径
    word_emb_dim = 50  # 嵌入维度

    # 载入标签字典
    label_dict = load_labels_to_dict(os.path.join(data_dir, 'labels.txt'))

    with open(sentence_file, 'r') as file:
        original_sentence = file.read().strip()

    data_loader = DataLoader(data_dir=data_dir, embedding_file=embedding_file, word_emb_dim=word_emb_dim,
                             max_len=100, pos_dis_limit=50, pad_word='<pad>', unk_word='<unk>', other_label='Other',
                             gpu=-1)
    data_loader.load_embeddings_from_file_and_unique_words(emb_path=embedding_file, emb_delimiter=' ', verbose=True)

    processed_sentence = process_sentence(original_sentence, data_loader)
    batch_data = create_batch_data(data_loader, [processed_sentence])

    # 对每个模型进行预测
    for model_type in MODEL_TYPES:
        print(f"Predicting with {model_type} model...")
        json_path = os.path.join(model_dir, model_type, 'params.json')
        assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
        params = utils.Params(json_path)
        params.gpu = -1
        model = load_model(model_dir, model_type, data_loader, params)
        relation = predict(batch_data, model, label_dict)
        print(f"Predicted relation by {model_type}: {relation}")

if __name__ == '__main__':
    main()