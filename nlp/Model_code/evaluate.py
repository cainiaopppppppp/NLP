"""Evaluate the model"""

import argparse
import logging
import os

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from tools import utils
import model.net as net
from tools.data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/SemEval2010_task8', help="Directory containing the dataset")
parser.add_argument('--embedding_file', default='data/embeddings/vector_50d.txt', help="Embedings file")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--gpu', default=0, help="GPU device number, 0 by default, -1 means CPU.")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")



def evaluate(model, data_iterator, num_steps, metric_labels):
    """使用step_num个batch来评估模型."""
    #设置为评估模式
    model.eval()

    output_labels = list()
    target_labels = list()

    # compute metrics over the dataset
    for _ in range(num_steps):
        # 循环获取评估batch
        batch_data, batch_labels = next(data_iterator)

        #计算模型输出
        batch_output = model(batch_data)  # batch_size x num_labels
        batch_output_labels = torch.max(batch_output, dim=1)[1]
        output_labels.extend(batch_output_labels.data.cpu().numpy().tolist())
        target_labels.extend(batch_labels.data.cpu().numpy().tolist())

    #计算各评估值
    p_r_f1_s = precision_recall_fscore_support(target_labels, output_labels, labels=metric_labels, average='micro')
    p_r_f1 = {'precison': p_r_f1_s[0] * 100,
              'recall': p_r_f1_s[1] * 100,
              'f1': p_r_f1_s[2] * 100}
    return p_r_f1


if __name__ == '__main__':
    #导入超参数
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    if torch.cuda.is_available():
        params.gpu = args.gpu
    else:
        params.gpu = -1
    
    #设置随机数种子
    torch.manual_seed(230)
    if params.gpu >= 0:
        torch.cuda.set_device(params.gpu)
        torch.cuda.manual_seed(230)
        
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    logging.info("Creating the dataset...")

    #初始化dataloader
    data_loader = DataLoader(data_dir=args.data_dir,
                             embedding_file=args.embedding_file,
                             word_emb_dim=params.word_emb_dim,
                             max_len=params.max_len,
                             pos_dis_limit=params.pos_dis_limit,
                             pad_word='<pad>',
                             unk_word='<unk>',
                             other_label='Other',
                             gpu=params.gpu)
    #导入词向量
    data_loader.load_embeddings_from_file_and_unique_words(emb_path=args.embedding_file,
                                                           emb_delimiter=' ',
                                                           verbose=True)
    metric_labels = data_loader.metric_labels  # relation labels to be evaluated

    #导入数据
    test_data = data_loader.load_data('test')
    test_data_iterator = data_loader.data_iterator(test_data, params.batch_size, shuffle='False')

    #指定测试集大小
    params.test_size = test_data['size']
    num_steps = params.test_size // params.batch_size # 多少个批次

    logging.info("- done.")


    logging.info("Starting evaluation...")
    #定义模型
    model = net.Net(data_loader, params)

    # 载入网络参数
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    #评估
    test_metrics = evaluate(model, test_data_iterator, num_steps, metric_labels)

    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- Test metrics: " + metrics_str)

