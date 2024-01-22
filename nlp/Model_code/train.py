"""Train and evaluate the model"""
import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange

import tools.utils as utils
import model.net as net
from tools.data_loader import DataLoader
from evaluate import evaluate
from download import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/SemEval2010_task8', help="Directory containing the dataset")
parser.add_argument('--embedding_file', default='data/embeddings/vector_50d.txt', help="Path to embeddings file.")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--model_name', default='CNN', help="Choose model")
parser.add_argument('--gpu', default=-1, help="GPU device number, 0 by default, -1 means CPU.")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")


def train(model, data_iterator, optimizer, scheduler, params, steps_num):
    """使用step_num个batch来训练模型"""
    # 设置为训练模式
    model.train()
    loss_avg = utils.RunningAverage()
    t = trange(steps_num)
    for _ in t:
        # 循环获取训练batch
        batch_data, batch_labels = next(data_iterator)
        #计算模型输出和损失
        batch_output = model(batch_data)
        loss = model.loss(batch_output, batch_labels)

        # 各个batch之间要清空梯度
        model.zero_grad()
        #误差反传
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), params.clip_grad)

        # 计算梯度
        optimizer.step()

        # 更新平均损失
        loss_avg.update(loss.item())
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    return loss_avg()
    

def train_and_evaluate(model, train_data, val_data, optimizer, scheduler, params, metric_labels, model_dir, restore_file=None):
    """在训练的同时，每个epoch都评估测试一次."""
    #如果指定了已经训练好的模型，就导入模型直接进行评估
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
        
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, params.epoch_num + 1):
        # 开始一个epoch
        logging.info("Epoch {}/{}".format(epoch, params.epoch_num))

        # 计算epoch中包含的batch个数
        train_steps_num = params.train_size // params.batch_size
        val_steps_num = params.val_size // params.batch_size

        # 用于生成训练集batch的迭代器
        train_data_iterator = data_loader.data_iterator(train_data, params.batch_size, shuffle='True')
        # 在训练集上训练一个epoch
        train_loss = train(model, train_data_iterator, optimizer, scheduler, params, train_steps_num)

        # 再生成两个分别用于生成训练集和测试集batch的迭代器
        train_data_iterator = data_loader.data_iterator(train_data, params.batch_size)
        val_data_iterator = data_loader.data_iterator(val_data, params.batch_size)

        # 在一个epoch内评估训练集和测试集
        train_metrics = evaluate(model, train_data_iterator, train_steps_num, metric_labels)
        train_metrics['loss'] = train_loss
        train_metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in train_metrics.items())
        logging.info("- Train metrics: " + train_metrics_str)
        
        val_metrics = evaluate(model, val_data_iterator, val_steps_num, metric_labels)
        val_metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in val_metrics.items())
        logging.info("- Eval metrics: " + val_metrics_str)
        
        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # 保存网络参数
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()}, 
                               is_best=improve_f1>0,
                               checkpoint=model_dir)
        if improve_f1 > 0:
            logging.info("- Found new best F1")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        if (patience_counter >= params.patience_num and epoch > params.min_epoch_num) or epoch == params.epoch_num:
            logging.info("best val f1: {:05.2f}".format(best_val_f1))
            break
        

def CNN(data_loader,params):
    #定义模型和优化器
    model = net.CNN(data_loader, params)
    if params.optim_method == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=params.weight_decay)
    elif params.optim_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999), weight_decay=params.weight_decay)
    elif params.optim_method == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, weight_decay=params.weight_decay)
    else:
        raise ValueError("Unknown optimizer, must be one of 'sgd'/'adam'/'adadelta'.")

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch)) # 动态改变学习率

    #训练并评估
    logging.info("Starting training for {} epoch(s)".format(params.epoch_num))
    train_and_evaluate(model=model,
                       train_data=train_data,
                       val_data=val_data,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       params=params,
                       metric_labels=metric_labels,
                       model_dir=args.model_dir,
                       restore_file=args.restore_file)

def BiLSTM_Att(data_loader,params):
    #定义模型和优化器
    model = net.BiLSTM_Att(data_loader, params)
    if params.optim_method == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=params.weight_decay)
    elif params.optim_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999), weight_decay=params.weight_decay)
    elif params.optim_method == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, weight_decay=params.weight_decay)
    else:
        raise ValueError("Unknown optimizer, must be one of 'sgd'/'adam'.")

    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch)) # 动态改变学习率

    #训练并评估
    logging.info("Starting training for {}  epoch(s)".format(params.epoch_num))
    train_and_evaluate(model=model,
                       train_data=train_data,
                       val_data=val_data,
                       optimizer=optimizer,
                       scheduler=scheduler,
                       # scheduler=None,
                       params=params,
                       metric_labels=metric_labels,
                       model_dir=args.model_dir,
                       restore_file=args.restore_file)


def BiLSTM_MaxPooling(data_loader,params):
    #定义模型和优化器
    model = net.BiLSTM_MaxPooling(data_loader, params)
    if params.optim_method == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=params.weight_decay)
    elif params.optim_method == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, betas=(0.9, 0.999), weight_decay=params.weight_decay)
    elif params.optim_method == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, weight_decay=params.weight_decay)
    else:
        raise ValueError("Unknown optimizer, must be one of 'sgd'/'adam'.")

    #scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(1 + 0.05*epoch)) 
    #动态改变学习率

    #训练并评估
    logging.info("Starting training for {}  epoch(s)".format(params.epoch_num))
    train_and_evaluate(model=model,
                       train_data=train_data,
                       val_data=val_data,
                       optimizer=optimizer,
                       # scheduler=scheduler,
                       scheduler=None,
                       params=params,
                       metric_labels=metric_labels,
                       model_dir=args.model_dir,
                       restore_file=args.restore_file)

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
    
    #设置日志
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("Loading the datasets...")
    
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
    metric_labels = data_loader.metric_labels 
    
    #导入数据
    train_data = data_loader.load_data('train')
    #由于数据量不大，所以测试集和验证集等价
    val_data = data_loader.load_data('test')

    #指定训练和测试集大小
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.info("- done.")

    if args.model_name == 'CNN':
        CNN(data_loader,params)
    elif args.model_name == 'BiLSTM_Att':
        BiLSTM_Att(data_loader,params)
    elif args.model_name == 'BiLSTM_MaxPooling':
        BiLSTM_MaxPooling(data_loader,params)
    else:
        print("Error! No model named {}!".format(args.model_name))

