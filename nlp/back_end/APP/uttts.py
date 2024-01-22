import json
import logging
import os
import shutil

import torch
import numpy as np

'''
    类的静态函数、类函数、普通函数、全局变量以及一些内置的属性都是放在类__dict__里的
    对象的__dict__中存储了一些self.xxx的一些东西
'''
class Params():
    """用于加载超参数的类"""

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__ # 返回对象的self.xxx组成的dict


class RunningAverage():
    """维护移动平均值"""

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """设置logger,同时在终端和日志文件中显示实验记录"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        #写入日志文件
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        #写入终端
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def save_checkpoint(state, is_best, checkpoint):
    """将各个checkpoint的模型和训练参数存入checkpoint + 'last.pth.tar'文件中，如果是当前最好模型，同时将其更新到checkpoint + 'best.pth.tar'中"""
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)

    """复制最好的模型"""
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))

def load_checkpoint(checkpoint, model, optimizer=None):
    """从文件中导入网络参数，如果同时给定优化器，就直接导入优化器的state_dict"""
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def generate_zero_vector(embedding_dim):
    return [0] * embedding_dim

# 生成embedding_dim长度的均匀分布向量z
def generate_random_vector(embedding_dim):
    return np.random.uniform(-np.sqrt(3.0 / embedding_dim), np.sqrt(3.0 / embedding_dim),
                             embedding_dim).tolist()

