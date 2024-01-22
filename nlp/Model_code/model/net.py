"""各模型和损失函数的具体实现"""

import torch

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
	def __init__(self, data_loader, params):
		super(CNN, self).__init__()
		#导入数据集对应的词向量
		embedding_vectors = data_loader.get_loaded_embedding_vectors()
		#文本信息和位置信息的嵌入层
		self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_vectors, freeze=False)
		self.pos1_embedding = nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
		self.pos2_embedding = nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)

		self.max_len = params.max_len
		#dropout层
		self.dropout = nn.Dropout(params.dropout_ratio)

		feature_dim = params.word_emb_dim + params.pos_emb_dim * 2
		#通过cnn对句子级别的特征进行编码
		self.covns = nn.ModuleList([nn.Sequential(nn.Conv1d(in_channels=feature_dim,
									out_channels=params.filter_num,
									kernel_size=k),nn.Tanh(),nn.MaxPool1d(kernel_size=self.max_len-k+1)) for k in params.filters])

		filter_dim = params.filter_num * len(params.filters)
		labels_num = len(data_loader.label2idx)
		#输出层
		self.linear = nn.Linear(filter_dim, labels_num)

		self.loss = nn.CrossEntropyLoss()

		if params.gpu >= 0:
			self.cuda(device=params.gpu)

	def forward(self, x):
		batch_sents = x['sents']
		batch_pos1s = x['pos1s']
		batch_pos2s = x['pos2s']
		word_embs = self.word_embedding(batch_sents)
		pos1_embs = self.pos1_embedding(batch_pos1s)
		pos2_embs = self.pos2_embedding(batch_pos2s)

		input_feature = torch.cat([word_embs, pos1_embs, pos2_embs], dim=2)  # batch_size x seq_len x feature_dim
		input_feature = input_feature.permute(0,2,1) #(batch_size,feature_dim,seq_len)
		input_feature = self.dropout(input_feature)

		out = [conv(input_feature) for conv in self.covns] #(batch_size,filter_num,1)
		"""
			对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1 (维度1是max_pool的结果)
			每个Window_size 产生filter_num个feature,然后把这些feature拼接起来
		"""
		out = torch.cat(out,dim=1)
		out = self.dropout(out)
		out = out.view(-1,out.size(1)) #(batch_size, (filter_num*window_num))
		x = self.dropout(out)
		x = self.linear(x)
		return x


class BiLSTM_Att(nn.Module):
	def __init__(self,data_loader,params):
		super(BiLSTM_Att, self).__init__()
		embedding_vectors = data_loader.get_loaded_embedding_vectors()

		self.out_size = len(data_loader.label2idx)
		self.hidden_dim = params.hidden_dim
		self.batch_size = params.batch_size
		self.feature_dim = params.word_emb_dim + params.pos_emb_dim * 2
		self.lstm = nn.LSTM(self.feature_dim,self.hidden_dim//2,bidirectional=True)

		self.word_embedding = nn.Embedding.from_pretrained(embedding_vectors,freeze=False)
		self.pos1_embedding = nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
		self.pos2_embedding = nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)

		self.att_weight = nn.Parameter(torch.randn((self.batch_size, 1, self.hidden_dim)))

		self.dropout_emb = nn.Dropout(p=0.3)
		self.dropout_lstm = nn.Dropout(p=0.3)
		self.dropout_att = nn.Dropout(p=0.5)

		self.dense = nn.Linear(self.hidden_dim,self.out_size)
		self.device = None
		self.loss = nn.CrossEntropyLoss()
		if params.gpu >= 0:
			self.device = self.cuda(device=params.gpu)

	def begin_state(self, batch_size):
		"""生成初始隐藏状态，大小与给定的batch_size匹配."""
		state = (
			torch.zeros(2, batch_size, self.hidden_dim // 2),
			torch.zeros(2, batch_size, self.hidden_dim // 2))
		if self.device:
			return state.to(self.device)
		else:
			return state
	'''
		H: (batch_size,hidden_dim,seq_len)
		att_weight: (batch_size,1,hidden_dim)
	'''

	def attention(self, H):
		"""计算注意力权重并应用于输入H."""
		batch_size, _, _ = H.size()  # 获取当前批量的大小
		att_weight = torch.randn((batch_size, 1, self.hidden_dim)).to(H.device)  # 动态生成att_weight

		M = torch.tanh(H)
		a = torch.bmm(att_weight, M)
		a = F.softmax(a, dim=2)  # (batch_size,1,seq_len)
		a = a.transpose(1, 2)  # after a:  (batch_size,seq_len,1)
		return torch.bmm(H, a)  # (batch_size,hidden_dim,1)

	def forward(self, X):
		batch_sents = X['sents']
		batch_pos1s = X['pos1s']
		batch_pos2s = X['pos2s']

		word_embs = self.word_embedding(batch_sents)
		pos1_embs = self.pos1_embedding(batch_pos1s)
		pos2_embs = self.pos2_embedding(batch_pos2s)

		input_feature = torch.cat([word_embs, pos1_embs, pos2_embs], dim=2).transpose(0,1)

		# (seq_len,batch_size,vector_size)
		embeds = self.dropout_emb(input_feature)
		batch_size = X['sents'].shape[0]

		# LSTM layer
		embeds = self.dropout_emb(input_feature)
		lstm_out, state = self.lstm(embeds,
									self.begin_state(batch_size))  # lstm_out : (seq_len, batch_size, hidden_dim)
		lstm_out = lstm_out.permute(1, 2, 0)  # (batch_size, hidden_dim, seq_len)
		lstm_out = self.dropout_lstm(lstm_out)

		# 注意力层处理
		att_out = self.attention(lstm_out)  # (batch_size, hidden_dim, 1)
		att_out = att_out.squeeze(2)  # 移除最后一个维度，变为 (batch_size, hidden_dim)

		# 全连接层
		out = self.dense(att_out)  # 经过一个全连接矩阵 W*h + b
		return out


class BiLSTM_MaxPooling(nn.Module):
	def __init__(self,data_loader,params):
		super(BiLSTM_MaxPooling, self).__init__()
		embedding_vectors = data_loader.get_loaded_embedding_vectors()

		self.out_size = len(data_loader.label2idx)
		self.hidden_dim = params.hidden_dim
		self.batch_size = params.batch_size
		self.feature_dim = params.word_emb_dim + params.pos_emb_dim * 2
		self.lstm = nn.LSTM(self.feature_dim,self.hidden_dim//2,bidirectional=True)

		self.word_embedding = nn.Embedding.from_pretrained(embedding_vectors,freeze=False)
		self.pos1_embedding = nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
		self.pos2_embedding = nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)

		self.att_weight = nn.Parameter(torch.randn((self.batch_size, 1, self.hidden_dim)))

		self.dense = nn.Linear(self.hidden_dim,self.out_size)
		self.device = None
		self.loss = nn.CrossEntropyLoss()
		if params.gpu >= 0:
			self.device = self.cuda(device=params.gpu)

	def begin_state(self, batch_size):  # Modified to accept batch_size
		state = (
			torch.zeros(2, batch_size, self.hidden_dim // 2),
			torch.zeros(2, batch_size, self.hidden_dim // 2))
		if self.device:
			return state.to(self.device)
		else:
			return state


	def forward(self, X):
		batch_sents = X['sents']
		batch_pos1s = X['pos1s']
		batch_pos2s = X['pos2s']

		word_embs = self.word_embedding(batch_sents)
		pos1_embs = self.pos1_embedding(batch_pos1s)
		pos2_embs = self.pos2_embedding(batch_pos2s)

		input_feature = torch.cat([word_embs, pos1_embs, pos2_embs], dim=2).transpose(0,1)
		batch_size = X['sents'].shape[0]

		# LSTM layer
		lstm_out, state = self.lstm(input_feature, self.begin_state(batch_size))  # Adjusted
		out,_ = torch.max(lstm_out,dim=0) # (1,batch_size,hidden_dim)
		out = self.dense(out.squeeze(0))  # 经过一个全连接矩阵 W*h + b
		return out