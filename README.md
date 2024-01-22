# 环境

front 使用 Vue + Element-plus

end 使用 Flask

Model_code代码部署

# requirements

python  == 3.7

torch == 1.12.1+cu113

numpy

sklearn

tqdm

neo4j-5-15.0

glove.6B.50d.txt

vector_50d.txt


# 运行步骤

1.提取标注语料中的关系和实体

```
python build_semeval_dataset.py
```
这里需要在**./data/SemEval2010_task8**文件夹下先放置原始语料文件**TRAIN.TXT**和**TEST.TXT**

完成后会在**./data/SemEval2010_task8/train**和**./data/SemEval2010_task8/test**下生成labels.txt和sentences.txt

2.生成词表

```
python build_vocab.py --data_dir data/SemEval2010_task8
```

完成后会在**./data/SemEval2010_task8**下生成**words.txt**和**labels.txt**

```
python entity_build_vocab.py 
```

完成后会在**./data/SemEval2010_task8**下生成**entities.txt

3.训练并评估

```
python train.py --data_dir data/SemEval2010_task8 --model_dir experiments/base_model --model_name CNN
```

训练关系抽取模型，其中参数model_name用于选择模型，共有三种模型可选，对应参数选项分别为“CNN”,"BiLSTM_Att","BiLSTM_MaxPooling",默认为CNN.若输入其他模型参数则会报错。

```
python entity/entity_model.py 
```

训练实体识别模型

4.预测

```
python relation_predict.py 
```

关系预测

```
python entity/entity_predict.py 
```

实体识别预测

# 文件结构说明

./tools：数据加载和预处理函数，以及其他utils函数

./data/SemEval2010_task8：语料数据

./data/embeddings：预训练词向量

./model/net：关系抽取各模型实现细节

./entity: 实体识别模型实现细节

./experiments/entity_model：实体识别模型训练后得到的模型

./experiments/base_model：关系抽取模型训练后得到的模型

./graph.py: 抽取到的数据存入图数据库中（neo4j）