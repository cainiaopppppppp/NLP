# /APP/views.py
# 路由 + 视图函数

import os

import torch
from flask import jsonify, render_template, Blueprint, request, redirect, session
from neo4j import GraphDatabase
from . import uttts
from .model import net
from .data_loader import DataLoader as RelationDataLoader
from .entity_predict import build_word2idx, load_model as load_entity_model, predict as predict_entity
from .relation_predict import process_sentence, create_batch_data, predict as predict_relation, load_labels_to_dict, \
    load_model as load_extract_model

blue = Blueprint("user", __name__)


@blue.route("/")
@blue.route("/home/")
def home():
    # username = request.cookies.get("user")
    username = session.get("user")
    return render_template("home.html",username = username)


@blue.route("/login/", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    elif request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if  username == "dp" and password == "666":
            response = redirect("/home/")
            # response.set_cookie("user",username,max_age=30*24*3600)
            session["user"] = username

            return response

        else :
            return "用户名或密码错误"

@blue.route("/logout/")
def logout():
    response = redirect("/home/")
    # response.delete_cookie("user")
    session.pop("user")
    return response

# 实体识别模型全局变量
entity_model = None
word2idx = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 关系抽取模型全局变量
relation_models = {}
relation_data_loader = None
label_dict = None

# 实体识别模型
def load_entity_resources():
    global entity_model, word2idx
    vocab_size = 7860
    embedding_dim = 50
    hidden_dim = 256
    num_tags = 2
    dropout = 0.2
    model_path = 'F:/nlp/Final/back_end/APP/entity_model/best_model.pt'
    vocab_path = 'F:/nlp/Final/back_end/APP/entities.txt'

    word2idx = build_word2idx(vocab_path)
    entity_model = load_entity_model(model_path, vocab_size, embedding_dim, hidden_dim, num_tags, dropout)
    entity_model.to(device)

# 关系抽取模型
def load_relation_resources():
    global relation_models, relation_data_loader, label_dict
    model_dir = 'F:/nlp/Final/back_end/APP/base_model'
    data_dir = 'F:/nlp/Final/back_end/APP/data/SemEval2010_task8'
    embedding_file = 'F:/nlp/Final/back_end/APP/data/embeddings/vector_50d.txt'
    word_emb_dim = 50
    max_len = 100
    pos_dis_limit = 50
    model_types = ['CNN', 'BiLSTM_Att', 'BiLSTM_MaxPooling']

    relation_data_loader = RelationDataLoader(data_dir, embedding_file, word_emb_dim, max_len, pos_dis_limit, pad_word='<pad>', unk_word='<unk>', other_label='Other', gpu=-1)
    relation_data_loader.load_embeddings_from_file_and_unique_words(embedding_file, emb_delimiter=' ', verbose=True)

    label_dict = load_labels_to_dict(data_dir + '/labels.txt')

    for model_type in model_types:
        params_path = os.path.join(model_dir, model_type, 'params.json')
        params = uttts.Params(params_path)
        params.gpu = -1
        model = load_extract_model(model_dir, model_type, relation_data_loader, params)
        relation_models[model_type] = model


@blue.route('/extract', methods=['POST'])
def extract():
    load_entity_resources()
    load_relation_resources()

    data = request.json
    sentence = data['sentence']
    model_type = data.get('model_type', 'CNN')  # 默认使用CNN模型

    # 实体预测
    entities = predict_entity(entity_model, sentence, word2idx, device)
    print(entities)

    # 句子中插入实体标识
    tagged_sentence = insert_entity_tags(sentence, entities)

    # 关系抽取
    processed_sentence = process_sentence(tagged_sentence, relation_data_loader)
    batch_data = create_batch_data(relation_data_loader, [processed_sentence])
    relation = predict_relation(batch_data, relation_models[model_type], label_dict)

    return jsonify({'entities': entities, 'relation': relation})

def insert_entity_tags(sentence, entities):
    if len(entities) < 2:
        raise ValueError("Less than two entities found in the sentence.")

    tagged_sentence = sentence
    for i, entity in enumerate(entities):
        if i >= 2:  # 只考虑前两个
            break

        start = tagged_sentence.find(entity)
        if start == -1:
            raise ValueError(f"Entity '{entity}' not found in the sentence.")

        end = start + len(entity)
        tagged_sentence = tagged_sentence[:start] + f"<e{i+1}>" + entity + f"</e{i+1}>" + tagged_sentence[end:]

    return tagged_sentence



class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__password = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__password))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, parameters=None, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response

    # 查询实体关系的函数
    def query_entity_relations(self, entity_name):
        query = """
        MATCH (e1:Entity)-[r]-(e2:Entity) 
        WHERE e1.name = $entity_name OR e2.name = $entity_name
        RETURN e1.name, type(r), e2.name
        """
        return self.query(query, parameters={'entity_name': entity_name})

    # 查询所有实体及关系的函数
    def query_all_relations(self):
        query = """
        MATCH (e1:Entity)-[r]->(e2:Entity)
        RETURN e1.name, type(r), e2.name
        """
        return self.query(query)

# 创建Neo4j连接
neo4j_conn = Neo4jConnection(uri="neo4j://localhost:7687", user="neo4j", pwd="12345678")

# 路由：根据文本中的实体查询关系
@blue.route('/query_relations_from_text', methods=['POST'])
def query_relations_from_text():
    data = request.json
    entity = data['entity']
    relations = neo4j_conn.query_entity_relations(entity)
    return jsonify(relations)

# 路由：根据任意实体查询关系
@blue.route('/query_relations', methods=['POST'])
def query_relations():
    data = request.json
    entity = data['entity']
    print(f"Received entity: {entity}")  # 打印接收到的实体
    relations = neo4j_conn.query_entity_relations(entity)
    print(relations)
    return jsonify(relations)

# 路由：查询并展示所有实体和关系
@blue.route('/show_all_relations', methods=['GET'])
def show_all_relations():
    relations = neo4j_conn.query_all_relations()
    return jsonify(relations)