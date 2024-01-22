from neo4j import GraphDatabase

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


# 连接到Neo4j数据库
conn = Neo4jConnection(uri="neo4j://localhost:7687", user="neo4j", pwd="12345678")


# 读取数据
def read_data(sentences_file, labels_file):
    with open(sentences_file, 'r') as f:
        sentences = f.readlines()

    with open(labels_file, 'r') as f:
        labels = f.readlines()

    assert len(sentences) == len(labels), "Sentences and labels files should have the same number of lines."

    data = []
    for sentence, label in zip(sentences, labels):
        entities = sentence.strip().split("\t")
        relation = label.strip()
        data.append((entities[0], entities[1], relation))
    return data


# 从文件中读取实体和关系
data = read_data('./data/SemEval2010_task8/train/sentences.txt', './data/SemEval2010_task8/train/labels.txt')

# 将数据插入Neo4j数据库
for entity1, entity2, relation in data:
    if relation != 'Other':
        relation_type, direction = relation.split("(")
        e1, e2 = direction[:-1].split(',')
        if e1 == 'e1' and e2 == 'e2':
            create_query = f"""
            MERGE (e1:Entity {{name: $entity1}})
            MERGE (e2:Entity {{name: $entity2}})
            MERGE (e1)-[:`{relation_type}`]->(e2)
            """
        else:
            create_query = f"""
            MERGE (e1:Entity {{name: $entity1}})
            MERGE (e2:Entity {{name: $entity2}})
            MERGE (e2)-[:`{relation_type}`]->(e1)
            """
    else:
        create_query = """
        MERGE (e1:Entity {name: $entity1})
        MERGE (e2:Entity {name: $entity2})
        MERGE (e1)-[:Other]->(e2)
        """
    conn.query(create_query, parameters={'entity1': entity1, 'entity2': entity2})

# 关闭连接
conn.close()
