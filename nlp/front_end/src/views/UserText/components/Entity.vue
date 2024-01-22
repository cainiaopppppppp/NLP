<template>
  <div id="app">
    <el-container style="margin-top: 0px;">
      <el-main>
        <el-row style="margin-bottom: 20px;">
          <el-col :span="24">
            <el-input type="textarea" placeholder="请输入文本..." v-model="inputText" :rows="4"></el-input>
          </el-col>
        </el-row>
        <el-row style="margin-top: 20px;">
          <el-col :span="8">
            <el-button type="primary" @click="extractEntities('CNN')">CNN模型</el-button>
            <div v-if="results.CNN.entities.length">
              实体: <el-tag v-for="(entity, index) in results.CNN.entities" :key="index">{{ entity }}</el-tag>
            </div>
            <div v-if="results.CNN.relation">
              关系: <el-tag>{{ results.CNN.relation }}</el-tag>
            </div>
          </el-col>
          <el-col :span="8">
            <el-button type="primary" @click="extractEntities('BiLSTM_Att')">BiLSTM_Att模型</el-button>
            <div v-if="results.BiLSTM_Att.entities.length">
              实体: <el-tag v-for="(entity, index) in results.BiLSTM_Att.entities" :key="index">{{ entity }}</el-tag>
            </div>
            <div v-if="results.BiLSTM_Att.relation">
              关系: <el-tag>{{ results.BiLSTM_Att.relation }}</el-tag>
            </div>
          </el-col>
          <el-col :span="8">
            <el-button type="primary" @click="extractEntities('BiLSTM_MaxPooling')">BiLSTM_MaxPooling模型</el-button>
            <div v-if="results.BiLSTM_MaxPooling.entities.length">
              实体: <el-tag v-for="(entity, index) in results.BiLSTM_MaxPooling.entities" :key="index">{{ entity }}</el-tag>
            </div>
            <div v-if="results.BiLSTM_MaxPooling.relation">
              关系: <el-tag>{{ results.BiLSTM_MaxPooling.relation }}</el-tag>
            </div>
          </el-col>
        </el-row>
        <el-row style="margin-top: 20px;">
          <el-col :span="8" >
            <el-input v-model="selectedEntity" placeholder="输入实体名称"></el-input>
          </el-col>
          <el-col :span="4" style="margin-left: 5px;">
            <el-button type="primary" @click="queryEntityRelations">查询实体关系</el-button>
          </el-col>
        </el-row>
        <el-row style="margin-top: 20px;">
          <el-col :span="24">
            <div id="entity-network" style="width: 100%; height: 600px;"></div>
          </el-col>
        </el-row>
      </el-main>
    </el-container>
  </div>
</template>

<script>
import { DataSet, Network } from "vis-network/standalone";
import { ref, onMounted } from 'vue';
import axios from 'axios';

export default {
  setup() {
    const selectedEntity = ref('');
    const inputText = ref('');
    const results = ref({
      CNN: { entities: [], relation: '' },
      BiLSTM_Att: { entities: [], relation: '' },
      BiLSTM_MaxPooling: { entities: [], relation: '' }
    });
    const entities = ref([]);
    const nodes = new DataSet();
    const edges = new DataSet();
    let network = null;

    onMounted(() => {
      const container = document.getElementById('entity-network');
      const data = {
        nodes: nodes,
        edges: edges,
      };
      network = new Network(container, data, {});
    });

    const extractEntities = async (modelType) => {
      try {
        const response = await axios.post('http://localhost:5000/extract', {
          sentence: inputText.value,
          model_type: modelType
        });
        results.value[modelType].entities = response.data.entities;
        results.value[modelType].relation = response.data.relation;
      } catch (error) {
        console.error('Error fetching entities and relations:', error);
      }
    };

    const queryEntityRelations = async (entity) => {
      try {
        nodes.clear();
        edges.clear();

        const response = await axios.post('http://localhost:5000/query_relations_from_text', {
          entity: selectedEntity.value
        });

        response.data.forEach(relation => {
          const fromId = relation[0];
          const toId = relation[2];
          const relationType = relation[1];

          if (!nodes.get(fromId)) {
            nodes.add({ id: fromId, label: fromId });
          }
          if (!nodes.get(toId)) {
            nodes.add({ id: toId, label: toId });
          }

          // 检查边是否存在，如果不存在则添加
          const edgeExists = edges.get({
            filter: function (item) {
              return (item.from === fromId && item.to === toId) || (item.from === toId && item.to === fromId);
            }
          });
          if (edgeExists.length === 0) {
            edges.add({ from: fromId, to: toId, label: relationType });
          }
        });

        network.setData({ nodes: nodes, edges: edges });
      } catch (error) {
        console.error('Error fetching relations for entity:', error);
      }
    };

    return {
      inputText,
      selectedEntity,
      results,
      entities,
      extractEntities,
      queryEntityRelations
    };
  }
}
</script>

<style>
/* Your existing styles */
#entity-network {
  width: 100%;
  height: 600px;
  border: 1px solid #ddd;
}
</style>
?