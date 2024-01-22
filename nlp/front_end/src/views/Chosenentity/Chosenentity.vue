<template>
  <div>
    <el-input v-model="selectedEntity" placeholder="输入实体名称"></el-input>
    <el-button type="primary" @click="queryEntityRelations">查询</el-button>
    <div id="entity-network" style="width: 100%; height: 600px;"></div>
  </div>
</template>

<script>
import { DataSet, Network } from "vis-network/standalone";
import { ref, onMounted } from 'vue';
import axios from 'axios';

export default {
  setup() {
    const selectedEntity = ref('');
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

    const queryEntityRelations = async () => {
      try {
        // 清空现有的图形数据
        nodes.clear();
        edges.clear();

        const response = await axios.post('http://localhost:5000/query_relations', {
          entity: selectedEntity.value
        });
        const relations = response.data;

        relations.forEach(relation => {
          const fromId = relation[0];
          const toId = relation[2];
          const relationType = relation[1];

          // 添加节点和边，如果它们不存在
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

        // 用更新后的数据重新渲染网络图
        network.setData({ nodes: nodes, edges: edges });
      } catch (error) {
        console.error('Error fetching relations for entity:', error);
      }
    };

    return {
      selectedEntity,
      queryEntityRelations
    };
  }
};
</script>

<style>
/* 确保网络图的容器足够大，以展示所有的节点和边 */
#entity-network {
  width: 100%;
  height: 600px;
  border: 1px solid #ddd;
}
</style>
