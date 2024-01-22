<template>
  <div>
    <el-button type="primary" @click="loadData">加载数据</el-button>
    <div id="network" style="width: 100%; height: 600px;"></div>
  </div>
</template>

<script>
import { Network, DataSet } from "vis-network/standalone";
import { ref, onMounted } from 'vue';
import axios from 'axios';

export default {
  setup() {
    const nodes = new DataSet();
    const edges = new DataSet();
    let network = null;
    const isLoading = ref(false); // 新增 isLoading 变量

    onMounted(() => {
      const container = document.getElementById('network');
      network = new Network(container, { nodes, edges }, {});
    });

    const fetchAllRelations = async () => {
      try {
        isLoading.value = true; // 设置 isLoading 为 true，表示正在加载

        const response = await axios.get('http://localhost:5000/show_all_relations');
        const relations = response.data;

        // 清空现有的图形数据
        nodes.clear();
        edges.clear();

        relations.forEach(relation => {
          // 添加节点，如果它们不存在
          if (!nodes.get(relation[0])) {
            nodes.add({ id: relation[0], label: relation[0] });
          }
          if (!nodes.get(relation[2])) {
            nodes.add({ id: relation[2], label: relation[2] });
          }

          // 检查边是否存在，如果不存在则添加
          const edgeId = relation[0] + "-" + relation[2];
          if (!edges.get(edgeId)) {
            edges.add({ id: edgeId, from: relation[0], to: relation[2], label: relation[1] });
          }
        });

        // 用更新后的数据重新渲染网络图
        network.setData({ nodes: nodes, edges: edges });
      } catch (error) {
        console.error('Error fetching all relations:', error);
      } finally {
        isLoading.value = false; // 加载完成后将 isLoading 设置为 false
      }
    }

    // 新增 loadData 方法，用于触发数据加载
    const loadData = () => {
      if (!isLoading.value) {
        fetchAllRelations();
      }
    }

    return {
      network,
      loadData,
      isLoading
    }
  }
}
</script>

<style>
/* 确保网络图的容器足够大，以展示所有的节点和边 */
#network {
  width: 100%;
  height: 600px;
  border: 1px solid #ddd;
}
</style>
