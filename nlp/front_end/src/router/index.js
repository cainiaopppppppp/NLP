// /src/router/index.js
import { createRouter, createWebHistory } from 'vue-router'
import Login from "@/views/Login/Login.vue"
import Layout from "@/views/Layout/index.vue"
import Register from "@/views/Register/Register.vue"
import ForgetPassword from "@/views/ForgetPassword.vue"
import Graph from "@/views/Graph/Graph.vue"
import Chosenentity from "@/views/Chosenentity/Chosenentity.vue"
import Usertext from "@/views/UserText/index.vue"

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      component: Layout
    },
    {
      path: '/login',
      component: Login
    },
    {
      path: '/forget',
      component: ForgetPassword
    },
    {
      path: '/register',
      component: Register
    },
    {
      path: '/view',
      component: Graph
    },
    {
      path: '/entity_choose',
      component: Chosenentity
    },
    {
      path: '/text',
      component: Usertext
    },
  ],
  scrollBehavior() {
    return {
      top: 0
    }
  }
})

export default router
