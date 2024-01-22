<!-- /src/components/Nav.vue -->
<script setup>
import { ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useUserStore } from '../stores/stores'
import pinia from '../stores/index.js'

const router = useRouter()

const user = useUserStore(pinia)

function reloadPage() {
    router.replace(router.currentRoute.value.fullPath)
}
function toggleLogin() {
    user.isLoggedIn = !user.isLoggedIn
    // isLoggedIn = user.isLoggedIn
    reloadPage()
}
watch(user.isLoggedIn, (newValue, oldValue) => {
    if (newValue !== oldValue) {
        reloadPage()
    }
})
</script>
<template>
    <nav>
        <div class="nav-left">
            <img src="../pictures/logo.jpg" class="logo" />
            <span class="app-name">信息抽取</span>
        </div>

        <div class="nav-right">
            <!-- <template v-if="user.isLoggedIn"> -->
                <router-link to="/">主页</router-link>
                <router-link to="/view">数据图</router-link>
                <router-link to="/entity_choose">可选实体</router-link>
                <!-- <router-link to="/opinion">已有图谱可视化</router-link> -->
                <router-link to="/text">实时分析</router-link>
                <router-link to="/" @click="toggleLogin">登出</router-link>
            <!-- </template>

            <template v-else>
                <router-link to="/">主页</router-link>
                <router-link to="/login">登录</router-link>
                <router-link to="/register">注册</router-link>
            </template> -->
        </div>
    </nav>
</template>
  
<style scoped>
nav {
    /* position: absolute; */
    position: fixed;
    top: 0;
    width: 100%;
    top: 0px;
    left: 0;
    right: 0;
    z-index: 999;
    display: flex;
    /* align-items: stretch; */
    justify-content: space-between;
    /* align-items: center; */
    background-color: #134202;
    color: #fff;
    padding: 5px;
    /* width: 100vw; */
}

.logo {
    height: 2rem;
    margin: 0.5rem 2rem 0 0.5rem;
    display: inline-block;
}

.app-name {
    font-weight: bold;
    font-size: 1.5rem;
    display: inline-block;
}

.nav-right {
    display: flex;
    align-items: center;
}

.nav-right a {
    color: #fff;
    text-decoration: none;
    margin-right: 1rem;
}

@media (max-width: 600px) {
    nav {
        flex-direction: column;
    }

    .nav-left,
    .nav-right {
        margin-bottom: 10px;
    }
}
</style>