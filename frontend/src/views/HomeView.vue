<template>
  <div class="home">
    <img alt="Vue logo" src="../assets/logo.png">
    <h1>MLAS v0.4</h1>
    <p>presented on 2025/04/07</p>
    <HelloWorld msg="Welcome to Your Vue.js App"/>
  </div>
</template>

<script>
// @ is an alias to /src
import HelloWorld from '@/components/HelloWorld.vue'

export default {
  name: 'HomeView',
  components: {
    HelloWorld
  }
}
</script>
