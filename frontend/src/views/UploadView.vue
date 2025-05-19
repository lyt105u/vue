<template>
  <div class="about">
    <h1>Upload</h1>
  </div>

  <ul class="nav nav-tabs">
    <li class="nav-item">
      <a class="nav-link" 
        :class="{ active: activeTab === 'local' }"
        @click="activeTab = 'local'"
        href="#">
        <i class="fa fa-upload me-1"></i> local
      </a>
    </li>
    <li class="nav-item">
      <a class="nav-link"
        :class="{ active: activeTab === 'cloud' }"
        @click="activeTab = 'cloud'"
        href="#">
      <i class="fa fa-cloud me-1"></i> cloud
      </a>
    </li>
  </ul>

  <div class="home">
    <img alt="Vue logo" src="../assets/logo.png">
    <h1>MLAS v0.5</h1>
    <p>presented on 2025/04/30</p>
    <HelloWorld msg="Welcome to Your Vue.js App"/>
  </div>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
</template>

<script>
export default {
  components: {},
  data() {
    return {
      activeTab: 'local',
    }
  }
}
</script>

<style scoped>
  .nav-link.active {
    background-color: #f2f4f7;
    border-bottom: transparent;
  }
  /* .nav-tabs {
    border-bottom: 0.25px 2c3e50;
  } */
</style>
