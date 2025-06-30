<template>
  <div class="home" style="min-height: auto;">
    <img alt="Vue logo" src="../assets/logo4.png" style="max-width: 200px; max-height: 200px;">
    <h1 class="display-5 fw-bold text-body-emphasis">MLAS</h1>
    <p>Machine Learning Analysis System</p>
  </div>
  <div class="container">
    <footer class="flex-wrap align-items-center py-3 my-4 border-top">
      <p class="text-center text-body-secondary" v-html="this.$t('msgFooter1')"></p>
      <p class="text-center text-body-secondary" v-html="this.$t('msgFooter2')"></p>
    </footer>
  </div>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
</template>

<script>
// @ is an alias to /src
// import HelloWorld from '@/components/HelloWorld.vue'

export default {
  name: 'HomeView',
  components: {},
  data() {
    return {}
  },
}
</script>


<style scoped>
.home {
  display: flex;
  flex-direction: column;
  align-items: center;      /* 水平置中 */
  min-height: 100vh;        /* 占滿整個視窗高度 */
  text-align: center;       /* 文字置中 */
}
select {
   width: 200px;
}
</style>
