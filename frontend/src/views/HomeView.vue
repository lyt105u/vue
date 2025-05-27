<template>
  <div class="home">
    <img alt="Vue logo" src="../assets/logo.png">
    <h1>{{ $t('lblMlas') }} v0.5</h1>
    <p>2025/04/30</p>
    <!-- <HelloWorld msg="Welcome to Your Vue.js App"/> -->
    <select class="form-select" v-model="$i18n.locale">
      <option v-for="locale in $i18n.availableLocales" :key="`locale-${locale}`" :value="locale">{{ localeLabels[locale] }}</option>
    </select>
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
    return {
      localeLabels: {
        en: 'English',
        zh: '中文',
      }
    }
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
