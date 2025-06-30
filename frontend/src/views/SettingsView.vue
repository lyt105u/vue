<template>
  <button @click="logout" class="btn btn-danger">登出</button>
  <div style="padding-top: 20px;">
    <select class="form-select" v-model="$i18n.locale" :disabled="loading" @change="saveLocale">
      <option v-for="locale in $i18n.availableLocales" :key="`locale-${locale}`" :value="locale">
        {{ localeLabels[locale] }}
      </option>
    </select>
  </div>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
  <ModalNotification ref="modalNotification" :title="modal.title" :content="modal.content" :icon="modal.icon" />
</template>

<script>
// import axios from 'axios'
import ModalNotification from "@/components/ModalNotification.vue"

export default {
  components: {
    ModalNotification,
  },
  data() {
    return {
      selected: {
        username: '',
      },
      localeLabels: {
        en: 'English',
        zh: '中文',
      },
      errors: {},
      loading: false,
      modal: {
        title: '',
        content: '',
        icon: 'info',
      },
    }
  },
  created() {},
  mounted() {},
  computed: {},
  watch: {},
  methods: {
    saveLocale() {
      sessionStorage.setItem('locale', this.$i18n.locale)
    },
    logout() {
      sessionStorage.removeItem('username')
      this.$router.push('/login')
    },
  },
}
</script>

<style scoped>
</style>
