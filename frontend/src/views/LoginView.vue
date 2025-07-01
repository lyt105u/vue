<template>
  <div class="home" style="min-height: auto;">
    <img alt="Vue logo" src="../assets/logo4.png" style="max-width: 200px; max-height: 200px;">
    <h1 class="display-5 fw-bold text-body-emphasis">MLAS</h1>
    <p>Machine Learning Analysis System</p>
    <div class="form-floating">
      <input
        v-model="selected.username"
        type="password"
        class="form-control"
        id="floatingUsername"
        placeholder="Username"
        :disabled="loading"
        @keyup.enter="login"
      >
      <label for="floatingUsername">{{ $t('lblUserName') }}</label>
    </div>
    <div v-if="errors.username" class="text-danger small">{{ errors.username }}</div>
    <button v-if="!loading" @click="login" class="btn btn-primary w-100 py-2" type="button">{{ $t('lblSignIn') }}</button>
    <button v-if="loading" class="btn btn-primary w-100 py-2" type="button" disabled>
      <span class="spinner-border spinner-border-sm" aria-hidden="true"></span>
    </button>
    <div style="padding-top: 20px;">
      <select class="form-select" v-model="$i18n.locale" :disabled="loading" @change="saveLocale">
        <option v-for="locale in $i18n.availableLocales" :key="`locale-${locale}`" :value="locale">
          {{ localeLabels[locale] }}
        </option>
      </select>
    </div>
  </div>
  <div class="container">
    <footer class="flex-wrap align-items-center py-3 my-4 border-top">
      <p class="text-center text-body-secondary" v-html="this.$t('msgFooter1')"></p>
      <p class="text-center text-body-secondary" v-html="this.$t('msgFooter2')"></p>
    </footer>
  </div>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
  <ModalNotification ref="modalNotification" :title="modal.title" :content="modal.content" :icon="modal.icon" />
</template>

<script>
import axios from 'axios'
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
  created() {
    // 登出後回到此頁，重建 menu 以移除使用者名稱
    this.$root.buildMenu()
  },
  mounted() {},
  computed: {},
  watch: {},
  methods: {
    saveLocale() {
      sessionStorage.setItem('locale', this.$i18n.locale)
    },

    validateForm() {
      this.errors = {}
      let isValid = true

      // Username
      if (!this.selected.username) {
        this.errors.username = this.$t('msgValRequired')
        isValid = false
      } else if (!/^[A-Za-z]+$/.test(this.selected.username)) {
        this.errors.username = this.$t('msgValAlphaOnly')
        isValid = false
      }

      return isValid
    },

    async login() {
      if (!this.validateForm()) {
        return
      }
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/login`, {
          username: this.selected.username,
        })
        if (response.data.status == "success") {
          sessionStorage.setItem('username', response.data.username)
          // 重建 menu 以顯示使用者名稱
          this.$root.buildMenu()
          this.$router.push('/')
        } else if (response.data.status == "unauthorized") {
          this.modal.title = this.$t('lblError')
          this.modal.content = this.selected.username + " " + this.$t('msgNotOnList')
          this.modal.icon = 'error'
          this.openModalNotification()
        } else if (response.data.status == "error") {
          this.modal.title = this.$t('lblError')
          this.modal.content = response.data.message
          this.modal.icon = 'error'
          this.openModalNotification()
        }
      } catch (error) {
        this.modal.title = this.$t('lblError')
        this.modal.content = error
        this.modal.icon = 'error'
        this.openModalNotification()
      }
      this.loading = false
    },

    openModalNotification() {
      if (this.$refs.modalNotification) {
        this.$refs.modalNotification.openModal()
      } else {
        console.error("ModalNotification component not found.")
      }
    },
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
    width: 220px;
  }
  button {
    max-width: 220px;
  }
  input {
    max-width: 220px;
  }
</style>
