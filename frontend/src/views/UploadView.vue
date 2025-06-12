<template>
  <div class="about">
    <h1>{{ $t('lblUpload') }}</h1>
    <h6 class="text-body-secondary">{{ $t('msgUploadDescription') }}</h6>
  </div>

  <!-- tab -->
  <ul class="nav nav-tabs">
    <li class="nav-item">
      <a class="nav-link" 
        :class="{ active: selected.mode === 'local' }"
        @click="selected.mode = 'local'"
        href="#">
        <i class="fa fa-upload me-1"></i> {{ $t('lblUpload') }}
      </a>
    </li>
    <li class="nav-item">
      <a class="nav-link"
        :class="{ active: selected.mode === 'samba' }"
        @click="selected.mode = 'samba'"
        href="#">
        <i class="fa fa-cloud me-1"></i> {{ $t('lblSamba') }} 
      </a>
    </li>
  </ul>

  <form class="row g-3 needs-validation" @submit.prevent="runUpload" style="margin-top: 16px">
    <!-- Local tab -->
    <template v-if="selected.mode=='local'">
      <!-- File Selection -->
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label">{{ $t('lblFileSelection') }}</label>
        <div class="col-sm-8">
          <input @change="handleFileChange" v-if="showInput" type="file" class="form-control" :disabled="loading">
          <div v-if="errors.file" class="text-danger small">{{ errors.file }}</div>
        </div>
      </div>
    </template>

    <!-- Samba tab -->
    <template v-if="selected.mode=='samba'">
      <!-- User Name -->
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label">{{ $t('lblUserName') }}</label>
        <div class="col-sm-8">
          <input v-model="selected.username" class="form-control" type="text" :disabled="loading">
          <div v-if="errors.username" class="text-danger small">{{ errors.username }}</div>
        </div>
      </div>

      <!-- Password -->
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label">{{ $t('lblPassword') }}</label>
        <div class="col-sm-8">
          <input v-model="selected.password" class="form-control" type="password" :disabled="loading">
          <div v-if="errors.password" class="text-danger small">{{ errors.password }}</div>
        </div>
      </div>

      <!-- Remote Path -->
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label">{{ $t('lblRemotePath') }}</label>
        <div class="col-sm-8">
          <input v-model="selected.remote_path" class="form-control" type="text" :disabled="loading">
          <div v-if="errors.remote_path" class="text-danger small">{{ errors.remote_path }}</div>
        </div>
      </div>
    </template>

    <!-- button -->
    <div class="col-12">
      <button v-if="!loading" type="submit" class="btn btn-primary">{{ $t('lblUpload') }}</button>
      <button v-if="loading" class="btn btn-primary" type="button" disabled>
        <span class="spinner-border spinner-border-sm" aria-hidden="true"></span>
      </button>
    </div>
  </form>

  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <!-- Note -->
  <div class="about text-body-secondary">
    <h6>{{ $t('lblNote') }}</h6>
    <ol class="h6">
      <li>{{ $t('msgUploadNote1') }}</li>
      <li>{{ $t('msgUploadNote2') }}</li>
      <li>{{ $t('msgUploadNote3') }}</li>
      <li>{{ $t('msgUploadNote4') }}</li>
    </ol>
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
      loading: false,
      selected:{
        mode: 'local',
        // local
        file: '',
        // samba
        username: '',
        password: '',
        remote_path: '',
      },
      errors: {}, // 檢核用
      showInput: true,  // 移除 input 的 UI 顯示用
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
  watch: {
    "selected.mode"() {
      this.selected.file = ''
      this.selected.username = ''
      this.selected.password = ''
      this.selected.remote_path = ''
      this.errors = {}
      // 移除 UI 顯示
      this.showInput = false
      requestAnimationFrame(() => {
        this.showInput = true
      })
    },
  },
  methods: {
    handleFileChange(event) {
      const file = event.target.files[0]
      if (!file) {
        // 使用者取消選檔 → 什麼都不做或清除選擇
        this.selected.file = ''
        return
      }
      this.selected.file = file.name
    },

    validateForm() {
      this.errors = {}
      let isValid = true

      if (this.selected.mode == 'local') {
        // File Selection
        if (!this.selected.file) {
          this.errors.file = this.$t('msgValRequired')
          isValid = false
        }
      } else if (this.selected.mode == 'samba') {
        // User Name
        if (!this.selected.username) {
          this.errors.username = this.$t('msgValRequired')
          isValid = false
        }
        // Password
        if (!this.selected.password) {
          this.errors.password = this.$t('msgValRequired')
          isValid = false
        }
        // Remote Path
        if (!this.selected.remote_path) {
          this.errors.remote_path = this.$t('msgValRequired')
          isValid = false
        }
      }
      return isValid
    },

    async runUpload() {
      if (!this.validateForm()) {
        return
      }
      if (this.selected.mode == 'local') {
        this.uploadLocal()
      }else if (this.selected.mode == 'samba') {
        this.uploadSamba()
      }      
    },

    async uploadLocal() {
      this.loading = true
      try {
        const fileInput = document.querySelector('input[type="file"]')
        const file = fileInput.files[0]
        const folder = 'upload' // upload/

        const formData = new FormData()
        formData.append("file", file)
        formData.append("folder", folder)
        const response = await axios.post('http://127.0.0.1:5000/upload-local-file', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        if (response.data.status == "success") {
          this.modal.title = this.$t('lblSuccess')
          this.modal.content = this.$t('msgUploadSuccess', { filename: this.selected.file })
          this.modal.icon = 'success'
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
      this.selected.file = ''
      // 移除 UI 顯示
      this.showInput = false
      requestAnimationFrame(() => {
        this.showInput = true
      })
      this.loading = false
    },

    async uploadSamba() {
      this.loading = true
      try {
        this.loading = true
        const response = await axios.post('http://127.0.0.1:5000/upload-samba-file', {
          username: this.selected.username,
          password: this.selected.password,
          remote_path: this.selected.remote_path,
          folder: 'upload'  // upload/
        })
        if (response.data.status == "success") {
          const parts = this.selected.remote_path.split(/[/\\]+/)
          const filename = parts[parts.length - 1]
          this.modal.title = this.$t('lblSuccess')
          this.modal.content = this.$t('msgUploadSuccess', { filename: filename })
          this.modal.icon = 'success'
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
      this.selected.username = ''
      this.selected.password = ''
      this.selected.remote_path = ''
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
  .nav-link.active {
    background-color: #f2f4f7;
  }
  .nav-tabs {
    --bs-nav-tabs-border-color: rgba(44, 62, 80, 0.25); /* 整條 tab 底線顏色 */ /* #2c3e50 */
    --bs-nav-tabs-link-active-border-color: rgba(44, 62, 80, 0.25) rgba(44, 62, 80, 0.25) transparent; /* active tab 底線 */
  }
</style>
