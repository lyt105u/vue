<template>
  <div class="about">
    <h1>{{ $t('lblSettings') }}</h1>
    <!-- <h6 class="text-body-secondary">{{ $t('msgUploadDescription') }}</h6> -->
  </div>

  <!-- hr -->
  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <div class="row g-3" style="margin-top: 16px">
    <!-- Tabular Data -->
    <div class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblTabularData') }}</label>
      <div class="col-sm-7" style="max-height: 300px; overflow-y: auto">
        <table class="table table-hover">
          <thead>
            <tr>
              <th scope="col">
                <input type="checkbox" v-model="selectAllFile" @change="toggleSelectAllFile" :disabled="loading" />
              </th>
              <th scope="col"></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(file, index) in displayedFiles" :key="index">
              <td>
                <input type="checkbox" v-model="file.selected" :disabled="loading" />
              </td>
              <td>{{ file.name }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="col-sm-2">
        <button
          @click="deleteSelectedFile"
          class="btn btn-outline-primary"
          :disabled="loading || selectedFileCnt==0"
        >
          <i class="fa fa-trash me-1"></i>
          {{ $t('lblDelete')}}
        </button>
      </div>
    </div>

    <!-- Trained Model -->
    <div class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblTrainedModel') }}</label>
      <div class="col-sm-7" style="max-height: 300px; overflow-y: auto">
        <table class="table table-hover">
          <thead>
            <tr>
              <th scope="col">
                <input type="checkbox" v-model="selectAllModel" @change="toggleSelectAllModel" :disabled="loading" />
              </th>
              <th scope="col"></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(file, index) in displayedModels" :key="index">
              <td>
                <input type="checkbox" v-model="file.selected" :disabled="loading" />
              </td>
              <td>{{ file.name }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="col-sm-2">
        <button
          @click="deleteSelectedModel"
          class="btn btn-outline-primary"
          :disabled="loading || selectedModelCnt==0"
        >
          <i class="fa fa-trash me-1"></i>
          {{ $t('lblDelete')}}
        </button>
      </div>
    </div>

    <!-- Language -->
    <div class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblLanguage') }}</label>
      <div class="col-sm-7">
        <select class="form-select" v-model="$i18n.locale" :disabled="loading" @change="saveLocale">
          <option v-for="locale in $i18n.availableLocales" :key="`locale-${locale}`" :value="locale">
            {{ localeLabels[locale] }}
          </option>
        </select>
      </div>
    </div>

    <!-- Logout -->
    <div class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblLogout') }}</label>
      <div class="col-sm-2">
        <button @click="logout" v-if="!loading" class="btn btn-danger" type="button">{{ $t('lblLogout') }}</button>
        <button v-if="loading" class="btn btn-danger" type="button" disabled>
          <span class="spinner-border spinner-border-sm" aria-hidden="true"></span>
        </button>
      </div>
    </div>
  </div>

  <!-- <div style="padding-top: 20px;">
    <select class="form-select" v-model="$i18n.locale" :disabled="loading" @change="saveLocale">
      <option v-for="locale in $i18n.availableLocales" :key="`locale-${locale}`" :value="locale">
        {{ localeLabels[locale] }}
      </option>
    </select>
  </div> -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
  <ModalNotification ref="modalNotification" :title="modal.title" :content="modal.content" :icon="modal.icon" />
  <ModalNotification ref="modalFinishDeleteRef" :title="modal.title" :content="modal.content" :icon="modal.icon" :onUserDismiss="reloadPage" />
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
      preview_data: {
        columns: [],
        preview: [],
        total_rows: 0,
        total_columns: 0
      },
      fileOptions: [],
      selectAllFile: false,
      displayedFiles: [],

      modelOptions: [],
      selectAllModel: false,
      displayedModels: [],
    }
  },
  async created() {
    await this.listFiles()
    this.displayedFiles = this.fileOptions.map(name => ({
      name,
      selected: false
    }))
    this.displayedModels = this.modelOptions.map(name => ({
      name,
      selected: false
    }))
  },
  mounted() {},
  computed: {
    selectedFileCnt() {
      return this.displayedFiles.filter(file => file.selected).length;
    },
    selectedModelCnt() {
      return this.displayedModels.filter(file => file.selected).length;
    }
  },
  watch: {
    // 監聽勾選狀態 → 自動更新全選 checkbox
    displayedFiles: {
      handler(newFiles) {
        const allChecked = newFiles.length > 0 && newFiles.every(f => f.selected)
        this.selectAllFile = allChecked
      },
      deep: true
    },
    displayedModels: {
      handler(newFiles) {
        const allChecked = newFiles.length > 0 && newFiles.every(f => f.selected)
        this.selectAllModel = allChecked
      },
      deep: true
    }
  },
  methods: {
    async listFiles() {
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/list-files`, {
          folder_path: `upload/${sessionStorage.getItem('username')}`, // upload/
          ext1: 'pkl',
          ext2: 'zip',
          ext3: 'json',
        })
        if (response.data.status == "success") {
          this.modelOptions = response.data.files
        } else if (response.data.status == "error") {
          this.modal.title = this.$t('lblError')
          this.modal.content = response.data.message
          this.modal.icon = 'error'
          this.openModalNotification()
          this.loading = false
          return
        }
      } catch (error) {
        this.modal.title = this.$t('lblError')
        this.modal.content = error
        this.modal.icon = 'error'
        this.openModalNotification()
        this.loading = false
        return
      }

      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/list-files`, {
          folder_path: `upload/${sessionStorage.getItem('username')}`, // upload/
          ext1: 'csv',
          ext2: 'xlsx',
        })
        if (response.data.status == "success") {
          this.fileOptions = response.data.files
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

    toggleSelectAllFile() {
      this.displayedFiles.forEach(file => {
        file.selected = this.selectAllFile
      })
    },
    async deleteSelectedFile() {
      const selectedFiles = this.displayedFiles
        .filter(file => file.selected)
        .map(file => file.name)
      
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/delete-files`, {
          folder_path: `upload/${sessionStorage.getItem('username')}`, // upload/
          files: selectedFiles,
        })
        if (response.data.status == "success") {
          this.modal.title = this.$t('lblDelete')
          this.modal.icon = 'success'
          this.modal.content = this.$t('msgFilesDeleted') + response.data.deleted
          this.openModalFinishDelete()
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

    toggleSelectAllModel() {
      this.displayedModels.forEach(file => {
        file.selected = this.selectAllModel
      })
    },
    async deleteSelectedModel() { // 照抄 deleteSelectedFile
      const selectedModels = this.displayedModels
        .filter(file => file.selected)
        .map(file => file.name)
      
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/delete-files`, {
          folder_path: `upload/${sessionStorage.getItem('username')}`, // upload/
          files: selectedModels,
        })
        if (response.data.status == "success") {
          this.modal.title = this.$t('lblDelete')
          this.modal.icon = 'success'
          this.modal.content = this.$t('msgFilesDeleted') + response.data.deleted
          this.openModalFinishDelete()
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

    saveLocale() {
      sessionStorage.setItem('locale', this.$i18n.locale)
    },

    logout() {
      sessionStorage.removeItem('username')
      this.$router.push('/login')
    },

    openModalNotification() {
      if (this.$refs.modalNotification) {
        this.$refs.modalNotification.openModal()
      } else {
        console.error("ModalNotification component not found.")
      }
    },

    openModalFinishDelete() {
      if (this.$refs.modalFinishDeleteRef) {
        this.$refs.modalFinishDeleteRef.openModal()
      } else {
        console.error("ModalNotification component not found.")
      }
    },

    reloadPage() {
      window.location.reload()
    },
  },
}
</script>

<style scoped>
</style>
