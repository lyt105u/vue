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
      <div class="col-sm-2">
        <button
          @click="deleteSelectedFile"
          class="btn btn-outline-danger"
          :disabled="loading || selectedFileCnt==0"
        >
          <i class="fa fa-trash me-1"></i>
          {{ $t('lblDelete')}}
        </button>
      </div>
    </div>
    <div class="row mb-3">
      <div class="col-sm-12" style="max-height: 300px; overflow-y: auto">
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
    </div>

    <!-- hr -->
    <div class="bd-example m-0 border-0">
      <hr>
    </div>

    <!-- Trained Model -->
    <div class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblModelFile') }}</label>
      <div class="col-sm-2">
        <button
          @click="deleteSelectedModel"
          class="btn btn-outline-danger"
          :disabled="loading || selectedModelCnt==0"
        >
          <i class="fa fa-trash me-1"></i>
          {{ $t('lblDelete')}}
        </button>
      </div>
    </div>
    <div class="row mb-3">
      <div class="col-sm-12" style="max-height: 300px; overflow-y: auto">
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
    </div>

    <!-- hr -->
    <div class="bd-example m-0 border-0">
      <hr>
    </div>

    <div class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblExeResults') }}</label>
      <div class="col-sm-2">
        <button
          @click="deleteSelectedTask"
          class="btn btn-outline-danger"
          :disabled="loading || selectedTaskCnt==0"
        >
          <i class="fa fa-trash me-1"></i>
          {{ $t('lblDelete')}}
        </button>
      </div>
    </div>
    <div class="row mb-3">
      <div class="col-sm-12" style="max-height: 300px; overflow-y: auto">
        <table class="table align-middle">
          <thead>
            <tr>
              <th>
                <input type="checkbox" v-model="selectAllTask" @change="toggleSelectAllTask" :disabled="loading" />
              </th>
              <th>{{ $t('lblApi') }}</th>
              <th>{{ $t('lblUser') }}</th>
              <th>{{ $t('lblStatus') }}</th>
              <th>{{ $t('lblStartTime') }}</th>
              <th>{{ $t('lblEndTime') }}</th>
              <th>{{ $t('lblActions') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="task in tasks" :key="task.username + task.start_time">
              <td>
                <input type="checkbox" v-model="task.selected" :disabled="task.status==='running' || loading" />
              </td>
              <td>{{ task.api }}</td>
              <td>{{ task.username }}</td>
              <td>
                <span v-if="task.status=='success'" class="badge rounded-pill text-bg-success">{{ $t('lblSuccess') }}</span>
                <span v-else-if="task.status=='error'" class="badge rounded-pill text-bg-danger">{{ $t('lblError') }}</span>
                <span v-else-if="task.status=='running'" class="badge rounded-pill text-bg-primary">{{ $t('lblRunning') }}</span>
                <span v-else-if="task.status=='terminated'" class="badge rounded-pill text-bg-secondary">{{ $t('lblTerminate') }}</span>
                <span v-else>{{ $t('lblTerminate') }}</span>
              </td>
              <td>{{ task.start_time }}</td>
              <td>
                <span v-if="task.status === 'success'">{{ task.end_time }}</span>
                <span v-else class="text-muted">N/A</span>
              </td>
              <td>
                <button v-if="task.status!='running'" @click="downloadReport(task.start_time)" type="button" class="btn" :title="$t('lblDownload')">
                  <i class="fa fa-download me-1 text-primary"></i>
                </button>
                <button v-if="task.status=='running'" @click="terminateTask(task.start_time)" type="button" class="btn" :title="$t('lblTerminate')">
                  <i class="fa fa-ban me-1 text-danger"></i>
                </button>
                <button @click="getStatus(task.start_time)" type="button" class="btn" :title="$t('lblInfo')">
                  <i class="fa fa-info-circle me-1 text-primary"></i>
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- hr -->
    <div class="bd-example m-0 border-0">
      <hr>
    </div>

    <!-- Language -->
    <div class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblLanguage') }}</label>
      <div class="col-sm-9">
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

      tasks: [
        // {
        //   "api": "train",
        //   "username": "alice",
        //   "status": "success",
        //   "params": "...",
        //   "start_time": "20250701_202245",
        //   "end_time": "20250701_202323"
        // },
        // {
        //   "task_id": "20250701_202136",
        //   "username": "alice",
        //   "status": "success",
        //   "params": {
        //     "file_path": "upload/alice/高醫csv有答案.csv",
        //     "label_column": "label",
        //     "split_strategy": "train_test_split",
        //     "split_value": "0.8",
        //     "model_name": "xgb",
        //     "username": "alice",
        //     "n_estimators": "100",
        //     "learning_rate": "0.300000012",
        //     "max_depth": "6"
        //   },
        //   "start_time": "20250701_202136",
        //   "api": "train",
        //   "end_time": "20250701_202215"
        // },
        // {
        //   "api": "train",
        //   "username": "carol",
        //   "status": "running",
        //   "params": "...",
        //   "start_time": "20250701_203233",
        // }
      ],
      selectAllTask: false,
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
    await this.listStatus()
  },
  mounted() {},
  computed: {
    selectedFileCnt() {
      return this.displayedFiles.filter(file => file.selected).length
    },
    selectedModelCnt() {
      return this.displayedModels.filter(file => file.selected).length
    },
    selectedTaskCnt() {
      return this.tasks.filter(task => task.selected && task.status !== 'running').length
    },
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
    },
    tasks: {
      handler(newTasks) {
        const selectable = newTasks.filter(t => t.status !== 'running')
        const allChecked = selectable.length > 0 && selectable.every(t => t.selected)
        this.selectAllTask = allChecked
      },
      deep: true
    },
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

    async listStatus() {
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/list-status`, {
          user_dir: `model/${sessionStorage.getItem('username')}`,
        })
        if (response.data.status == "success") {
          this.tasks = response.data.data
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
          this.modal.content = this.$t('msgItemsDeleted') + response.data.deleted
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
          this.modal.content = this.$t('msgItemsDeleted') + response.data.deleted
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

    toggleSelectAllTask() {
      this.tasks.forEach(task => {
        if (task.status !== 'running') {
          task.selected = this.selectAllTask
        }
      })
    },
    async deleteSelectedTask() { // 照抄 deleteSelectedFile
      const selectedTask = this.tasks
        .filter(task => task.selected && task.status !== 'running')
        .map(task => task.start_time)
      
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/delete-folders`, {
          folder_path: `model/${sessionStorage.getItem('username')}`,
          folders: selectedTask,
        })
        if (response.data.status == "success") {
          this.modal.title = this.$t('lblDelete')
          this.modal.icon = 'success'
          this.modal.content = this.$t('msgItemsDeleted') + response.data.deleted
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

    // 下載 icon
    async downloadReport(start_time) {
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/download-report`, {
          task_dir: `model/${sessionStorage.getItem('username')}/${start_time}`,
        }, {
          responseType: 'blob'
        })
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        link.setAttribute('download', `${start_time}.zip`)
        document.body.appendChild(link)
        link.click()
        link.remove()
      } catch (err) {
        this.modal.title = this.$t('lblError')
        this.modal.content = err
        this.modal.icon = 'error'
        this.openModalNotification()
      }
      this.loading = false
    },
    
    // info icon
    async getStatus(start_time) {
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/get-status`, {
          folder_path: `model/${sessionStorage.getItem('username')}/${start_time}`,
        })
        if (response.data.status == "success") {
          this.modal.title = start_time
          this.modal.content = response.data.data
          this.modal.icon = 'info'
          this.openModalNotification()
          this.loading = false
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
      this.loading = false
    },

    // ban icon
    async terminateTask(start_time) {
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/terminate-task`, {
          task_id: start_time,
          username: sessionStorage.getItem('username')
        })
        if (response.data.status == "success") {
          this.modal.title = start_time
          this.modal.content = start_time + " " + this.$t('msgTerminated')
          this.modal.icon = 'success'
          this.openModalFinishDelete()
          this.loading = false
        } else if (response.data.status == "notRunning") {
          this.modal.title = start_time
          this.modal.content = start_time + " " + this.$t('msgNotRunning')
          this.modal.icon = 'info'
          this.openModalFinishDelete()
          this.loading = false
          return
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
