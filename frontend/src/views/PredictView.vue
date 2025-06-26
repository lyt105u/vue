<template>
  <div class="about">
    <h1>{{ $t('lblClinicalPrediction') }}</h1>
    <h6 class="text-body-secondary">{{ $t('msgPredictDescription') }}</h6>
  </div>

  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <form class="row g-3" @submit.prevent="runPredict" style="margin-top: 16px">
    <!-- Trained Model -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblTrainedModel') }}</label>
      <div class="col-sm-8">
        <select class="form-select" aria-label="Small select example" v-model="selected.model_name" :disabled="loading">
          <option v-for="model in modelOptions" :key="model" :value="model">
            {{ model }}
          </option>
        </select>
        <div v-if="errors.model_name" class="text-danger small">{{ errors.model_name }}</div>
      </div>
    </div>

    <!-- Prediction Type -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblPredictionType') }}</label>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.mode" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1_file" value="file"  :disabled="loading">
          <label class="form-check-label" for="gridRadios1">
            {{ $t('lblFilePrediction') }}
          </label>
        </div>
      </div>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.mode" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1_input" value="input"  :disabled="loading">
          <label class="form-check-label" for="gridRadios1">
            {{ $t('lblManualInput') }}
          </label>
        </div>
      </div>
    </div>

    <template v-if="selected.mode=='file'">
      <!-- File Selection -->
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblFileSelection') }}</label>
        <div class="col-sm-8">
          <select class="form-select" aria-label="Small select example" v-model="selected.data_name" :disabled="loading">
            <option v-for="file in fileOptions" :key="file" :value="file">
              {{ file }}
            </option>
          </select>
          <div v-if="errors.data_name" class="text-danger small">{{ errors.data_name }}</div>
        </div>
        <div class="col-sm-1">
          <button v-if="preview_data.columns != 0" class="btn btn-outline-primary" style="white-space: nowrap" type="button" @click="toggleCollapse" :disabled="loading">{{ $t('lblPreview') }}</button>
        </div>
      </div>

      <!-- preview -->
      <div v-if="preview_data.total_rows != 0" class="row mb-3">
        <div class="collapse" ref="collapsePreview">
          <div class="card card-body">
            <div class="table-responsive">
              <table class="table">
                <caption>{{ $t('msgPreviewCaption', { count: preview_data.total_rows }) }}</caption>
                <thead>
                  <tr>
                    <th v-for="col in preview_data.columns" :key="col">
                      {{ col }}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(row, rowIndex) in preview_data.preview" :key="rowIndex">
                    <td v-for="col in preview_data.columns" :key="col">
                      {{ row[col] }}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <!-- Predict Column -->
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblPredictionColumn') }}</label>
        <div class="col-sm-8">
          <input v-model="selected.label_column" class="form-control" type="text" :disabled="loading">
          <div v-if="errors.label_column" class="text-danger small">{{ errors.label_column }}</div>
        </div>
      </div>

      <!-- Results Saved as -->
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblResultsSavedAs') }}</label>
        <div class="col-sm-8">
          <div class="input-group">
            <input v-model="selected.output_name" class="form-control" type="text" :disabled="loading">
            <span class="input-group-text">{{ watched.file_extension }}</span>
          </div>
          <div v-if="errors.output_name" class="text-danger small">{{ errors.output_name }}</div>
        </div>
      </div>
    </template>

    <!-- manual input -->
    <template v-if="selected.mode=='input'">
      <div class="row mb-3" v-for="(row, rowIndex) in rows" :key="rowIndex">
        <!-- 第一行顯示，其他行保持空白，排版用 -->
        <label for="inputEmail3" class="col-sm-3 col-form-label">
          {{ rowIndex === 0 ? $t('lblManualInput') : "" }}
        </label>

        <div v-for="(field, fieldIndex) in row" :key="`${rowIndex}-${fieldIndex}`" class="col-sm-2">
          <div class="form-floating">
            <input
              v-model="selected.input_values[rowIndex * 4 + fieldIndex]"
              type="text" 
              class="form-control" 
              :id="`floatingInput-${rowIndex}-${fieldIndex}`" 
              :placeholder="`Field ${rowIndex * 4 + fieldIndex + 1}`"
              :disabled="loading"
              autocomplete="off"
            />
            <label :for="`floatingInput-${rowIndex}-${fieldIndex}`">
              {{ rowIndex * 4 + fieldIndex + 1 }}
            </label>
          </div>
          <div v-if="errors.input_values && errors.input_values[rowIndex * 4 + fieldIndex]" class="text-danger small">
            {{ errors.input_values[rowIndex * 4 + fieldIndex] }}
          </div>
        </div>
      </div>
    </template>
    
    <!-- button -->
    <div class="col-12">
      <button v-if="!loading" type="submit" class="btn btn-primary">{{ $t('lblPredict') }}</button>
      <button v-if="loading" class="btn btn-primary" type="button" disabled>
        <span class="spinner-border spinner-border-sm" aria-hidden="true"></span>
      </button>
    </div>
  </form>

  <!-- hr -->
  <div v-if="output" class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <!-- output -->
  <div v-if="output" class="row row-cols-1 row-cols-md-3 mb-3 text-center">
    <div v-if="output.status=='success'" class="col">
      <div class="card mb-4 rounded-3 shadow-sm">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">{{ $t('lblPredictionResult') }}</h4>
        </div>
        {{ msgResult }}
      </div>
    </div>

    <!-- SHAP -->
    <div v-if="output.shap_plot" class="col">
      <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalShap(`data:image/png;base64,${output.shap_plot}`, output.shap_importance)" style="cursor: pointer;">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">{{ $t('lblShap') }}</h4>
        </div>
        <img :src="`data:image/png;base64,${output.shap_plot}`" :alt="$t('lblShap')" />
      </div>
    </div>

    <!-- LIME -->
    <div v-if="output.lime_plot" class="col">
      <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalLime(`data:image/png;base64,${output.lime_plot}`, output.lime_example_0)" style="cursor: pointer;">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">{{ $t('lblLime') }}</h4>
        </div>
        <img :src="`data:image/png;base64,${output.lime_plot}`" :alt="$t('lblLime')" />
      </div>
    </div>
  </div>

  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <!-- Note -->
  <div class="about text-body-secondary">
    <h6>{{ $t('lblNote') }}</h6>
    <ol class="h6">
      <li>{{ $t('msgPredictNote1') }}</li>
      <li>{{ $t('msgMissingDataNote') }}</li>
      <li>{{ $t('msgPredictNote2') }}</li>
      <li>{{ $t('msgPredictNote3') }}</li>
      <li>{{ $t('msgPredictNote4') }}</li>
    </ol>
  </div>

  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
  <ModalNotification ref="modalNotification" :title="modal.title" :content="modal.content" :icon="modal.icon" />
  <ModalNotification ref="modalMissingDataRef" :title="modal.title" :content="modal.content" :icon="modal.icon" :primaryButton="modalButtons.primary" :secondaryButton="modalButtons.secondary" :onUserDismiss="closeModalMissingData" />
  <ModalShap ref="modalShapRef" :imageSrc="modal.content" :shapImportance="modal.shap_importance" :columns="preview_data.columns"/>
  <ModalLime ref="modalLimeRef" :imageSrc="modal.content" :lime_example_0="modal.lime_example_0" :columns="preview_data.columns"/>
</template>

<script>
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import axios from 'axios';
import ModalNotification from "@/components/ModalNotification.vue"
import ModalShap from "@/components/ModalShap.vue"
import ModalLime from "@/components/ModalLime.vue"
import { Collapse } from 'bootstrap'

export default {
  components: {
    ModalNotification,
    ModalShap,
    ModalLime,
  },
  data() {
    return {
      selected: {
        model_name: '',
        mode: 'file',
        data_name: '',
        output_name: '',
        input_values: [],
        label_column: ''
      },
      watched: {
        file_extension: '',
      },
      output: '',
      modal: {
        title: '',
        content: '',
        icon: 'info',
        shap_importance: {},
      },
      loading: false,
      errors: {}, // for validation
      preview_data: {
        columns: [],
        preview: [],
        total_rows: 0,
        total_columns: 0
      },
      modelOptions: [],
      fileOptions: [],
      controller: null,
      isAborted: false,
      msgResult: '',
    }
  },
  created() {
    this.listFiles()
  },
  mounted() {
    window.addEventListener('beforeunload', this.handleBeforeUnload)  // 嘗試離開時觸發（重整或按叉）
    window.addEventListener('pagehide', this.handlePageHide)  // 在真的離開時觸發
  },
  beforeUnmount() {
    window.removeEventListener('beforeunload', this.handleBeforeUnload)
    window.removeEventListener('pagehide', this.handlePageHide)
  },
  computed: {
    rows() {
      const result = [];
      for (let i = 0; i < this.selected.input_values.length; i += 4) {
        result.push(this.selected.input_values.slice(i, i + 4));
      }
      return result;
    },

    modalButtons() {
      return {
        primary: {
          text: this.$t('lblDelete'),
          onClick: this.deleteMissingData,
        },
        secondary: {
          text: this.$t('lblCancel'),
          onClick: this.closeModalMissingData,
        }
      }
    }

  },
  watch: {
    async "selected.model_name"() {
      this.output = ''
      this.errors = {}
      this.selected.input_values = []
      await this.getFieldNumber()
    },

    "selected.data_name"() {
      if (this .selected.data_name.endsWith(".csv")) {
        this.watched.file_extension = ".csv"
      } else if (this.selected.data_name.endsWith(".xlsx")) {
        this.watched.file_extension = ".xlsx"
      } else {
        this.watched.file_extension = ""
      }

      this.initPreviewData()
      if (this.selected.data_name != '') {
        this.previewTab()
      }
    },
  },
  beforeRouteLeave(to, from, next) {
    if (this.loading) {
      const answer = window.confirm(this.$t('msgSysRunning'))
      if (answer) {
        if (this.controller) {
          this.controller.abort()
          this.isAborted = true
          navigator.sendBeacon(`${process.env.VUE_APP_API_URL}/cancel`)
        }
        next()
      } else {
        next(false)
      }
    } else {
      next()
    }
  },
  methods: {
    handleBeforeUnload(event) {
      // 僅提示，若確認離開則觸發 handlePageHide
      if (this.loading) {
        event.preventDefault()
        event.returnValue = '' // 必需，讓瀏覽器顯示警示對話框
      }
    },

    handlePageHide() {
      if (this.loading && this.controller) {
        this.controller.abort()
        this.isAborted = true
        navigator.sendBeacon(`${process.env.VUE_APP_API_URL}/cancel`)
      }
    },

    // list both model and tabular files
    async listFiles() {
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/list-files`, {
          folder_path: 'upload', // upload/
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
          folder_path: 'upload', // upload/
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

    initPreviewData() {
      this.preview_data = {
        columns: [],
        preview: [],
        total_rows: 0,
        total_columns: 0
      }
    },

    async getFieldNumber() {
      this.isAborted = false
      this.controller = new AbortController()
      this.selected.input_values = []
      if (this.selected.model_name) {
        this.loading = true
        try {
          const response = await axios.post(`${process.env.VUE_APP_API_URL}/get-field-number`,
            {
              model_path: `upload/${this.selected.model_name}`, // upload/
            },
            {
              signal: this.controller.signal
            }
          )
          if (response.data.status == "success" && !this.isAborted) {
            this.selected.input_values = Array(response.data.field_count).fill("");
          } else if (response.data.status == "error") {
            this.modal.title = this.$t('lblError')
            this.modal.content = response.data.message
            this.modal.icon = 'error'
            this.openModalNotification()
          }
        } catch (error) {
          if (axios.isCancel(error)) {
            console.warn("Prediction aborted")
            this.isAborted = true
            return
          }
          this.modal.title = this.$t('lblError')
          this.modal.content = error
          this.modal.icon = 'error'
          this.openModalNotification()
        } finally {
          this.loading = false
        }
      }
    },

    async previewTab() {
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/preview-tabular`, {
          file_path: `upload/${this.selected.data_name}`, // upload/
        })
        if (response.data.status == "success") {
          this.preview_data = response.data.preview_data
        } else if (response.data.status == "errorMissing") {
          this.modal.title = this.$t('lblError')
          this.modal.content = this.$t('msgMissingDataFound') + response.data.message + '\n' + this.$t('msgConfirmDeleteRows')
          this.modal.icon = 'error'
          this.openModalMissingData()
        } else if (response.data.status == "error") {
          this.modal.title = this.$t('lblError')
          this.modal.content = response.data.message
          this.modal.icon = 'error'
          this.initPreviewData()
          this.selected.data_name = ''
          this.openModalNotification()
        }
      } catch (error) {
        this.modal.title = this.$t('lblError')
        this.modal.content = error
        this.modal.icon = 'error'
        this.openModalNotification()
        this.initPreviewData()
        this.selected.data_name = ''
      }
      this.loading = false
    },

    toggleCollapse() {
      let collapseElement = this.$refs.collapsePreview
      let collapseInstance = Collapse.getInstance(collapseElement) || new Collapse(collapseElement)
      collapseInstance.toggle()
    },

    async deleteMissingData() {
      // 關閉 modal
      if (this.$refs.modalMissingDataRef) {
        this.$refs.modalMissingDataRef.closeModal()
      }
      this.loading = true

      // 從 message (Missing data: ['K3', 'O6']) 切割座標
      const match = this.modal.content.match(/\[(.*?)\]/)
      const missingCells = match[1]
        .split(',')
        .map(item => item.trim().replace(/'/g, ''))
      const rowsToDelete = []
      // 換算成 row index
      missingCells.forEach(cell => {
        const match = cell.match(/[A-Z]+(\d+)/)
        if (match) {
          const excelRow = parseInt(match[1])
          const dfIndex = excelRow - 2
          if (dfIndex >= 0) rowsToDelete.push(dfIndex)
        }
      })

      // delete-Tabular-Rows 成功才會執行 preview-Tabula
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/delete-tabular-rows`, {
          file_path: `upload/${this.selected.data_name}`, // upload/
          rows: rowsToDelete
        })
        if (response.data.status == "success") {
          await this.previewTab()
        } else if (response.data.status == "error") {
          this.modal.title = this.$t('lblError')
          this.modal.content = response.data.message
          this.modal.icon = 'error'
          this.openModalNotification()
          this.initPreviewData()
          this.selected.data_name = ''
        }
      } catch (error) {
        this.modal.title = this.$t('lblError')
        this.modal.content = error
        this.modal.icon = 'error'
        this.initPreviewData()
        this.selected.data_name = ''
        this.openModalNotification()
      }
      this.loading = false
    },

    closeModalMissingData() {
      this.initPreviewData()
      this.selected.data_name = ''
      if (this.$refs.modalMissingDataRef) {
        this.$refs.modalMissingDataRef.closeModal()
      }
    },

    validateForm() {
      this.errors = {}
      let isValid = true

      // Trained Model
      if (!this.selected.model_name) {
        this.errors.model_name = this.$t('msgValRequired')
        isValid = false
      }

      // Prediction Type
      if (this.selected.mode === "file") {  // File mode
        // File Selection
        if (!this.selected.data_name) {
          this.errors.data_name = this.$t('msgValRequired')
          isValid = false
        }
        // Results Saved as
        if (!this.selected.output_name) {
          this.errors.output_name = this.$t('msgValRequired')
          isValid = false
        }
        // Outcome Column
        if (!this.selected.label_column) {
          this.errors.label_column = this.$t('msgValRequired')
          isValid = false
        }
      } else if (this.selected.mode === "input") {  // Input mode
        this.errors.input_values = {}
        for (let i = 0; i < this.selected.input_values.length; i++) {
          let value = this.selected.input_values[i]
          if (!value || value.trim() === "") {
            this.errors.input_values[i] = this.$t('msgValRequired')
            isValid = false
          }
        }
      }

      return isValid
    },

    async runPredict() {
      if (!this.validateForm()) {
        return
      }

      this.isAborted = false
      this.controller = new AbortController()
      this.loading = true
      this.output = null

      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/run-predict`,
          {
            model_path: `upload/${this.selected.model_name}`, // upload/
            mode: this.selected.mode,
            data_path: `upload/${this.selected.data_name}`, // upload/
            output_name: this.selected.output_name,
            input_values: this.selected.input_values,
            label_column: this.selected.label_column,
          },
          {
            signal: this.controller.signal
          }
        )
        if (response.data.status == "success" && !this.isAborted) {
          this.output = response.data
          this.modal.title = this.$t('lblPredictionCompleted')
          this.modal.icon = 'success'
          if (this.selected.mode === 'file') {
            this.modal.content = this.$t('msgFileDownloaded')
            this.msgResult = this.$t('msgFileDownloaded')
            // download api
            // 結果檔案會暫時存在 data/result/ 裡面，懶得改了，牽動到 predict.py，好麻煩
            const path = `data/result/${this.selected.output_name}${this.watched.file_extension}`
            await this.downloadFile(path)
          } else if (this.selected.mode === 'input') {
            this.modal.content = this.$t('lblPredictionResult') + ':' + this.output.message[0]
            this.msgResult = this.$t('lblPredictionResult') + ':' + this.output.message[0]
          }
          this.openModalNotification()
        } else if (response.data.status == "error") {
          this.modal.title = this.$t('lblError')
          this.modal.content = response.data.message
          this.modal.icon = 'error'
          this.openModalNotification()
        }
      } catch (error) {
        if (axios.isCancel(error)) {
          console.warn("Prediction aborted")
          this.isAborted = true
          return
        }
        this.output = {
          status: 'error',
          message: error,
        }
        this.modal.title = this.$t('lblError')
        this.modal.icon = 'error'
        this.modal.content = error
        this.openModalNotification()
      } finally {
        this.loading = false
      }
    },

    async downloadFile(path) {
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/download`, {
          download_path: path
        }, {
          responseType: 'blob' // 關鍵：支援二進位檔案格式
        })

        const blob = response.data
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = path.split('/').pop()  // 取檔名
        a.click()
        URL.revokeObjectURL(url)
      } catch (err) {
        console.error('下載檔案失敗：', err)
      }
    },

    openModalNotification() {
      if (this.$refs.modalNotification) {
        this.$refs.modalNotification.openModal()
      } else {
        console.error("ModalNotification component not found.")
      }
    },

    openModalMissingData() {
      this.initPreviewData()
      if (this.$refs.modalMissingDataRef) {
        this.$refs.modalMissingDataRef.openModal()
      } else {
        console.error("ModalNotification component not found.")
      }
    },

    openModalShap(imageSrc, shap_importance) {
      if (this.$refs.modalShapRef) {
        this.modal.content = imageSrc
        this.modal.shap_importance = shap_importance
        this.$refs.modalShapRef.openModal()
      }
    },

    openModalLime(imageSrc, lime_example_0) {
      if (this.$refs.modalLimeRef) {
        this.modal.content = imageSrc
        this.modal.lime_example_0 = lime_example_0
        this.$refs.modalLimeRef.openModal()
      }
    },
  },
};
</script>
