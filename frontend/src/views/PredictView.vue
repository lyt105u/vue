<template>
  <div class="about">
    <h1>Predict</h1>
  </div>

  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <form class="row g-3" @submit.prevent="runPredict" style="margin-top: 16px">
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Trained Model</label>
      <div class="col-sm-8">
        <input @change="handleModelChange" v-if="showInputModel" type="file" class="form-control" :disabled="loading" ref="modelInput">
        <div v-if="errors.model_path" class="text-danger small">{{ errors.model_path }}</div>
      </div>
    </div>

    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Prediction Type</label>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.mode" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1" value="file"  :disabled="loading">
          <label class="form-check-label" for="gridRadios1">
            File Prediction
          </label>
        </div>
      </div>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.mode" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1" value="input"  :disabled="loading">
          <label class="form-check-label" for="gridRadios1">
            Manual Input
          </label>
        </div>
      </div>
    </div>

    <template v-if="selected.mode=='file'">
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label">File Selection</label>
        <div class="col-sm-8">
          <input @change="handleFileChange" v-if="showInputFile" type="file" class="form-control" :disabled="loading" ref="fileInput">
          <div v-if="errors.data_path" class="text-danger small">{{ errors.data_path }}</div>
        </div>
        <div class="col-sm-1">
          <button v-if="preview_data.columns != 0" class="btn btn-outline-primary" type="button" @click="toggleCollapse" :disabled="loading">Preview</button>
        </div>
      </div>

      <div v-if="preview_data.total_rows != 0" class="row mb-3">
        <div class="collapse" ref="collapsePreview">
          <div class="card card-body">
            <div class="table-responsive">
              <table class="table">
                <caption> Showing first 10 rows (total: {{ preview_data.total_rows }} rows) </caption>
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

      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label">Results Saved as</label>
        <div class="col-sm-8">
          <div class="input-group">
            <input v-model="selected.output_name" class="form-control" type="text" :disabled="loading">
            <span class="input-group-text">{{ watched.file_extension }}</span>
          </div>
          <div v-if="errors.output_name" class="text-danger small">{{ errors.output_name }}</div>
        </div>
      </div>
      
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label">Outcome Column</label>
        <div class="col-sm-8">
          <input v-model="selected.label_column" class="form-control" type="text" :disabled="loading">
          <div v-if="errors.label_column" class="text-danger small">{{ errors.label_column }}</div>
        </div>
      </div>
    </template>

    <template v-if="selected.mode=='input'">
      <div class="row mb-3" v-for="(row, rowIndex) in rows" :key="rowIndex">
        <!-- 第一行顯示，其他行保持空白，排版用 -->
        <label for="inputEmail3" class="col-sm-3 col-form-label">
          {{ rowIndex === 0 ? "Manual Input" : "" }}
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
    
    <div class="col-12">
      <button v-if="!loading" type="submit" class="btn btn-primary">Predict</button>
      <button v-if="loading" class="btn btn-primary" type="button" disabled>
        <span class="spinner-border spinner-border-sm" aria-hidden="true"></span>
      </button>
    </div>
  </form>

  <div v-if="output" class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <div v-if="output" class="about">
    <h3>
      Results
    </h3>
    {{ modal.content }}
  </div>

  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
  <ModalNotification ref="modalNotification" :title="modal.title" :content="modal.content" :icon="modal.icon" />
  <ModalNotification ref="modalMissingDataRef" :title="modal.title" :content="modal.content" :icon="modal.icon" :primaryButton="{ text: 'Delete', onClick: deleteMissingData }" :secondaryButton="{ text: 'Cancel', onClick: removeFileUI }" />
</template>

<script>
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import axios from 'axios';
import ModalNotification from "@/components/ModalNotification.vue"
import { Collapse } from 'bootstrap'

export default {
  components: {
    ModalNotification,
  },
  data() {
    return {
      modelNames: '',
      xlsxNames: '',
      selected: {
        model_path: '',
        mode: 'file',
        data_path: '',
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
        icon: '',
      },
      loading: false,
      errors: {}, // for validation
      preview_data: {
        columns: [],
        preview: [],
        total_rows: 0,
        total_columns: 0
      },
      showInputFile: true,  // 移除 input 的 UI 顯示用
      showInputModel: true,
    };
  },
  created() {},
  mounted() {},
  computed: {
    rows() {
      const result = [];
      for (let i = 0; i < this.selected.input_values.length; i += 4) {
        result.push(this.selected.input_values.slice(i, i + 4));
      }
      return result;
    },
  },
  watch: {
    "selected.model_path"() {
      this.selected.data_path = ''
      this.selected.output_name = ''
      this.selected.input_name = []
      this.output = ''
      this.getFieldNumber()
      this.errors = {}
      if (this.selected.model_path != '') {
        this.uploadModel()
      }
    },
    "selected.data_path"() {
      if (this .selected.data_path.endsWith(".csv")) {
        this.watched.file_extension = ".csv"
      } else if (this.selected.data_path.endsWith(".xlsx")) {
        this.watched.file_extension = ".xlsx"
      } else {
        this.watched.file_extension = ""
      }
      if (this.selected.data_path != '') {
        this.uploadTabular()
      }
    },
    "selected.mode"() {
      this.output = ''
      this.initPreviewData()
      this.selected.data_path = ''
    }
  },
  methods: {
    initPreviewData() {
      this.preview_data = {
        columns: [],
        preview: [],
        total_rows: 0,
        total_columns: 0
      }
    },

    async uploadModel() {
      this.loading = true
      try {
        const file = this.$refs.modelInput.files[0]
        const formData = new FormData()
        formData.append("file", file)
        const response = await axios.post('http://127.0.0.1:5000/upload-Model', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
       if (response.data.status == "error") {
          this.modal.title = 'Error'
          this.modal.content = response.data.message
          this.modal.icon = 'error'
          this.initPreviewData()
          this.selected.data_path = ''
          this.openModalNotification()
        }
      } catch (error) {
        this.modal.title = 'Error'
        this.modal.content = error
        this.modal.icon = 'error'
        this.selected.data_path = ''
        this.openModalNotification()
      }
      this.loading = false
    },

    async getFieldNumber() {
      this.loading = true
      this.selected.input_values = []
      if (this.selected.model_path) {
        try {
          const response = await axios.post('http://127.0.0.1:5000/get-fieldNumber', {
            param: this.selected.model_path
          });
          if (response.data.status == "success") {
            this.selected.input_values = Array(response.data.field_count).fill("");
          }
        } catch (error) {
          console.error("fetchData error: " + error)
          this.modelNames = { status: 'fail', error: '無法連接後端服務' };
        }
      }
      this.loading = false
    },

    handleFileChange(event) {
      // 得到的檔名和 this.selected.data_path 綁定，再用 watch 去呼叫檢查預覽 (ckheckPreviewTab.py) 的腳本
      const file = event.target.files[0]
      if (!file) {
        // 使用者取消選檔 → 什麼都不做或清除選擇
        this.selected.data_path = ''
        this.initPreviewData()
        return
      }
      if (!file.name.endsWith('.csv') && !file.name.endsWith('.xlsx')) {
        this.modal.title = "Error"
        this.modal.content = "Unsupported file format. Please provide a CSV or Excel file."
        this.modal.icon = "error"
        this.openModalNotification()
        this.selected.data_path = ''
        this.initPreviewData()
        // 移除 UI 顯示
        this.showInputFile = false
        requestAnimationFrame(() => {
          this.showInputFile = true
        })
      } else {
        this.selected.data_path = file.name
      }
    },

    toggleCollapse() {
      let collapseElement = this.$refs.collapsePreview
      let collapseInstance = Collapse.getInstance(collapseElement) || new Collapse(collapseElement)
      collapseInstance.toggle()
    },

    async uploadTabular() {
      this.initPreviewData()
      this.loading = true
      try {
        const file = this.$refs.fileInput.files[0]
        const formData = new FormData()
        formData.append("file", file)
        const response = await axios.post('http://127.0.0.1:5000/upload-Tabular', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        if (response.data.status == "success") {
          this.preview_data = response.data.preview_data
        } else if (response.data.status == "error") {
          this.modal.title = 'Error'
          this.modal.content = response.data.message + '\nDo you want to delete these rows?'
          this.modal.icon = 'error'
          this.openModalMissingData()
        }
      } catch (error) {
        this.modal.title = 'Error'
        this.modal.content = error
        this.modal.icon = 'error'
        this.initPreviewData()
        this.selected.data_path = ''
        this.openModalNotification()
        // 移除 UI 顯示
        this.showInputFile = false
        requestAnimationFrame(() => {
          this.showInputFile = true
        })
      }
      this.loading = false
    },

    handleModelChange(event) {
      // 得到的檔名和 this.selected.model_path 綁定
      const file = event.target.files[0]
      if (!file) {
        // 使用者取消選檔 → 什麼都不做或清除選擇
        this.selected.model_path = ''
        return
      }
      if (!file.name.endsWith('.zip') && !file.name.endsWith('.json') && !file.name.endsWith('.pkl')) {
        this.modal.title = "Error"
        this.modal.content = "Unsupported model format. Please provide a zip, json or pkl file."
        this.modal.icon = "error"
        this.openModalNotification()
        this.selected.model_path = ''
        // 移除 UI 顯示
        this.showInputModel = false
        requestAnimationFrame(() => {
          this.showInputModel = true
        })
      } else {
        this.selected.model_path = file.name
      }
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
        const response = await axios.post('http://127.0.0.1:5000/delete-Tabular-Rows', {
          filename: this.selected.data_path,
          rows: rowsToDelete
        })
        if (response.data.status == "success") {
          const response = await axios.post('http://127.0.0.1:5000/preview-Tabular', {
            filename: this.selected.data_path,
          })
          if (response.data.status == "success") {
            this.preview_data = response.data.preview_data
          } else if (response.data.status == "error") {
            this.modal.title = 'Error'
            this.modal.content = response.data.message
            this.modal.icon = 'error'
            this.initPreviewData()
            this.selected.data_path = ''
            this.openModalNotification()
            // 移除 UI 顯示
            this.showInputFile = false
            requestAnimationFrame(() => {
              this.showInputFile = true
            })
          }
        } else if (response.data.status == "error") {
          this.modal.title = 'Error'
          this.modal.content = response.data.message
          this.modal.icon = 'error'
          this.openModalNotification()
        }
      } catch (error) {
        this.modal.title = 'Error'
        this.modal.content = error
        this.modal.icon = 'error'
        this.initPreviewData()
        this.selected.data_path = ''
        this.openModalNotification()
        // 移除 UI 顯示
        this.showInputFile = false
        requestAnimationFrame(() => {
          this.showInputFile = true
        })
      }
      this.loading = false
    },

    removeFileUI() {
      this.initPreviewData()
      this.selected.data_path = ''
      console.log(this.selected.data_path)
      // 移除 UI 顯示
      this.showInputFile = false
      requestAnimationFrame(() => {
        this.showInputFile = true
      })
      if (this.$refs.modalMissingDataRef) {
        this.$refs.modalMissingDataRef.closeModal()
      }
    },

    validateForm() {
      this.errors = {}
      let isValid = true

      // Trained Model
      if (!this.selected.model_path) {
        this.errors.model_path = "Choose a model."
        isValid = false
      }

      // Prediction Type
      if (this.selected.mode === "file") {  // File mode
        // File Selection
        if (!this.selected.data_path) {
          this.errors.data_path = "Choose a file."
          isValid = false
        }
        // Results Saved as
        if (!this.selected.output_name) {
          this.errors.output_name = "Output name is required."
          isValid = false
        }
        // Outcome Column
        if (!this.selected.label_column) {
          this.errors.label_column = "Outcome column is required."
          isValid = false
        }
      } else if (this.selected.mode === "input") {  // Input mode
        this.errors.input_values = {}
        for (let i = 0; i < this.selected.input_values.length; i++) {
          let value = this.selected.input_values[i]
          if (!value || value.trim() === "") {
            this.errors.input_values[i] = `Field ${i + 1} is required.`
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

      this.loading = true
      this.output = null

      try {
        const response = await axios.post('http://127.0.0.1:5000/run-predict', this.selected)
        this.output = response.data
        this.modal.title = 'Training completed'
        this.modal.icon = 'success'
        if (this.selected.mode === 'file') {
          this.modal.content = 'Results file downloaded.'
          // download api
          const path = `data/result/${this.selected.output_name}${this.watched.file_extension}`
          await this.downloadFile(path)
        } else if (this.selected.mode === 'input') {
          this.modal.content = this.output.message[0]
        }
        this.openModalNotification()
      } catch (error) {
        console.error('Error:', error);
        this.output = {
          status: 'error',
          message: error.response?.data?.message || error.message,
        }
        this.modal.title = 'Error'
        this.modal.icon = 'error'
        this.modal.content = this.output.message
        this.openModalNotification()
      }

      this.loading = false
    },

    async downloadFile(path) {
      try {
        const response = await axios.post('http://127.0.0.1:5000/download', {
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
  },
};
</script>
