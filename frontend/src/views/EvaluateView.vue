<template>
  <div class="about">
    <h1>{{ $t('lblModelEvaluation') }}</h1>
    <h6 class="text-body-secondary">{{ $t('msgEvaluateDescription') }}</h6>
  </div>

  <!-- hr -->
  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <form class="row g-3" @submit.prevent="runPredict" style="margin-top: 16px">
    <!-- Trained Model -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblModelFile') }}</label>
      <div class="col-sm-8">
        <select class="form-select" aria-label="Small select example" v-model="selected.model_name" :disabled="loading">
          <option v-for="model in modelOptions" :key="model" :value="model">
            {{ model }}
          </option>
        </select>
        <div v-if="errors.model_name" class="text-danger small">{{ errors.model_name }}</div>
      </div>
    </div>

    <!-- Tabular Data -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblTabularData') }}</label>
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
                  <th></th>
                  <th v-for="col in preview_data.columns" :key="col">
                    {{ col }}
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, rowIndex) in preview_data.preview" :key="rowIndex">
                  <td></td>
                  <td v-for="col in preview_data.columns" :key="col">
                    {{ row[col] }}
                  </td>
                </tr>
                <tr>
                  <td></td>
                  <td v-for="col in preview_data.columns" :key="col">
                    ...
                  </td>
                </tr>
              </tbody>
              <tfoot>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblMean') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'mean-' + col">
                    {{ summary[col] ? summary[col].mean.toFixed(2) : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblMedian') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'median-' + col">
                    {{ summary[col] ? summary[col].median : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblMin') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'min-' + col">
                    {{ summary[col] ? summary[col].min : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblMax') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'max-' + col">
                    {{ summary[col] ? summary[col].max : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblStd') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'std-' + col">
                    {{ summary[col] ? summary[col].std.toFixed(2) : '-' }}
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- True Label Column -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblOutcomeColumn') }}</label>
      <div class="col-sm-8">
        <select class="form-select" aria-label="Small select example" v-model="selected.label_column" :disabled="loading">
          <option v-for="column in preview_data.columns" :key="column" :value="column">
            {{ column }}
          </option>
        </select>
        <div v-if="errors.label_column" class="text-danger small">{{ errors.label_column }}</div>
      </div>
    </div>

    <!-- Pred Column -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblPredictionColumn') }}</label>
      <div class="col-sm-8">
        <input v-model="selected.pred_column" class="form-control" type="text" :disabled="loading">
        <div v-if="errors.pred_column" class="text-danger small">{{ errors.pred_column }}</div>
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
  <!-- <div v-if="output" class="about">
    <h3>
      {{ $t('lblPredictionResult') }}
    </h3>
    {{ output }}
  </div> -->

  <!-- Results 標題 -->
  <div v-if="output" class="about d-flex align-items-center gap-2" style="padding-bottom:12px;">
    <h3 class="mb-0 d-flex align-items-center">
      {{ $t('lblPredictionResult') }}
      <button style="border: none; background: none; cursor: pointer;" @click="openFormulaExplainModal"  :disabled="loading">
        <i class="fa fa-question-circle" style="font-size:24px;color:lightblue"></i>
      </button>
    </h3>
    <button v-if="!loading" @click="downloadReport" type="button" class="btn btn-outline-primary">
      <i class="fa fa-download me-1"></i>{{ $t('lblDownload') }}
    </button>
    <button v-if="loading" class="btn btn-outline-primary" type="button" disabled>
      <span class="spinner-border spinner-border-sm" aria-hidden="true"></span>
    </button>
  </div>
  
  <!-- 訓練結果 -->
  <div v-if="output" class="row row-cols-1 row-cols-md-3 mb-3 text-center">
    <div class="col">
      <div class="card mb-4 rounded-3 shadow-sm">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">{{ $t('lblTrainingResult') }}</h4>
        </div>
        <div class="card-body">
          <ul class="list-unstyled mt-3 mb-4">
            <div class="bd-example-snippet bd-code-snippet">
              <div class="bd-example m-0 border-0">
                <table class="table table-sm table-bordered">
                  <thead>
                    <tr>
                      <th scope="col" colspan="2">{{ $t('lblConfusionMatrix') }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>{{ output.confusion_matrix.true_positive }}</td>
                      <td>{{ output.confusion_matrix.false_negative }}</td>
                    </tr>
                    <tr>
                      <td>{{ output.confusion_matrix.false_positive }}</td>
                      <td>{{ output.confusion_matrix.true_negative }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <li>Accuracy : {{ output.metrics.accuracy.toFixed(2) }}%</li>
            <li>Recall : {{ output.metrics.recall.toFixed(2) }}%</li>
            <li>Precision : {{ output.metrics.precision.toFixed(2) }}%</li>
            <li>F1_score : {{ output.metrics.f1_score.toFixed(2) }}%</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- Recall 列表 -->
    <div class="col" v-for="recall in recallLevels" :key="recall.level">
      <div class="card mb-4 rounded-3 shadow-sm">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">Recall > {{ recall.level }}%</h4>
        </div>
        <div class="card-body">
          <ul class="list-unstyled mt-3 mb-4">
            <div class="bd-example-snippet bd-code-snippet">
              <div class="bd-example m-0 border-0">
                <table class="table table-sm table-bordered">
                  <thead>
                    <tr>
                      <th scope="col" colspan="2">{{ $t('lblConfusionMatrix') }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>{{ output[recall.key].true_positive }}</td>
                      <td>{{ output[recall.key].false_negative }}</td>
                    </tr>
                    <tr>
                      <td>{{ output[recall.key].false_positive }}</td>
                      <td>{{ output[recall.key].true_negative }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <li>Recall: {{ output[recall.key].recall.toFixed(2) }}%</li>
            <li>Specificity: {{ output[recall.key].specificity.toFixed(2) }}%</li>
            <li>Precision: {{ output[recall.key].precision.toFixed(2) }}%</li>
            <li>NPV: {{ output[recall.key].npv.toFixed(2) }}%</li>
            <li>F1 Score: {{ output[recall.key].f1_score.toFixed(2) }}%</li>
            <li>F2 Score: {{ output[recall.key].f2_score.toFixed(2) }}%</li>
            <li>Accuracy: {{ output[recall.key].accuracy.toFixed(2) }}%</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- ROC 曲線 -->
    <div class="col">
      <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalImage($t('lblRocCurve'), imageRoc)" style="cursor: pointer;">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">{{ $t('lblRocCurve') }}</h4>
        </div>
        <img :src="imageRoc" :alt="$t('lblRocCurve')" />
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

  <!-- hr -->
  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <!-- Note -->
  <div class="about text-body-secondary">
    <h6>{{ $t('lblNote') }}</h6>
    <ol class="h6">
      <li>{{ $t('msgEvaluateNote1') }}</li>
      <li>{{ $t('msgMissingDataNote') }}</li>
      <li>{{ $t('msgPredictNote3') }}</li>
      <li>{{ $t('msgPredictNote4') }}</li>
    </ol>
  </div>

  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
  <ModalNotification ref="modalNotification" :title="modal.title" :content="modal.content" :icon="modal.icon" />
  <ModalNotification ref="modalMissingDataRef" :title="modal.title" :content="modal.content" :icon="modal.icon" :primaryButton="modalButtons.primary" :secondaryButton="modalButtons.secondary" :onUserDismiss="closeModalMissingData" />
  <ModalFormulaExplain ref="formulaExplainModal" />
  <ModalImage ref="modalImageRef" :title="modal.title" :imageSrc="modal.content"/>
  <ModalShap ref="modalShapRef" :imageSrc="modal.content" :shapImportance="modal.shap_importance" :columns="preview_data.columns"/>
  <ModalLime ref="modalLimeRef" :imageSrc="modal.content" :lime_example_0="modal.lime_example_0" :columns="preview_data.columns"/>
</template>

<script>
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import axios from 'axios';
import ModalNotification from "@/components/ModalNotification.vue"
import { Collapse } from 'bootstrap'
import ModalFormulaExplain from "@/components/ModalFormulaExplain.vue"
import ModalImage from "@/components/ModalImage.vue"
import ModalShap from "@/components/ModalShap.vue"
import ModalLime from "@/components/ModalLime.vue"

export default {
  components: {
    ModalNotification,
    ModalFormulaExplain,
    ModalImage,
    ModalShap,
    ModalLime,
  },
  data() {
    return {
      selected: {
        model_name: '',
        data_name: '',
        output_name: '',
        label_column: '',
        pred_column: '',
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
      summary: {},
      modelOptions: [],
      fileOptions: [],
      recallLevels: [
        { level: 80, key: 'recall_80' },
        { level: 85, key: 'recall_85' },
        { level: 90, key: 'recall_90' },
        { level: 95, key: 'recall_95' }
      ],
      imageRoc: null,
      isUnmounted: false, // 防止跳轉後，API執行完仍繼續執行js，造成錯誤
    }
  },
  created() {
    this.listFiles()
  },
  mounted() {
    window.addEventListener('beforeunload', this.handleBeforeUnload)  // 嘗試離開時觸發（重整或按叉）
  },
  beforeUnmount() {
    window.removeEventListener('beforeunload', this.handleBeforeUnload)
    this.isUnmounted = true
  },
  computed: {
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
      // 僅提示
      if (this.loading) {
        event.preventDefault()
        event.returnValue = '' // 讓瀏覽器顯示警示對話框
      }
    },

    // list both model and tabular files
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

    initPreviewData() {
      this.preview_data = {
        columns: [],
        preview: [],
        total_rows: 0,
        total_columns: 0
      }
    },

    async previewTab() {
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/preview-tabular`, {
          file_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data_name}`, // upload/
        })
        if (response.data.status == "success") {
          this.preview_data = response.data.preview_data
          this.summary = response.data.summary
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
          file_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data_name}`, // upload/
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
      // Pred Column
      if (!this.selected.pred_column) {
        this.errors.pred_column = this.$t('msgValRequired')
      }

      return isValid
    },

    async runPredict() {
      if (!this.validateForm()) {
        return
      }
      this.loading = true
      this.output = null

      // 檢查 label 是否只有一種 class
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/check-label-uniqueness`, {
          file_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data_name}`, // upload/
          label_column: this.selected.label_column
        })
        if (response.data.status == "errorUnique") {
          this.modal.title = this.$t('lblError')
          this.modal.content = this.$t('msgLabelColumnClass')
          this.modal.icon = 'error'
          this.openModalNotification()
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

      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/run-evaluate`,
          {
            model_path: `upload/${sessionStorage.getItem('username')}/${this.selected.model_name}`, // upload/
            data_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data_name}`, // upload/
            output_name: this.selected.output_name,
            label_column: this.selected.label_column,
            pred_column: this.selected.pred_column,
            username: sessionStorage.getItem('username'),
          },
        )
        if (this.isUnmounted) return // 若頁面已離開就不要繼續處理
        if (response.data.status == "success") {
          this.output = response.data
          this.imageRoc = `data:image/png;base64,${this.output.roc}`
          this.modal.title = this.$t('lblPredictionCompleted')
          this.modal.icon = 'success'
          this.modal.content = this.$t('lblPredictionCompleted')
          this.openModalNotification()
        } else if (response.data.status == "error") {
          this.modal.title = this.$t('lblError')
          this.modal.content = response.data.message
          this.modal.icon = 'error'
          this.openModalNotification()
        }
      } catch (error) {
        if (this.isUnmounted) return // 頁面已離開就忽略錯誤處理
        this.output = {
          status: 'error',
          message: error,
        }
        this.modal.title = this.$t('lblError')
        this.modal.icon = 'error'
        this.modal.content = error
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

    openModalMissingData() {
      this.initPreviewData()
      if (this.$refs.modalMissingDataRef) {
        this.$refs.modalMissingDataRef.openModal()
      } else {
        console.error("ModalNotification component not found.")
      }
    },

    openFormulaExplainModal() {
      if (this.$refs.formulaExplainModal) {
        this.$refs.formulaExplainModal.openModal()
      } else {
        console.error("ModalFormulaExplain component not found.")
      }
    },

    openModalImage(title, imageSrc) {
      if (this.$refs.modalImageRef) {
        this.modal.title = title
        this.modal.content = imageSrc
        this.$refs.modalImageRef.openModal()
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

    async downloadReport() {
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/download-report`, {
          task_dir: this.output.task_dir,
        }, {
          responseType: 'blob'
        })
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        link.setAttribute('download', `${this.output.task_dir.split(/[\\/]/).pop()}.zip`)
        document.body.appendChild(link)
        link.click()
        link.remove()
      } catch (err) {
        this.modal.title = this.$t('lblError')
        this.modal.icon = 'error'
        this.modal.content = err
        this.openModalNotification()
      }
      this.loading = false
    },
  },
}
</script>
