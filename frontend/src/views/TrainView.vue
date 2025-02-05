<template>
  <div class="about">
    <h1>Train</h1>
  </div>
  
  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <form class="row g-3" @submit.prevent="runTrain" style="margin-top: 16px">
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Model Type</label>
      <div class="col-sm-8">
        <select class="form-select" aria-label="Small select example" v-model="selected.model_type">
          <option v-for="(label, value) in modelOptions" :key="value" :value="value">
            {{ label }}
          </option>
        </select>
      </div>
    </div>

    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">File Selection</label>
      <div class="col-sm-8">
        <select class="form-select" aria-label="Small select example" v-model="selected.data">
          <option v-for="data in dataNames" :key="data" :value="data">{{ data }}</option>
        </select>
      </div>
    </div>

    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Outcome Column</label>
      <div class="col-sm-8">
        <input v-model="selected.label_column" class="form-control" type="text">
      </div>
    </div>

    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Data Split</label>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.split_strategy" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1" value="train_test_split">
          <label class="form-check-label" for="gridRadios1">
            Split into Train and Test
          </label>
        </div>
      </div>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.split_strategy" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1" value="k_fold">
          <label class="form-check-label" for="gridRadios1">
            K-Fold Cross Validation
          </label>
        </div>
      </div>
    </div>

    <div v-if="selected.split_strategy=='train_test_split'" class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label"></label> <!-- 排版用 -->
      <div class="col-sm-4 d-flex align-items-center">
        <input v-model="selected.split_value" type="range" class="form-range" min="0.5" max="0.9" step="0.1">
      </div>
      <div class="col-sm-4 d-flex align-items-center">
        <span id="passwordHelpInline" class="form-text">
          Train: <strong>{{ selected.split_value}}</strong>, Test: {{ watched.test_size }}
        </span>
      </div>
    </div>
    <div v-if="selected.split_strategy=='k_fold'" class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label"></label> <!-- 排版用 -->
      <div class="col-sm-4 d-flex align-items-center">
        <input v-model="selected.split_value" type="range" class="form-range" min="2" max="10">
      </div>
      <div class="col-sm-4 d-flex align-items-center">
        <span id="passwordHelpInline" class="form-text">
          cv_folds: <strong>{{ selected.split_value}}</strong>
        </span>
      </div>
    </div>

    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Model Saved as</label>
      <div class="col-sm-8">
        <div class="input-group">
          <input v-model="selected.model_name" class="form-control" type="text">
          <span class="input-group-text">{{ watched.file_extension }}</span>
        </div>
      </div>
    </div>

    <div class="col-12">
      <button v-if="!loading" type="submit" class="btn btn-primary">Train</button>
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
      <button style="border: none; background: none; cursor: pointer;" @click="openFormulaExplainModal">
        <i class="fa fa-question-circle" style="font-size:24px;color:lightblue"></i>
      </button>
    </h3>
  </div>
  
  <div v-if="output" class="row row-cols-1 row-cols-md-3 mb-3 text-center">
    <div class="col">
      <div class="card mb-4 rounded-3 shadow-sm">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">Result</h4>
        </div>
        <div class="card-body">
          <ul class="list-unstyled mt-3 mb-4">
            <div class="bd-example-snippet bd-code-snippet">
              <div class="bd-example m-0 border-0">
                <table class="table table-sm table-bordered">
                  <thead>
                    <tr>
                      <th scope="col" colspan="2">Confusion Matrix</th>
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
    <template v-if="selected.split_strategy === 'train_test_split'">
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
                        <th scope="col" colspan="2">Confusion Matrix</th>
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
    </template>
    <div class="col">
      <div class="card mb-4 rounded-3 shadow-sm">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">ROC Curve</h4>
        </div>
        <img :src="imageData" alt="ROC Curve" />
      </div>
    </div>
  </div>

  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"> <!-- question mark icon -->
  <ModalNotification ref="modalNotification" title="Training Complete" content="Model trained successfully!" />
  <ModalFormulaExplain ref="formulaExplainModal" />
</template>

<script>
import axios from 'axios';
import ModalNotification from "@/components/ModalNotification.vue"
import ModalFormulaExplain from "@/components/ModalFormulaExplain.vue"

export default {
  components: {
    ModalNotification,
    ModalFormulaExplain,
  },
  data() {
    return {
      dataNames: '',
      modelOptions: {
        xgb: "XGB",
        lightgbm: "lightGBM",
        random_forest: "Random Forest",
        logistic_regression: "Logistic Regression",
      },
      trainSizeOptions: [
        '1.0',
        '0.9',
        '0.8',
        '0.7',
      ],
      selected: {
        model_type: '',
        data: '',
        label_column: '',
        split_strategy: 'train_test_split',
        split_value: '0.8',
        model_name: '',
      },
      watched: {
        test_size: '',
        file_extension: '',
      },
      recallLevels: [
        { level: 80, key: 'recall_80' },
        { level: 85, key: 'recall_85' },
        { level: 90, key: 'recall_90' },
        { level: 95, key: 'recall_95' }
      ],
      output: '',
      loading: false,
      imageData: null,
    };
  },
  created() {
    this.fetchData()
    this.updateTestSize()
    this.updateFileExtension()
  },
  mounted() {},
  computed: {},
  watch: {
    "selected.split_strategy"() {
      if (this.selected.split_strategy == 'train_test_split') {
        this.selected.split_value = '0.8'
        this.output = null
      } else if (this.selected.split_strategy == 'k_fold') {
        this.selected.split_value = '5'
        this.output = null
      }
    },
    "selected.split_value"() {
      if (this.selected.split_strategy == 'train_test_split') {
        this.updateTestSize()
      } else if (this.selected.split_strategy == 'k_fold') {
        this.watched.test_size = ''
      }
    },
    "selected.model_type"() {
      this.updateFileExtension()
    }
  },
  methods: {
    async fetchData() {
      try {
        const response = await axios.post('http://127.0.0.1:5000/fetch-data', {
          param: 'data/train'
        });
        if (response.data.status == "success") {
          this.dataNames = response.data.files
        }
      } catch (error) {
        console.error("fetchData error: " + error)
        this.dataNames = { status: 'fail', error: '無法連接後端服務' };
      }
    },
    updateTestSize() {
      this.watched.test_size = (1 - parseFloat(this.selected.split_value)).toFixed(1)
    },
    updateFileExtension() {
      if (this.selected.model_type == "xgb") {
        this.watched.file_extension = ".json"
      } else {
        this.watched.file_extension = ".pkl"
      }
    },
    async runTrain() {
      this.loading = true
      this.output = null
      try {
        const response = await axios.post('http://127.0.0.1:5000/run-train', {
          arg1: this.selected.model_type,
          arg2: this.selected.data,
          arg3: this.selected.label_column,
          arg4: this.selected.split_strategy,
          arg5: this.selected.split_value,
          arg6: this.selected.model_name,
        }, {
          headers: { 'Content-Type': 'application/json' }
        })
        
        this.output = response.data
        this.imageData = `data:image/png;base64,${this.output.roc}`
      } catch (error) {
        console.error('Error:', error)
        this.output = null
      }
      this.loading = false
      this.openModalNotification()
    },
    openModalNotification() {
      if (this.$refs.modalNotification) {
        this.$refs.modalNotification.openModal();
      } else {
        console.error("ModalNotification component not found.");
      }
    },
    openFormulaExplainModal() {
      if (this.$refs.formulaExplainModal) {
        this.$refs.formulaExplainModal.openModal();
      } else {
        console.error("ModalFormulaExplain component not found.");
      }
    },
  },
};
</script>
