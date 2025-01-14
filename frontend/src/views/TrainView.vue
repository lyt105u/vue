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
      <label for="inputEmail3" class="col-sm-3 col-form-label">Model type</label>
      <div class="col-sm-8">
        <select class="form-select" aria-label="Small select example" v-model="selected.model_type">
          <option v-for="(label, value) in modelOptions" :key="value" :value="value">
            {{ label }}
          </option>
        </select>
      </div>
    </div>

    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Form data</label>
      <div class="col-sm-8">
        <select class="form-select" aria-label="Small select example" v-model="selected.data">
          <option v-for="data in dataNames" :key="data" :value="data">{{ data }}</option>
        </select>
      </div>
    </div>

    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Label column</label>
      <div class="col-sm-8">
        <input v-model="selected.label_column" class="form-control" type="text">
      </div>
    </div>

    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Split train and test set</label>
      <div class="col-sm-4">
        <div class="form-floating">
          <select v-model="selected.train_size" class="form-select" id="floatingSelectGrid">
            <option v-for="size in trainSizeOptions" :key="size" :value="size">
              {{ size }}
            </option>
          </select>
          <label for="floatingSelectGrid">Training set</label>
        </div>
      </div>
      <div class="col-sm-4">
        <div class="form-floating">
          <select v-model="watched.test_size" class="form-select" id="floatingSelectGrid" disabled>
            <option :value="watched.test_size">{{ watched.test_size }}</option>
          </select>
          <label for="floatingSelectGrid">Testing set</label>
        </div>
      </div>
    </div>

    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Model saved as</label>
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
            <li>accuracy : {{ output.metrics.accuracy.toFixed(2) }}%</li>
            <li>recall : {{ output.metrics.recall.toFixed(2) }}%</li>
            <li>precision : {{ output.metrics.precision.toFixed(2) }}%</li>
            <li>f1_score : {{ output.metrics.f1_score.toFixed(2) }}%</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="col">
      <div class="card mb-4 rounded-3 shadow-sm">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">Recall > 80%</h4>
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
                      <td>{{ output.recall_80.true_positive }}</td>
                      <td>{{ output.recall_80.false_negative }}</td>
                    </tr>
                    <tr>
                      <td>{{ output.recall_80.false_positive }}</td>
                      <td>{{ output.recall_80.true_negative }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <li>Recall: {{ output.recall_80.recall.toFixed(2) }}%</li>
            <li>Specificity: {{ output.recall_80.specificity.toFixed(2) }}%</li>
            <li>Precision: {{ output.recall_80.precision.toFixed(2) }}%</li>
            <li>NPV: {{ output.recall_80.npv.toFixed(2) }}%</li>
            <li>F1 Score: {{ output.recall_80.f1_score.toFixed(2) }}%</li>
            <li>F2 Score: {{ output.recall_80.f2_score.toFixed(2) }}%</li>
            <li>Accuracy: {{ output.recall_80.accuracy.toFixed(2) }}%</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="col">
      <div class="card mb-4 rounded-3 shadow-sm">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">Recall > 85%</h4>
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
                      <td>{{ output.recall_85.true_positive }}</td>
                      <td>{{ output.recall_85.false_negative }}</td>
                    </tr>
                    <tr>
                      <td>{{ output.recall_85.false_positive }}</td>
                      <td>{{ output.recall_85.true_negative }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <li>Recall: {{ output.recall_85.recall.toFixed(2) }}%</li>
            <li>Specificity: {{ output.recall_85.specificity.toFixed(2) }}%</li>
            <li>Precision: {{ output.recall_85.precision.toFixed(2) }}%</li>
            <li>NPV: {{ output.recall_85.npv.toFixed(2) }}%</li>
            <li>F1 Score: {{ output.recall_85.f1_score.toFixed(2) }}%</li>
            <li>F2 Score: {{ output.recall_85.f2_score.toFixed(2) }}%</li>
            <li>Accuracy: {{ output.recall_85.accuracy.toFixed(2) }}%</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="col">
      <div class="card mb-4 rounded-3 shadow-sm">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">Recall > 90%</h4>
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
                      <td>{{ output.recall_90.true_positive }}</td>
                      <td>{{ output.recall_90.false_negative }}</td>
                    </tr>
                    <tr>
                      <td>{{ output.recall_90.false_positive }}</td>
                      <td>{{ output.recall_90.true_negative }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <li>Recall: {{ output.recall_90.recall.toFixed(2) }}%</li>
            <li>Specificity: {{ output.recall_90.specificity.toFixed(2) }}%</li>
            <li>Precision: {{ output.recall_90.precision.toFixed(2) }}%</li>
            <li>NPV: {{ output.recall_90.npv.toFixed(2) }}%</li>
            <li>F1 Score: {{ output.recall_90.f1_score.toFixed(2) }}%</li>
            <li>F2 Score: {{ output.recall_90.f2_score.toFixed(2) }}%</li>
            <li>Accuracy: {{ output.recall_90.accuracy.toFixed(2) }}%</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="col">
      <div class="card mb-4 rounded-3 shadow-sm">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">Recall > 95%</h4>
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
                      <td>{{ output.recall_95.true_positive }}</td>
                      <td>{{ output.recall_95.false_negative }}</td>
                    </tr>
                    <tr>
                      <td>{{ output.recall_95.false_positive }}</td>
                      <td>{{ output.recall_95.true_negative }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <li>Recall: {{ output.recall_95.recall.toFixed(2) }}%</li>
            <li>Specificity: {{ output.recall_95.specificity.toFixed(2) }}%</li>
            <li>Precision: {{ output.recall_95.precision.toFixed(2) }}%</li>
            <li>NPV: {{ output.recall_95.npv.toFixed(2) }}%</li>
            <li>F1 Score: {{ output.recall_95.f1_score.toFixed(2) }}%</li>
            <li>F2 Score: {{ output.recall_95.f2_score.toFixed(2) }}%</li>
            <li>Accuracy: {{ output.recall_95.accuracy.toFixed(2) }}%</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="col">
      <div class="card mb-4 rounded-3 shadow-sm">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">ROC curve</h4>
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
        lightgbm: "lightGBM",
        random_forest: "Random Forest",
        xgb: "XGB"
      },
      trainSizeOptions: [
        1.0,
        0.9,
        0.8,
        0.7,
      ],
      selected: {
        model_type: '',
        data: '',
        label_column: '',
        train_size: 0.8,
        model_name: '',
      },
      watched: {
        test_size: '',
        file_extension: '',
      },
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
    "selected.train_size"() {
      this.updateTestSize()
    },
    "selected.model_type"() {
      this.updateFileExtension()
    }
  },
  methods: {
    async fetchData() {
      try {
        const response = await axios.post('http://127.0.0.1:5000/fetch-data', {
          param: 'data'
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
      this.watched.test_size = (1 - parseFloat(this.selected.train_size)).toFixed(1)
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
        const response = await fetch('http://127.0.0.1:5000/run-train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ arg1: this.selected.model_type, arg2: this.selected.data }),
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        this.output = await response.json();

        this.imageData = `data:image/png;base64,${this.output.roc}`;

      } catch (error) {
        console.error('Error:', error);
        this.output = null;
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
