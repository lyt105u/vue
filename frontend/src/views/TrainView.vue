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
      <div class="col-sm-9">
        <select class="form-select" aria-label="Small select example" v-model="selected.model">
          <!-- <option value="catboost">Cat Boost</option> -->
          <option value="lightgbm">lightGBM</option>
          <option value="random_forest">Random Forest</option>
          <option value="xgb">XGB</option>
        </select>
      </div>
    </div>
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Data labeled 1</label>
      <div class="col-sm-9">
        <select class="form-select" aria-label="Small select example" v-model="selected.data1">
          <option v-for="data in xlsxNames" :key="data" :value="data">{{ data }}</option>
        </select>
      </div>
    </div>
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Data labeled 0</label>
      <div class="col-sm-9">
        <select class="form-select" aria-label="Small select example" v-model="selected.data0">
          <option v-for="data in xlsxNames" :key="data" :value="data">{{ data }}</option>
        </select>
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
      <button style="border: none; background: none; cursor: pointer;" data-bs-toggle="modal" data-bs-target="#exampleModal">
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

  <!-- finish training modal -->
  <div class="modal fade" id="finishTrainingModal" tabindex="-1" aria-labelledby="finishTrainingModalLabel" aria-hidden="true">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h1 class="modal-title fs-5" id="finishTrainingModalLabel">Modal title</h1>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div class="bd-example-snippet bd-code-snippet">
            <div class="bd-example m-0 border-0">
              Model trained successfully!
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
        </div>
      </div>
    </div>
  </div>

  <!-- question mark modal -->
  <div class="modal fade" id="exampleModal" tabindex="-1" aria-labelledby="exampleModalLabel" aria-hidden="true">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h1 class="modal-title fs-5" id="exampleModalLabel">Modal title</h1>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
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
                    <td>True Positive</td>
                    <td>False Negative</td>
                  </tr>
                  <tr>
                    <td>False Positive</td>
                    <td>True Negative</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          <div class="text-center">
            <div class="formula">
              <div>
                Recall =
                <span class="fraction">
                  <span class="numerator">
                    TP
                  </span>
                  <span class="denominator">
                    TP + FN
                  </span>
                </span>
              </div>
            </div>
          </div>
          <div class="text-center">
            <div class="formula">
              <div>
                Specificity =
                <span class="fraction">
                  <span class="numerator">
                    TN
                  </span>
                  <span class="denominator">
                    TN + FP
                  </span>
                </span>
              </div>
            </div>
          </div>
          <div class="text-center">
            <div class="formula">
              <div>
                Precision =
                <span class="fraction">
                  <span class="numerator">
                    TP
                  </span>
                  <span class="denominator">
                    TP + FP
                  </span>
                </span>
              </div>
            </div>
          </div>
          <div class="text-center">
            <div class="formula">
              <div>
                NPV =
                <span class="fraction">
                  <span class="numerator">
                    TN
                  </span>
                  <span class="denominator">
                    TN + FN
                  </span>
                </span>
              </div>
            </div>
          </div>
          <div class="text-center">
            <div class="formula">
              <div>
                F1-score =
                <span class="fraction">
                  <span class="numerator">
                    2 × Precision × Recall
                  </span>
                  <span class="denominator">
                    Precision + Recall
                  </span>
                </span>
              </div>
            </div>
          </div>
          <div class="text-center">
            <div class="formula">
              <div>
                F2-score =
                <span class="fraction">
                  <span class="numerator">
                    5 × Precision × Recall
                  </span>
                  <span class="denominator">
                    4 × Precision + Recall
                  </span>
                </span>
              </div>
            </div>
          </div>
          <div class="text-center">
            <div class="formula">
              <div>
                Accuracy =
                <span class="fraction">
                  <span class="numerator">
                    TP + TN
                  </span>
                  <span class="denominator">
                    TP + TN + FP + FN
                  </span>
                </span>
              </div>
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
        </div>
      </div>
    </div>
  </div>

  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"> <!-- question mark icon -->
</template>

<style scoped>
/* 公式用 */
.formula {
  display: inline-block;
  font-size: 1rem;
}

.fraction {
  display: inline-block;
  text-align: center;
  vertical-align: middle;
}

.numerator {
  display: block;
  border-bottom: 1px solid black;
  padding-bottom: 0.2rem;
}

.denominator {
  display: block;
  padding-top: 0.2rem;
}
</style>

<script>
import axios from 'axios';
import { Modal } from 'bootstrap';
export default {
  data() {
    return {
      xlsxNames: '',
      selected: {
        model: '',
        data1: '',
        data0: '',
      },
      output: '',
      loading: false,
      imageData: null,
    };
  },
  created() {
    this.fetchData();
  },
  mounted() {},
  computed: {},
  watch: {},
  methods: {
    async fetchData() {
      try {
        const response = await axios.post('http://127.0.0.1:5000/fetch-data', {
          param: 'data'
        });
        if (response.data.status == "success") {
          this.xlsxNames = response.data.files
        }
      } catch (error) {
        console.error("fetchData error: " + error)
        this.xlsxNames = { status: 'fail', error: '無法連接後端服務' };
      }
    },
    async runTrain() {
      this.loading = true
      this.output = null
      try {
        const response = await fetch('http://127.0.0.1:5000/run-train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ arg1: this.selected.model, arg2: this.selected.data1, arg3: this.selected.data0 }),
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
      this.openModal()
    },
    openModal() {
      const modalElement = document.getElementById('finishTrainingModal');
      if (modalElement) {
        const modalInstance = new Modal(modalElement);
        modalInstance.show();
      } else {
        console.error('Modal element not found');
      }
    },
  },
};
</script>
