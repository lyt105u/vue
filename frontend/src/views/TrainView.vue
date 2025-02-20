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
    <!-- Model 種類 -->
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

    <!-- XGB 參數 -->
    <template v-if="selected.model_type=='xgb'">
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label"> Parameters </label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.xgb.n_estimators"
            type="text" 
            class="form-control" 
            id="floatingXgbEstimators" 
            placeholder="floatingXgbEstimators" 
          />
          <label for="floatingXgbEstimators" style="margin-left:9px;"> n_estimators </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.xgb.learning_rate"
            type="text" 
            class="form-control" 
            id="floatingXgbLearningRate" 
            placeholder="floatingXgbLearningRate" 
          />
          <label for="floatingXgbLearningRate" style="margin-left:9px;"> learning_rate </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.xgb.max_depth"
            type="text" 
            class="form-control" 
            id="floatingXgbMaxDepth" 
            placeholder="floatingXgbMaxDepth" 
          />
          <label for="floatingXgbMaxDepth" style="margin-left:9px;"> max_depth </label>
        </div>
      </div>
    </template>

    <!-- LightGBM 參數 -->
    <template v-if="selected.model_type=='lightgbm'">
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label"> Parameters </label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.lgbm.n_estimators"
            type="text" 
            class="form-control" 
            id="floatingEstimators" 
            placeholder="floatingEstimators" 
          />
          <label for="floatingEstimators" style="margin-left:9px;"> n_estimators </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.lgbm.learning_rate"
            type="text" 
            class="form-control" 
            id="floatingLearningRate" 
            placeholder="floatingLearningRate" 
          />
          <label for="floatingLearningRate" style="margin-left:9px;"> learning_rate </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.lgbm.max_depth"
            type="text" 
            class="form-control" 
            id="floatingMaxDepth" 
            placeholder="floatingMaxDepth" 
          />
          <label for="floatingMaxDepth" style="margin-left:9px;"> max_depth </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.lgbm.num_leaves"
            type="text" 
            class="form-control" 
            id="floatingNumLeaves" 
            placeholder="floatingNumLeaves" 
          />
          <label for="floatingNumLeaves" style="margin-left:9px;"> num_leaves </label>
        </div>
      </div>
    </template>

    <!-- Random Forest 參數 -->
    <template v-if="selected.model_type=='random_forest'">
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label"> Parameters </label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.rf.n_estimators"
            type="text" 
            class="form-control" 
            id="floatingRfEstimators" 
            placeholder="floatingRfEstimators" 
          />
          <label for="floatingRfEstimators" style="margin-left:9px;"> n_estimators </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.rf.max_depth"
            type="text" 
            class="form-control" 
            id="floatingRfMaxDepth" 
            placeholder="floatingRfMaxDepth" 
          />
          <label for="floatingRfMaxDepth" style="margin-left:9px;"> max_depth </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.rf.random_state"
            type="text" 
            class="form-control" 
            id="floatingRfRandomState" 
            placeholder="floatingRfRandomState" 
          />
          <label for="floatingRfRandomState" style="margin-left:9px;"> random_state </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.rf.n_jobs"
            type="text" 
            class="form-control" 
            id="floatingRfNJobs" 
            placeholder="floatingRfNJobs" 
          />
          <label for="floatingRfNJobs" style="margin-left:9px;"> n_jobs </label>
        </div>
      </div>
    </template>

    <!-- Logistic Regression 參數 -->
    <template v-if="selected.model_type=='logistic_regression'">
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label"> Parameters </label>
        <div class="col-sm-2 form-floating">
          <select v-model="selected.lr.penalty" class="form-select" id="floatingLrPenalty">
            <option v-for="(label, value) in rfPenaltyOptions" :key="value" :value="value">
              {{ label }}
            </option>
          </select>
          <label for="floatingLrPenalty" style="margin-left:9px;"> penalty </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.lr.C"
            type="text" 
            class="form-control" 
            id="floatingLrC" 
            placeholder="floatingLrC" 
          />
          <label for="floatingLrC" style="margin-left:9px;"> C </label>
        </div>
        <div class="col-sm-2 form-floating">
          <select v-model="selected.lr.solver" class="form-select" id="floatingLrSolver">
            <option v-for="(label, value) in rfSolverOptions" :key="value" :value="value">
              {{ label }}
            </option>
          </select>
          <label for="floatingLrSolver" style="margin-left:9px;"> solver </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.lr.max_iter"
            type="text" 
            class="form-control" 
            id="floatingLrMaxIter" 
            placeholder="floatingLrMaxIter" 
          />
          <label for="floatingLrMaxIter" style="margin-left:9px;"> max_iter </label>
        </div>
      </div>
    </template>

    <!-- TabNet 參數 -->
    <template v-if="selected.model_type=='tabnet'">
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label"> Parameters </label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tabnet.batch_size"
            type="text" 
            class="form-control" 
            id="floatingTabnetBatchSize" 
            placeholder="floatingTabnetBatchSize" 
          />
          <label for="floatingTabnetBatchSize" style="margin-left:9px;"> batch_size </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tabnet.max_epochs"
            type="text" 
            class="form-control" 
            id="floatingTabnetMaxEpochs" 
            placeholder="floatingTabnetMaxEpochs" 
          />
          <label for="floatingTabnetMaxEpochs" style="margin-left:9px;"> max_epochs </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tabnet.patience"
            type="text" 
            class="form-control" 
            id="floatingTabnetPatience" 
            placeholder="floatingTabnetPatience" 
          />
          <label for="floatingTabnetPatience" style="margin-left:9px;"> patience </label>
        </div>
      </div>
    </template>

    <!-- MLP 參數 -->
    <template v-if="selected.model_type=='mlp'">
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label"> Parameters </label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.mlp.hidden_layer_1"
            type="text" 
            class="form-control" 
            id="floatingMlpHiddenLayer1" 
          />
          <label for="floatingMlpHiddenLayer1" style="margin-left:9px;"> hidden_layer_1 </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.mlp.hidden_layer_2"
            type="text" 
            class="form-control" 
            id="floatingMlpHiddenLayer2" 
          />
          <label for="floatingMlpHiddenLayer2" style="margin-left:9px;"> hidden_layer_2 </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.mlp.hidden_layer_3"
            type="text" 
            class="form-control" 
            id="floatingMlpHiddenLayer3" 
          />
          <label for="floatingMlpHiddenLayer3" style="margin-left:9px;"> hidden_layer_3 </label>
        </div>
        <div class="col-sm-3 form-text"> Leave hidden layer 2 or 3 blank if not needed. </div>
      </div>

      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label"></label> <!-- 排版用 -->
        <div class="col-sm-2 form-floating">
          <select v-model="selected.mlp.activation" class="form-select" id="floatingMlpActivation">
            <option v-for="(label, value) in mlpActivactionOptions" :key="value" :value="value">
              {{ label }}
            </option>
          </select>
          <label for="floatingMlpActivation" style="margin-left:9px;"> activation </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.mlp.learning_rate_init"
            type="text" 
            class="form-control" 
            id="floatingMlpLearningRateInit" 
            placeholder="floatingMlpLearningRateInit" 
          />
          <label for="floatingMlpLearningRateInit" style="margin-left:9px;"> learning_rate </label>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.mlp.max_iter"
            type="text" 
            class="form-control" 
            id="floatingMlpMaxIter" 
            placeholder="floatingMlpMaxIter" 
          />
          <label for="floatingMlpMaxIter" style="margin-left:9px;"> max_iter </label>
        </div>
      </div>
    </template>

    <!-- 訓練資料 -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">File Selection</label>
      <div class="col-sm-8">
        <select class="form-select" aria-label="Small select example" v-model="selected.data">
          <option v-for="data in dataNames" :key="data" :value="data">{{ data }}</option>
        </select>
      </div>
    </div>

    <!-- Outcome 欄位 -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Outcome Column</label>
      <div class="col-sm-8">
        <input v-model="selected.label_column" class="form-control" type="text">
      </div>
    </div>

    <!-- 切分訓練集 -->
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

    <!-- Range 拉桿 -->
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

    <!-- Model 儲存檔名 -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Model Saved as</label>
      <div class="col-sm-8">
        <div class="input-group">
          <input v-model="selected.model_name" class="form-control" type="text">
          <span class="input-group-text">{{ watched.file_extension }}</span>
        </div>
      </div>
    </div>

    <!-- button -->
    <div class="col-12">
      <button v-if="!loading" type="submit" class="btn btn-primary">Train</button>
      <button v-if="loading" class="btn btn-primary" type="button" disabled>
        <span class="spinner-border spinner-border-sm" aria-hidden="true"></span>
      </button>
    </div>
  </form>

  <!-- 橫線 -->
  <div v-if="output" class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <!-- Results 標題 -->
  <div v-if="output" class="about">
    <h3>
      Results
      <button style="border: none; background: none; cursor: pointer;" @click="openFormulaExplainModal">
        <i class="fa fa-question-circle" style="font-size:24px;color:lightblue"></i>
      </button>
    </h3>
  </div>
  
  <!-- 訓練結果 -->
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

    <!-- Recall 列表 -->
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

    <!-- ROC 曲線 -->
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
  <ModalNotification ref="modalNotification" :title="modal.title" :content="modal.content" />
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
        tabnet: "TabNet",
        mlp: "Multi-Layer Perceptron"
      },
      rfPenaltyOptions: {
        l1: 'l1',
        l2: 'l2',
        elasticnet: 'elasticnet',
        none: 'none'
      },
      rfSolverOptions: {
        lbfgs: 'lbfgs',
        liblinear: 'liblinear',
        newton_cg: 'newton-cg',
        sag: 'sag',
        saga: 'saga'
      },
      mlpActivactionOptions: {
        relu: 'relu',
        tanh: 'tanh',
        logistic: 'logistic'
      },
      selected: {
        model_type: '',
        data: '',
        label_column: '',
        split_strategy: 'train_test_split',
        split_value: '0.8',
        model_name: '',
        xgb: {
          n_estimators: '100',
          learning_rate: '0.300000012',
          max_depth: '6'
        },
        lgbm: {
          n_estimators: '100',
          learning_rate: '0.1',
          max_depth: '-1',
          num_leaves: '31'
        },
        rf: {
          n_estimators: '900',
          max_depth: '50',
          random_state: '0',
          n_jobs: '-1'
        },
        lr: {
          penalty: 'l2',    // L2 正歸化
          C: '1.0',         // 正歸化強度
          solver: 'lbfgs',
          max_iter: '500', 
        },
        tabnet: {
          batch_size: '256',
          max_epochs: '2',
          patience: '10',
        },
        mlp: {
          hidden_layer_1: '128',
          hidden_layer_2: '64',
          hidden_layer_3: '',
          activation: 'relu',
          learning_rate_init: '0.001',
          max_iter: '300',
        }
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
      modal: {
        title: '',
        content: '',
      },
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
        this.dataNames = { status: 'error', error: '無法連接後端服務' };
      }
    },

    updateTestSize() {
      this.watched.test_size = (1 - parseFloat(this.selected.split_value)).toFixed(1)
    },

    updateFileExtension() {
      if (this.selected.model_type == "xgb") {
        this.watched.file_extension = ".json"
      } else if (this.selected.model_type == "tabnet") {
        this.watched.file_extension = ".zip"
      } else {
        this.watched.file_extension = ".pkl"
      }
    },
    
    async runTrain() {
      try {
        this.loading = true
        this.output = null
        let api = ''
        let payload = {
          file_name: this.selected.data,
          label_column: this.selected.label_column,
          split_strategy: this.selected.split_strategy,
          split_value: this.selected.split_value,
          model_name: this.selected.model_name,
        }

        if (this.selected.model_type == "xgb") {
          api = "run-train-xgb"
          payload["n_estimators"] = this.selected.xgb.n_estimators
          payload["learning_rate"] = this.selected.xgb.learning_rate
          payload["max_depth"] = this.selected.xgb.max_depth
        } else if (this.selected.model_type == "lightgbm") {
          api = "run-train-lgbm"
          payload["n_estimators"] = this.selected.lgbm.n_estimators
          payload["learning_rate"] = this.selected.lgbm.learning_rate
          payload["max_depth"] = this.selected.lgbm.max_depth
          payload["num_leaves"] = this.selected.lgbm.num_leaves
        } else if (this.selected.model_type == "random_forest") {
          api = "run-train-rf"
          payload["n_estimators"] = this.selected.rf.n_estimators
          payload["max_depth"] = this.selected.rf.max_depth
          payload["random_state"] = this.selected.rf.random_state
          payload["n_jobs"] = this.selected.rf.n_jobs
        } else if (this.selected.model_type == "logistic_regression") {
          api = "run-train-lr"
          payload["penalty"] = this.selected.lr.penalty
          payload["C"] = this.selected.lr.C
          payload["solver"] = this.selected.lr.solver
          payload["max_iter"] = this.selected.lr.max_iter
        } else if (this.selected.model_type == "tabnet") {
          api = "run-train-tabnet"
          payload["batch_size"] = this.selected.tabnet.batch_size
          payload["max_epochs"] = this.selected.tabnet.max_epochs
          payload["patience"] = this.selected.tabnet.patience
        } else if (this.selected.model_type == "mlp") {
          api = "run-train-mlp"
          payload["hidden_layer_1"] = this.selected.mlp.hidden_layer_1
          payload["hidden_layer_2"] = this.selected.mlp.hidden_layer_2
          payload["hidden_layer_3"] = this.selected.mlp.hidden_layer_3
          payload["activation"] = this.selected.mlp.activation
          payload["learning_rate_init"] = this.selected.mlp.learning_rate_init
          payload["max_iter"] = this.selected.mlp.max_iter
        } else {
          this.output = {
            "status": "error",
            "message": "Unsupported model type"
          }
          return
        }

        console.log(payload)

        const response = await axios.post(`http://127.0.0.1:5000/${api}`, payload)
        this.output = response.data
        this.imageData = `data:image/png;base64,${this.output.roc}`

      } catch (error) {
        this.output = {
          "status": "error",
          "message": error,
        }
      } finally {
        if (this.output.status == 'success') {
          this.modal.title = 'Training Complete'
          this.modal.content = 'Model trained successfully!'
        } else if (this.output.status == 'error') {
          this.modal.title = 'Error'
          this.modal.content = this.output.message
          this.output = null
        }
        this.loading = false
        this.openModalNotification()
      }
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
