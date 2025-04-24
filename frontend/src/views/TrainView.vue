<template>
  <div class="about">
    <h1>Train</h1>
  </div>
  
  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <form class="row g-3 needs-validation" @submit.prevent="runTrain" style="margin-top: 16px">
    <!-- Model 種類 -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Model Type</label>
      <div class="col-sm-8">
        <select class="form-select" aria-label="Small select example" v-model="selected.model_type" :disabled="loading">
          <option v-for="(label, value) in modelOptions" :key="value" :value="value">
            {{ label }}
          </option>
        </select>
        <div v-if="errors.model_type" class="text-danger small">{{ errors.model_type }}</div>
      </div>
    </div>

    <!-- XGB 參數 -->
    <template v-if="selected.model_type=='xgb'">
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label"> Parameters </label>
        <div class="col-sm-2 form-floating">
          <!-- remove placeholder to cancel floating animation -->
          <input v-model="selected.xgb.n_estimators"
            type="text" 
            class="form-control" 
            id="floatingXgbEstimators" 
            :disabled="loading"
          />
          <label for="floatingXgbEstimators" style="margin-left:9px;"> n_estimators </label>
          <div v-if="errors.n_estimators" class="text-danger small">{{ errors.n_estimators }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.xgb.learning_rate"
            type="text" 
            class="form-control" 
            id="floatingXgbLearningRate" 
            :disabled="loading"
          />
          <label for="floatingXgbLearningRate" style="margin-left:9px;"> learning_rate </label>
          <div v-if="errors.learning_rate" class="text-danger small">{{ errors.learning_rate }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.xgb.max_depth"
            type="text" 
            class="form-control" 
            id="floatingXgbMaxDepth" 
            :disabled="loading"
          />
          <label for="floatingXgbMaxDepth" style="margin-left:9px;"> max_depth </label>
          <div v-if="errors.max_depth" class="text-danger small">{{ errors.max_depth }}</div>
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
            :disabled="loading"
          />
          <label for="floatingEstimators" style="margin-left:9px;"> n_estimators </label>
          <div v-if="errors.n_estimators" class="text-danger small">{{ errors.n_estimators }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.lgbm.learning_rate"
            type="text" 
            class="form-control" 
            id="floatingLearningRate" 
            :disabled="loading"
          />
          <label for="floatingLearningRate" style="margin-left:9px;"> learning_rate </label>
          <div v-if="errors.learning_rate" class="text-danger small">{{ errors.learning_rate }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.lgbm.max_depth"
            type="text" 
            class="form-control" 
            id="floatingMaxDepth" 
            :disabled="loading"
          />
          <label for="floatingMaxDepth" style="margin-left:9px;"> max_depth </label>
          <div v-if="errors.max_depth" class="text-danger small">{{ errors.max_depth }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.lgbm.num_leaves"
            type="text" 
            class="form-control" 
            id="floatingNumLeaves" 
            :disabled="loading"
          />
          <label for="floatingNumLeaves" style="margin-left:9px;"> num_leaves </label>
          <div v-if="errors.num_leaves" class="text-danger small">{{ errors.num_leaves }}</div>
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
            :disabled="loading"
          />
          <label for="floatingRfEstimators" style="margin-left:9px;"> n_estimators </label>
          <div v-if="errors.n_estimators" class="text-danger small">{{ errors.n_estimators }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.rf.max_depth"
            type="text" 
            class="form-control" 
            id="floatingRfMaxDepth" 
            :disabled="loading"
          />
          <label for="floatingRfMaxDepth" style="margin-left:9px;"> max_depth </label>
          <div v-if="errors.max_depth" class="text-danger small">{{ errors.max_depth }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.rf.random_state"
            type="text" 
            class="form-control" 
            id="floatingRfRandomState" 
            :disabled="loading"
          />
          <label for="floatingRfRandomState" style="margin-left:9px;"> random_state </label>
          <div v-if="errors.random_state" class="text-danger small">{{ errors.random_state }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.rf.n_jobs"
            type="text" 
            class="form-control" 
            id="floatingRfNJobs" 
            :disabled="loading"
          />
          <label for="floatingRfNJobs" style="margin-left:9px;"> n_jobs </label>
          <div v-if="errors.n_jobs" class="text-danger small">{{ errors.n_jobs }}</div>
        </div>
      </div>
    </template>

    <!-- Logistic Regression 參數 -->
    <template v-if="selected.model_type=='logistic_regression'">
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label"> Parameters </label>
        <div class="col-sm-2 form-floating">
          <select v-model="selected.lr.penalty" class="form-select" id="floatingLrPenalty" :disabled="loading">
            <option v-for="(label, value) in rfPenaltyOptions" :key="value" :value="value">
              {{ label }}
            </option>
          </select>
          <label for="floatingLrPenalty" style="margin-left:9px;"> penalty </label>
          <div v-if="errors.penalty" class="text-danger small">{{ errors.penalty }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.lr.C"
            type="text" 
            class="form-control" 
            id="floatingLrC" 
            :disabled="loading"
          />
          <label for="floatingLrC" style="margin-left:9px;"> C </label>
          <div v-if="errors.C" class="text-danger small">{{ errors.C }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <select v-model="selected.lr.solver" class="form-select" id="floatingLrSolver" :disabled="loading">
            <option v-for="(label, value) in rfSolverOptions" :key="value" :value="value">
              {{ label }}
            </option>
          </select>
          <label for="floatingLrSolver" style="margin-left:9px;"> solver </label>
          <div v-if="errors.solver" class="text-danger small">{{ errors.solver }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.lr.max_iter"
            type="text" 
            class="form-control" 
            id="floatingLrMaxIter" 
            :disabled="loading"
          />
          <label for="floatingLrMaxIter" style="margin-left:9px;"> max_iter </label>
          <div v-if="errors.max_iter" class="text-danger small">{{ errors.max_iter }}</div>
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
            :disabled="loading"
          />
          <label for="floatingTabnetBatchSize" style="margin-left:9px;"> batch_size </label>
          <div v-if="errors.batch_size" class="text-danger small">{{ errors.batch_size }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tabnet.max_epochs"
            type="text" 
            class="form-control" 
            id="floatingTabnetMaxEpochs" 
            :disabled="loading"
          />
          <label for="floatingTabnetMaxEpochs" style="margin-left:9px;"> max_epochs </label>
          <div v-if="errors.max_epochs" class="text-danger small">{{ errors.max_epochs }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.tabnet.patience"
            type="text" 
            class="form-control" 
            id="floatingTabnetPatience" 
            :disabled="loading"
          />
          <label for="floatingTabnetPatience" style="margin-left:9px;"> patience </label>
          <div v-if="errors.patience" class="text-danger small">{{ errors.patience }}</div>
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
            :disabled="loading"
          />
          <label for="floatingMlpHiddenLayer1" style="margin-left:9px;"> hidden_layer_1 </label>
          <div v-if="errors.hidden_layer_1" class="text-danger small">{{ errors.hidden_layer_1 }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.mlp.hidden_layer_2"
            type="text" 
            class="form-control" 
            id="floatingMlpHiddenLayer2"
            :disabled="loading"
          />
          <label for="floatingMlpHiddenLayer2" style="margin-left:9px;"> hidden_layer_2 </label>
          <div v-if="errors.hidden_layer_2" class="text-danger small">{{ errors.hidden_layer_2 }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.mlp.hidden_layer_3"
            type="text" 
            class="form-control" 
            id="floatingMlpHiddenLayer3"
            :disabled="loading"
          />
          <label for="floatingMlpHiddenLayer3" style="margin-left:9px;"> hidden_layer_3 </label>
          <div v-if="errors.hidden_layer_3" class="text-danger small">{{ errors.hidden_layer_3 }}</div>
        </div>
        <div class="col-sm-3 form-text"> Leave hidden layer 2 or 3 blank if not needed. </div>
      </div>

      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label"></label> <!-- 排版用 -->
        <div class="col-sm-2 form-floating">
          <select v-model="selected.mlp.activation" class="form-select" id="floatingMlpActivation" :disabled="loading">
            <option v-for="(label, value) in mlpActivactionOptions" :key="value" :value="value">
              {{ label }}
            </option>
          </select>
          <label for="floatingMlpActivation" style="margin-left:9px;"> activation </label>
          <div v-if="errors.activation" class="text-danger small">{{ errors.activation }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.mlp.learning_rate_init"
            type="text" 
            class="form-control" 
            id="floatingMlpLearningRateInit" 
            :disabled="loading"
          />
          <label for="floatingMlpLearningRateInit" style="margin-left:9px;"> learning_rate </label>
          <div v-if="errors.learning_rate_init" class="text-danger small">{{ errors.learning_rate_init }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.mlp.max_iter"
            type="text" 
            class="form-control" 
            id="floatingMlpMaxIter" 
            :disabled="loading"
          />
          <label for="floatingMlpMaxIter" style="margin-left:9px;"> max_iter </label>
          <div v-if="errors.max_iter" class="text-danger small">{{ errors.max_iter }}</div>
        </div>
      </div>
    </template>

    <!-- 訓練資料 -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">File Selection</label>
      <div class="col-sm-8">
        <input @change="handleFileChange" v-if="showInput" type="file" class="form-control" :disabled="loading">
        <div v-if="errors.data" class="text-danger small">{{ errors.data }}</div>
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

    <!-- Outcome 欄位 -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Outcome Column</label>
      <div class="col-sm-8">
        <select class="form-select" aria-label="Small select example" v-model="selected.label_column" :disabled="loading">
          <option v-for="column in preview_data.columns" :key="column" :value="column">
            {{ column }}
          </option>
        </select>
        <div v-if="errors.label_column" class="text-danger small">{{ errors.label_column }}</div>
      </div>
    </div>

    <!-- 切分訓練集 -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Data Split</label>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.split_strategy" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1" value="train_test_split" :disabled="loading">
          <label class="form-check-label" for="gridRadios1">
            Split into Train and Test
          </label>
        </div>
      </div>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.split_strategy" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1" value="k_fold" :disabled="loading">
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
        <input v-model="selected.split_value" type="range" class="form-range" min="0.5" max="0.9" step="0.1" :disabled="loading">
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
        <input v-model="selected.split_value" type="range" class="form-range" min="2" max="10" :disabled="loading">
      </div>
      <div class="col-sm-4 d-flex align-items-center">
        <span id="passwordHelpInline" class="form-text">
          cv_folds: <strong>{{ selected.split_value}}</strong>
        </span>
      </div>
    </div>

    <!-- Model 儲存檔名 -->
    <div v-if="selected.split_strategy=='train_test_split'" class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Model Saved as</label>
      <div class="col-sm-8">
        <div class="input-group">
          <input v-model="selected.model_name" class="form-control" type="text" :disabled="loading">
          <span class="input-group-text">{{ watched.file_extension }}</span>
        </div>
        <div v-if="errors.model_name" class="text-danger small">{{ errors.model_name }}</div>
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
      <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalImage('ROC Curve', imageRoc)" style="cursor: pointer;">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">ROC Curve</h4>
        </div>
        <img :src="imageRoc" alt="ROC Curve" />
      </div>
    </div>

    <!-- SHAP -->
    <div class="col">
      <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalShap(imageShap, output.shap_importance)" style="cursor: pointer;">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">SHAP</h4>
        </div>
        <img :src="imageShap" alt="SHAP" />
      </div>
    </div>

    <!-- LIME -->
    <div class="col">
      <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalLime(imageLime, output.lime_example_0)" style="cursor: pointer;">
        <div class="card-header py-3">
          <h4 class="my-0 fw-normal">LIME</h4>
        </div>
        <img :src="imageLime" alt="LIME" />
      </div>
    </div>
  </div>

  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"> <!-- question mark icon -->
  <ModalNotification ref="modalNotification" :title="modal.title" :content="modal.content" :icon="modal.icon" />
  <ModalNotification ref="modalFinishTrainingRef" :title="modal.title" :content="modal.content" :icon="modal.icon" :primaryButton="{ text: '下載' }" />
  <ModalFormulaExplain ref="formulaExplainModal" />
  <ModalImage ref="modalImageRef" :title="modal.title" :imageSrc="modal.content"/>
  <ModalShap ref="modalShapRef" :imageSrc="modal.content" :shapImportance="modal.shap_importance" :columns="preview_data.columns"/>
  <ModalLime ref="modalLimeRef" :imageSrc="modal.content" :lime_example_0="modal.lime_example_0" :columns="preview_data.columns"/>
</template>

<script>
import axios from 'axios';
import ModalNotification from "@/components/ModalNotification.vue"
import ModalFormulaExplain from "@/components/ModalFormulaExplain.vue"
import ModalImage from "@/components/ModalImage.vue"
import ModalShap from "@/components/ModalShap.vue"
import ModalLime from "@/components/ModalLime.vue"
import { Collapse } from 'bootstrap'

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
      modelOptions: {
        xgb: "XGB",
        lightgbm: "lightGBM",
        random_forest: "Random Forest",
        logistic_regression: "Logistic Regression",
        tabnet: "TabNet",
        mlp: "Multi-Layer Perceptron"
      },
      preview_data: {
        columns: [],
        preview: [],
        total_rows: 0,
        total_columns: 0
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
        'newton-cg': 'newton-cg',
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
        icon: 'info',
        shap_importance: {},
      },
      loading: false,
      imageRoc: null,
      imageShap: null,
      imageLime: null,
      errors: {}, // 檢核用
      showInput: true,  // 移除 input 的 UI 顯示用
    };
  },
  created() {
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
      // this.errors = {}
    },
    "selected.data"() {
      if (this.selected.data != '') {
        this.uploadTabular()
        this.selected.label_column = ''
      }
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

    handleFileChange(event) {
      // 得到的檔名和 this.selected.data 綁定，再用 watch 去呼叫檢查預覽 (ckheckPreviewTab.py) 的腳本
      const file = event.target.files[0]
      if (!file) {
        // 使用者取消選檔 → 什麼都不做或清除選擇
        this.selected.data = ''
        this.initPreviewData()
        return
      }
      if (!file.name.endsWith('.csv') && !file.name.endsWith('.xlsx')) {
        this.modal.title = "Error"
        this.modal.content = "Unsupported file format. Please provide a CSV or Excel file."
        this.modal.icon = "error"
        this.openModalNotification()
        this.selected.data = ''
        this.initPreviewData()
        // 移除 UI 顯示
        this.showInput = false
        requestAnimationFrame(() => {
          this.showInput = true
        })
      } else {
        this.selected.data = file.name
      }
    },

    toggleCollapse() {
      let collapseElement = this.$refs.collapsePreview
      let collapseInstance = Collapse.getInstance(collapseElement) || new Collapse(collapseElement)
      collapseInstance.toggle()
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

    async uploadTabular() {
      this.initPreviewData()
      this.loading = true
      try {
        const fileInput = document.querySelector('input[type="file"]')
        const file = fileInput.files[0]
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
          this.modal.content = response.data.message
          this.modal.icon = 'error'
          this.initPreviewData()
          this.selected.data = ''
          this.openModalNotification()
          this.selected.data = ''
          this.initPreviewData()
          // 移除 UI 顯示
          this.showInput = false
          requestAnimationFrame(() => {
            this.showInput = true
          })
        }
      } catch (error) {
        this.modal.title = 'Error'
        this.modal.content = error
        this.modal.icon = 'error'
        this.initPreviewData()
        this.selected.data = ''
        this.openModalNotification()
        this.selected.data = ''
        this.initPreviewData()
        // 移除 UI 顯示
        this.showInput = false
        requestAnimationFrame(() => {
          this.showInput = true
        })
      }
      this.loading = false
    },

    isInt(value) {
      return /^-?(0|[1-9][0-9]*)$/.test(value) // 允許可選的負號，後面跟著至少一個數字
    },
    isFloat(value) {
      return /^[0-9]+\.[0-9]+$/.test(value)
    },

    validateForm() {
      this.errors = {}
      let isValid = true

      // Model Type
      if (!this.selected.model_type) {
        this.errors.model_type = "Choose a model."
        isValid = false
      }

      // Parameters
      if (this.selected.model_type === "xgb") {
        if (!this.selected.xgb.n_estimators || !this.isInt(this.selected.xgb.n_estimators)) {
          this.errors.n_estimators = "Integer only."
          isValid = false
        }
        if (!this.selected.xgb.learning_rate || !this.isFloat(this.selected.xgb.learning_rate)) {
          this.errors.learning_rate = "Floating point only."
          isValid = false
        }
        if (!this.selected.xgb.max_depth || !this.isInt(this.selected.xgb.max_depth)) {
          this.errors.max_depth = "Integer only."
          isValid = false
        }
      } else if (this.selected.model_type === "lightgbm") {
        if (!this.selected.lgbm.n_estimators || !this.isInt(this.selected.lgbm.n_estimators)) {
          this.errors.n_estimators = "Integer only."
          isValid = false
        }
        if (!this.selected.lgbm.learning_rate || !this.isFloat(this.selected.lgbm.learning_rate)) {
          this.errors.learning_rate = "Floating point only."
          isValid = false
        }
        if (!this.selected.lgbm.max_depth || !this.isInt(this.selected.lgbm.max_depth)) {
          this.errors.max_depth = "Integer only."
          isValid = false
        }
        if (!this.selected.lgbm.num_leaves || !this.isInt(this.selected.lgbm.num_leaves)) {
          this.errors.num_leaves = "Integer only."
          isValid = false
        }
      } else if (this.selected.model_type === "random_forest") {
        if (!this.selected.rf.n_estimators || !this.isInt(this.selected.rf.n_estimators)) {
          this.errors.n_estimators = "Integer only."
          isValid = false
        }
        if (!this.selected.rf.max_depth || !this.isInt(this.selected.rf.max_depth)) {
          this.errors.max_depth = "Integer only."
          isValid = false
        }
        if (!this.selected.rf.random_state || !this.isInt(this.selected.rf.random_state)) {
          this.errors.random_state = "Integer only."
          isValid = false
        }
        if (!this.selected.rf.n_jobs || !this.isInt(this.selected.rf.n_jobs)) {
          this.errors.n_jobs = "Integer only."
          isValid = false
        }
      } else if (this.selected.model_type === "logistic_regression") {
        if (!this.selected.lr.penalty) {
          this.errors.penalty = "Choose a penalty."
          isValid = false
        }
        if (!this.selected.lr.C || !this.isFloat(this.selected.lr.C)) {
          this.errors.C = "Floating point only."
          isValid = false
        }
        if (!this.selected.lr.solver) {
          this.errors.solver = "Choose a solver."
          isValid = false
        }
        if (!this.selected.lr.max_iter || !this.isInt(this.selected.lr.max_iter)) {
          this.errors.max_iter = "Integer only."
          isValid = false
        }
      } else if (this.selected.model_type === "tabnet") {
        if (!this.selected.tabnet.batch_size || !this.isInt(this.selected.tabnet.batch_size)) {
          this.errors.batch_size = "Integer only."
          isValid = false
        }
        if (!this.selected.tabnet.max_epochs || !this.isInt(this.selected.tabnet.max_epochs)) {
          this.errors.max_epochs = "Integer only."
          isValid = false
        }
        if (!this.selected.tabnet.patience || !this.isInt(this.selected.tabnet.patience)) {
          this.errors.patience = "Integer only."
          isValid = false
        }
      } else if (this.selected.model_type === "mlp") {
        if (!this.selected.mlp.hidden_layer_1 || !this.isInt(this.selected.mlp.hidden_layer_1)) {
          this.errors.hidden_layer_1 = "Integer only."
          isValid = false
        }
        if (this.selected.mlp.hidden_layer_2 && !this.isInt(this.selected.mlp.hidden_layer_2)) {
          this.errors.hidden_layer_2 = "Integer only."
          isValid = false
        }
        if (this.selected.mlp.hidden_layer_3) {
          if (!this.selected.mlp.hidden_layer_2) {
            this.errors.hidden_layer_3 = "Require Layer 2."
            isValid = false
          }
          if (!this.isInt(this.selected.mlp.hidden_layer_3)) {
            this.errors.hidden_layer_3 = "Integer only."
            isValid = false
          }
        }
        if (!this.selected.mlp.activation) {
          this.errors.activation = "Choose an activation."
          isValid = false
        }
        if (!this.selected.mlp.learning_rate_init || !this.isFloat(this.selected.mlp.learning_rate_init)) {
          this.errors.learning_rate_init = "Floating point only."
          isValid = false
        }
        if (!this.selected.mlp.max_iter || !this.isInt(this.selected.mlp.max_iter)) {
          this.errors.max_iter = "Integer only."
          isValid = false
        }
      }

      // File Selection (data)
      if (!this.selected.data) {
        this.errors.data = "Choose a file."
        isValid = false
      }

      // Outcome Column (label_column)
      if (!this.selected.label_column) {
        this.errors.label_column = "Outcome column is required."
        isValid = false
      }

      // Model Saved as (model_name)
      if (!this.selected.model_name) {
        this.errors.model_name = "Model name is required."
        isValid = false
      }

      return isValid
    },
    
    async runTrain() {
      if (!this.validateForm()) {
        return
      }

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
        this.imageRoc = `data:image/png;base64,${this.output.roc}`
        this.imageShap = `data:image/png;base64,${this.output.shap_plot}`
        this.imageLime = `data:image/png;base64,${this.output.lime_plot}`

      } catch (error) {
        this.output = {
          "status": "error",
          "message": error,
        }
      } finally {
        if (this.output.status == 'success') {
          this.modal.title = 'Training Complete'
          this.modal.content = 'Model trained successfully!'
          this.modal.icon = 'success'

          // download api
          let extension = ".pkl"
          if (this.selected.model_type === "tabnet") extension = ".zip"
          else if (this.selected.model_type === "xgb") extension = ".json"
          const path = `model/${this.selected.model_name}${extension}`
          await this.downloadFile(path)
          this.loading = false
          this.openModalFinishTraining()

        } else if (this.output.status == 'error') {
          this.modal.title = 'Error'
          this.modal.content = this.output.message
          this.modal.icon = 'error'
          this.output = null
          this.loading = false
          this.openModalNotification()
        }
      }
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
        this.$refs.modalNotification.openModal();
      } else {
        console.error("ModalNotification component not found.");
      }
    },

    openModalFinishTraining() {
      if (this.$refs.modalFinishTrainingRef) {
        this.$refs.modalFinishTrainingRef.openModal();
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

    openModalImage(title, imageSrc) {
      if (this.$refs.modalImageRef) {
        this.modal.title = title
        this.modal.content = imageSrc
        this.$refs.modalImageRef.openModal();
      }
    },

    openModalShap(imageSrc, shap_importance) {
      if (this.$refs.modalShapRef) {
        this.modal.content = imageSrc
        this.modal.shap_importance = shap_importance
        this.$refs.modalShapRef.openModal();
      }
    },

    openModalLime(imageSrc, lime_example_0) {
      if (this.$refs.modalLimeRef) {
        this.modal.content = imageSrc
        this.modal.lime_example_0 = lime_example_0
        this.$refs.modalLimeRef.openModal();
      }
    },
  },
};
</script>
