<template>
  <div class="about">
    <h1>{{ $t('lblTrain') }}</h1>
    <h6 class="text-body-secondary">{{ $t('msgTrainDescription') }}</h6>
  </div>
  
  <!-- 橫線 -->
  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <form class="row g-3 needs-validation" @submit.prevent="runTrain" style="margin-top: 16px">
    <!-- Model 種類 -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblModelType') }}</label>
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
        <label for="inputEmail3" class="col-sm-3 col-form-label"> {{ $t('lblParameter') }} </label>
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
        <label for="inputEmail3" class="col-sm-3 col-form-label"> {{ $t('lblParameter') }} </label>
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
        <label for="inputEmail3" class="col-sm-3 col-form-label"> {{ $t('lblParameter') }} </label>
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
        <label for="inputEmail3" class="col-sm-3 col-form-label"> {{ $t('lblParameter') }} </label>
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
        <label for="inputEmail3" class="col-sm-3 col-form-label"> {{ $t('lblParameter') }} </label>
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
        <label for="inputEmail3" class="col-sm-3 col-form-label"> {{ $t('lblParameter') }} </label>
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
        <div class="col-sm-3 form-text"> {{ $t('msgHiddenLayerHint') }} </div>
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
        <div class="col-sm-2 form-floating">
          <input v-model="selected.mlp.n_iter_no_change"
            type="text" 
            class="form-control" 
            id="floatingMlpMaxIter" 
            :disabled="loading"
          />
          <label for="floatingMlpMaxIter" style="margin-left:9px;"> n_iter_no_change </label>
          <div v-if="errors.n_iter_no_change" class="text-danger small">{{ errors.n_iter_no_change }}</div>
        </div>
      </div>
    </template>

    <!-- CatBoost 參數 -->
    <template v-if="selected.model_type=='catboost'">
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label"> {{ $t('lblParameter') }} </label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.catboost.iterations"
            type="text" 
            class="form-control" 
            id="floatingCatBoostIterations" 
            :disabled="loading"
          />
          <label for="floatingCatBoostIterations" style="margin-left:9px;"> iterations </label>
          <div v-if="errors.iterations" class="text-danger small">{{ errors.iterations }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.catboost.learning_rate"
            type="text" 
            class="form-control" 
            id="floatingCatBoostLearningRate" 
            :disabled="loading"
          />
          <label for="floatingCatBoostLearningRate" style="margin-left:9px;"> learning_rate </label>
          <div v-if="errors.learning_rate" class="text-danger small">{{ errors.learning_rate }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.catboost.depth"
            type="text" 
            class="form-control" 
            id="floatingCatBoostDepth" 
            :disabled="loading"
          />
          <label for="floatingCatBoostDepth" style="margin-left:9px;"> depth </label>
          <div v-if="errors.depth" class="text-danger small">{{ errors.depth }}</div>
        </div>
      </div>
    </template>

    <!-- AdaBoost 參數 -->
    <template v-if="selected.model_type=='adaboost'">
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label"> {{ $t('lblParameter') }} </label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.adaboost.n_estimators"
            type="text" 
            class="form-control" 
            id="floatingAdaBoostEstimators" 
            :disabled="loading"
          />
          <label for="floatingAdaBoostEstimators" style="margin-left:9px;"> n_estimators </label>
          <div v-if="errors.n_estimators" class="text-danger small">{{ errors.n_estimators }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.adaboost.learning_rate"
            type="text" 
            class="form-control" 
            id="floatingAdaBoostLearningRate" 
            :disabled="loading"
          />
          <label for="floatingAdaBoostLearningRate" style="margin-left:9px;"> learning_rate </label>
          <div v-if="errors.learning_rate" class="text-danger small">{{ errors.learning_rate }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.adaboost.depth"
            type="text" 
            class="form-control" 
            id="floatingAdaBoostDepth" 
            :disabled="loading"
          />
          <label for="floatingAdaBoostDepth" style="margin-left:9px;"> depth </label>
          <div v-if="errors.depth" class="text-danger small">{{ errors.depth }}</div>
        </div>
      </div>
    </template>

    <!-- SVM 參數 -->
    <template v-if="selected.model_type=='svm'">
      <div class="row mb-3">
        <label class="col-sm-3 col-form-label"> {{ $t('lblParameter') }} </label>
        <div class="col-sm-2 form-floating">
          <input v-model="selected.svm.C"
            type="text" 
            class="form-control" 
            id="floatingSvmC" 
            :disabled="loading"
          />
          <label for="floatingSvmC" style="margin-left:9px;"> C </label>
          <div v-if="errors.C" class="text-danger small">{{ errors.C }}</div>
        </div>
        <div class="col-sm-2 form-floating">
          <select v-model="selected.svm.kernel" class="form-select" id="floatingSvmKernel" :disabled="loading">
            <option v-for="(label, value) in svmKernelOptions" :key="value" :value="value">
              {{ label }}
            </option>
          </select>
          <label for="floatingSvmKernel" style="margin-left:9px;"> kernel </label>
          <div v-if="errors.kernel" class="text-danger small">{{ errors.kernel }}</div>
        </div>
      </div>
    </template>

    <!-- 表格式資料 -->
    <div class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblTabularData') }}</label>
      <div class="col-sm-8">
        <select class="form-select" aria-label="Small select example" v-model="selected.data" :disabled="loading">
          <option v-for="file in fileOptions" :key="file" :value="file">
            {{ file }}
          </option>
        </select>
        <div v-if="errors.data" class="text-danger small">{{ errors.data }}</div>
      </div>
      <div class="col-sm-1">
        <button v-if="preview_data.columns != 0" class="btn btn-outline-primary" style="white-space: nowrap" type="button" @click="toggleCollapse" :disabled="loading">{{ $t('lblPreview') }}</button>
      </div>
    </div>

    <!-- Preview -->
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
                  <td style="white-space: nowrap"><strong>{{ $t('lblMedian') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'median-' + col">
                    {{ summary[col] ? summary[col].median : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblMean') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'mean-' + col">
                    {{ summary[col] ? summary[col].mean.toFixed(2) : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblMode') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'mode-' + col">
                    {{ summary[col] ? summary[col].mode : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblStd') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'std-' + col">
                    {{ summary[col] ? summary[col].std.toFixed(2) : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblMaxZscore') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'zscore_max-' + col">
                    {{ summary[col] ? summary[col].zscore_max.toFixed(2) : '-' }}
                  </td>
                </tr>
                <tr>
                  <td style="white-space: nowrap"><strong>{{ $t('lblMinZscore') }}</strong></td>
                  <td v-for="col in preview_data.columns" :key="'zscore_min-' + col">
                    {{ summary[col] ? summary[col].zscore_min.toFixed(2) : '-' }}
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- 缺失值處理 -->
    <!-- <template v-if="missing_cords && missing_cords.length > 0">
      <div class="row mb-3" v-for="(row, rowIndex) in rows" :key="rowIndex">
        <label class="col-sm-3 col-form-label">
          {{ rowIndex === 0 ? $t('lblMissingValueHandling') : "" }}
        </label>
        <div
          v-for="(header, colIndex) in row"
          :key="`${rowIndex}-${colIndex}`"
          class="col-sm-2"
        >
          <div class="form-floating">
            <select
              class="form-select"
              v-model="missing_methods[header]"
              :id="`select-${header}`"
              :disabled="loading"
            >
              <option v-for="option in missing_options" :key="option.value" :value="option.value">{{ $t(option.label) }}</option>
            </select>
            <label :for="`select-${header}`">
              {{ header }}
            </label>
          </div>
          <div v-if="errors_preprocess[header]" class="text-danger small"> {{ errors_preprocess[header] }} </div>
        </div>
        <div v-if="rowIndex === 0" class="col-sm-1 d-flex align-items-center">
          <button
            class="btn btn-outline-primary"
            style="white-space: nowrap"
            type="button"
            @click="preprocess"
            :disabled="loading"
          >
            {{ $t('lblPreprocess') }}
          </button>
        </div>
      </div>
    </template> -->

    <!-- 資料前處理 -->
    <div v-if="preview_data.total_rows != 0" class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblDataPreprocessing') }}</label>
      <div class="col-sm-8">
        <DataPreprocessing
          ref="preprocessor"
          :columns="preview_data.columns"
          :missing-columns="missing_header"
          :loading="loading"
          @update:rules="handleRuleUpdate"
          @update:unhandled="onUnhandledMissingUpdate"
          :previewTab = "previewTab"
          :data = "selected.data"
        />
      </div>
    </div>

    <!-- True Label Column 欄位 -->
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

    <!-- 切分訓練集 -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblDataSplit') }}</label>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.split_strategy" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1" value="train_test_split" :disabled="loading">
          <label class="form-check-label" for="gridRadios1">
            {{ $t('lblSplitTrainTest') }}
          </label>
        </div>
      </div>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.split_strategy" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1" value="k_fold" :disabled="loading">
          <label class="form-check-label" for="gridRadios1">
            {{ $t('lblKfoldValid') }}
          </label>
        </div>
      </div>
    </div>

    <!-- Range 拉桿 -->
    <div v-if="selected.split_strategy=='train_test_split'" class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label"></label>
      <div class="col-sm-4 d-flex align-items-center">
        <input v-model="selected.split_value" type="range" class="form-range" min="0.5" max="0.9" step="0.1" :disabled="loading">
      </div>
      <div class="col-sm-4 d-flex align-items-center">
        <span id="passwordHelpInline" class="form-text">
          {{ $t('lblTrain') }}: <strong>{{ selected.split_value}}</strong>, {{ $t('lblTest') }}: {{ watched.test_size }}
        </span>
      </div>
    </div>
    <div v-if="selected.split_strategy=='k_fold'" class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label"></label>
      <div class="col-sm-4 d-flex align-items-center">
        <input v-model="selected.split_value" type="range" class="form-range" min="2" max="10" :disabled="loading">
      </div>
      <div class="col-sm-4 d-flex align-items-center">
        <span id="passwordHelpInline" class="form-text">
          cv_folds: <strong>{{ selected.split_value}}</strong>
        </span>
      </div>
    </div>

    <!-- 只留 train_test_split -->
    <!-- <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblTrainTestSplit') }}</label>
      <div class="col-sm-4 d-flex align-items-center">
        <input v-model="selected.split_value" type="range" class="form-range" min="0.5" max="0.9" step="0.1" :disabled="loading">
      </div>
      <div class="col-sm-4 d-flex align-items-center">
        <span id="passwordHelpInline" class="form-text">
          {{ $t('lblTrain') }}: <strong>{{ selected.split_value}}</strong>, {{ $t('lblTest') }}: {{ watched.test_size }}
        </span>
      </div>
    </div> -->

    <!-- Model 儲存檔名 -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblModelSavedAs') }}</label>
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
      <button v-if="!loading" type="submit" class="btn btn-primary">{{ $t('lblTrain') }}</button>
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
  <div v-if="output" class="about d-flex align-items-center gap-2" style="padding-bottom:12px;">
    <h3 class="mb-0 d-flex align-items-center">
      {{ $t('lblTrainingResult') }}
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
    <!-- train_test_split -->
    <template v-if="output.split_strategy === 'train_test_split'">
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

      <!-- Loss 曲線 -->
      <div v-if="output.loss_plot" class="col">
        <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalImage('Loss', imageLoss)" style="cursor: pointer;">
          <div class="card-header py-3">
            <h4 class="my-0 fw-normal">Loss</h4>
          </div>
          <img :src="imageLoss" alt="Loss" />
        </div>
      </div>

      <!-- Accuracy 曲線 -->
      <div v-if="output.accuracy_plot" class="col">
        <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalImage('Accuracy', imageAccuracy)" style="cursor: pointer;">
          <div class="card-header py-3">
            <h4 class="my-0 fw-normal">Accuracy</h4>
          </div>
          <img :src="imageAccuracy" alt="Accuracy" />
        </div>
      </div>

      <!-- SHAP -->
      <div v-if="!output.shap_error" class="col">
        <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalShap(imageShap, output.shap_importance)" style="cursor: pointer;">
          <div class="card-header py-3">
            <h4 class="my-0 fw-normal">{{ $t('lblShap') }}</h4>
          </div>
          <img :src="imageShap" :alt="$t('lblShap')" />
        </div>
      </div>

      <!-- LIME -->
      <div v-if="!output.lime_error" class="col">
        <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalLime(imageLime, output.lime_example_0)" style="cursor: pointer;">
          <div class="card-header py-3">
            <h4 class="my-0 fw-normal">{{ $t('lblLime') }}</h4>
          </div>
          <img :src="imageLime" :alt="$t('lblLime')" />
        </div>
      </div>
    </template>

    <template v-if="output.split_strategy === 'k_fold'">
      <!-- 每個 fold -->
      <div v-for="fold in output.folds" :key="fold.fold">
        <div class="col">
          <div class="card mb-4 rounded-3 shadow-sm">
            <div class="card-header py-3">
              <h4 class="my-0 fw-normal">Fold {{ fold.fold }}</h4>
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
                          <td>{{ fold.confusion_matrix.true_positive }}</td>
                          <td>{{ fold.confusion_matrix.false_negative }}</td>
                        </tr>
                        <tr>
                          <td>{{ fold.confusion_matrix.false_positive }}</td>
                          <td>{{ fold.confusion_matrix.true_negative }}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
                <li>Accuracy : {{ fold.metrics.accuracy.toFixed(2) }}%</li>
                <li>Recall : {{ fold.metrics.recall.toFixed(2) }}%</li>
                <li>Precision : {{ fold.metrics.precision.toFixed(2) }}%</li>
                <li>F1_score : {{ fold.metrics.f1_score.toFixed(2) }}%</li>
                <li>AUC : {{ fold.metrics.auc.toFixed(2) }}%</li>
              </ul>
            </div>
          </div>
        </div>

        <!-- ROC -->
        <div class="col">
          <div class="card mb-4 rounded-3 shadow-sm" @click="openModalImage('ROC', `data:image/png;base64,${fold.roc}`)" style="cursor: pointer;">
            <div class="card-header py-3">
              <h4 class="my-0 fw-normal">ROC</h4>
            </div>
            <img :src="`data:image/png;base64,${fold.roc}`" alt="ROC" />
          </div>
        </div>

        <!-- Loss -->
        <div class="col" v-if="fold.loss_plot">
          <div class="card mb-4 rounded-3 shadow-sm" @click="openModalImage('Loss', `data:image/png;base64,${fold.loss_plot}`)" style="cursor: pointer;">
            <div class="card-header py-3">
              <h4 class="my-0 fw-normal">Loss</h4>
            </div>
            <img :src="`data:image/png;base64,${fold.loss_plot}`" alt="Loss" />
          </div>
        </div>

        <!-- Accuracy -->
        <div class="col" v-if="fold.accuracy_plot">
          <div class="card mb-4 rounded-3 shadow-sm" @click="openModalImage('Accuracy', `data:image/png;base64,${fold.accuracy_plot}`)" style="cursor: pointer;">
            <div class="card-header py-3">
              <h4 class="my-0 fw-normal">Accuracy</h4>
            </div>
            <img :src="`data:image/png;base64,${fold.accuracy_plot}`" alt="Accuracy" />
          </div>
        </div>

        <!-- SHAP -->
        <div v-if="fold.shap_plot" class="col">
          <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalShap(`data:image/png;base64,${fold.shap_plot}`, fold.shap_importance)" style="cursor: pointer;">
            <div class="card-header py-3">
              <h4 class="my-0 fw-normal">{{ $t('lblShap') }}</h4>
            </div>
            <img :src="`data:image/png;base64,${fold.shap_plot}`" :alt="$t('lblShap')" />
          </div>
        </div>

        <!-- LIME -->
        <div v-if="fold.lime_plot" class="col">
          <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalLime(`data:image/png;base64,${fold.lime_plot}`, fold.lime_example_0)" style="cursor: pointer;">
            <div class="card-header py-3">
              <h4 class="my-0 fw-normal">{{ $t('lblLime') }}</h4>
            </div>
            <img :src="`data:image/png;base64,${fold.lime_plot}`" :alt="$t('lblLime')" />
          </div>
        </div>
      </div>

      <!-- 平均值結果 -->
      <div class="col">
        <div class="card mb-4 rounded-3 shadow-sm">
          <div class="card-header py-3">
            <h4 class="my-0 fw-normal">{{ $t('lblAverage') }}</h4>
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
                        <td>{{ output.average.confusion_matrix.true_positive }}</td>
                        <td>{{ output.average.confusion_matrix.false_negative }}</td>
                      </tr>
                      <tr>
                        <td>{{ output.average.confusion_matrix.false_positive }}</td>
                        <td>{{ output.average.confusion_matrix.true_negative }}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
              <li>Accuracy : {{ output.average.accuracy.toFixed(2) }}%</li>
              <li>Recall : {{ output.average.recall.toFixed(2) }}%</li>
              <li>Precision : {{ output.average.precision.toFixed(2) }}%</li>
              <li>F1_score : {{ output.average.f1_score.toFixed(2) }}%</li>
              <li>AUC : {{ output.average.auc.toFixed(2) }}%</li>
            </ul>
          </div>
        </div>
      </div>
    </template>
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
      <li>{{ $t('lblTabularData') }}</li>
        <ol type="i">
          <li>{{ $t('msgLabelColumnClass') }}</li>
          <li>{{ $t('msgMissingDataNote') }}</li>
        </ol>
      <li>{{ $t('lblXgb') }}
        <ol type="i">
          <li><code>n_estimators</code>{{ $t('msgTrainNoteXgb1') }}</li>
          <li><code>learning_rate</code>{{ $t('msgTrainNoteXgb2') }}</li>
          <li><code>max_depth</code>{{ $t('msgTrainNoteXgb3') }}</li>
        </ol>
      </li>
      <li>{{ $t('lblLightGBM') }}
        <ol type="i">
          <li><code>n_estimators</code>{{ $t('msgTrainNoteLgbm1') }}</li>
          <li><code>learning_rate</code>{{ $t('msgTrainNoteLgbm2') }}</li>
          <li><code>max_depth</code>{{ $t('msgTrainNoteLgbm3') }}</li>
          <li><code>num_leaves</code>{{ $t('msgTrainNoteLgbm4') }}</li>
        </ol>
      </li>
      <li>{{ $t('lblRandomForest') }}
        <ol type="i">
          <li>{{ $t('msgTrainNoteRf1') }}</li>
          <li><u>{{ $t('msgTrainLonger') }}</u></li>
          <li><code>n_estimators</code>{{ $t('msgTrainNoteRf2') }}</li>
          <li><code>max_depth</code>{{ $t('msgTrainNoteRf3') }}</li>
          <li><code>random_state</code>{{ $t('msgTrainNoteRf4') }}</li>
          <li><code>n_jobs</code>{{ $t('msgTrainNoteRf5') }}</li>
        </ol>
      </li>
      <li>{{ $t('lblLogisticRegression') }}
        <ol type="i">
          <li><code>penalty</code>{{ $t('msgTrainNoteRf2') }}</li>
          <li><code>C</code>{{ $t('msgTrainNoteRf3') }}</li>
          <li><code>max_iter</code>{{ $t('msgTrainNoteRf4') }}</li>
        </ol>
      </li>
      <li>{{ $t('lblTabNet') }}
        <ol type="i">
          <li>{{ $t('msgTrainNoteTabnet1') }}</li>
          <li><code>batch_size</code>{{ $t('msgTrainNoteTabnet2') }}</li>
          <li><code>max_epochs</code>{{ $t('msgTrainNoteTabnet3') }}</li>
          <li><code>patience</code>{{ $t('msgTrainNoteTabnet4') }}</li>
        </ol>
      </li>
      <li>{{ $t('lblMultiLayerPerceptron') }}
        <ol type="i">
          <li>{{ $t('msgTrainNoteMlp8') }}</li>
          <li>{{ $t('msgTrainNoteMlp9') }}</li>
          <li>{{ $t('msgTrainNoteMlp10') }}</li>
          <li><code>hidden_layer_1</code>{{ $t('msgTrainNoteMlp1') }}</li>
          <li><code>hidden_layer_2</code>{{ $t('msgTrainNoteMlp2') }}</li>
          <li><code>hidden_layer_3</code>{{ $t('msgTrainNoteMlp3') }}</li>
          <li><code>activation</code>{{ $t('msgTrainNoteMlp4') }}</li>
          <li><code>learning_rate_init</code>{{ $t('msgTrainNoteMlp5') }}</li>
          <li><code>max_iter</code>{{ $t('msgTrainNoteMlp6') }}</li>
          <li><code>n_iter_no_change</code>{{ $t('msgTrainNoteMlp7') }}</li>
        </ol>
      </li>
    </ol>
  </div>

  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"> <!-- question mark icon -->
  <ModalNotification ref="modalNotification" :title="modal.title" :content="modal.content" :icon="modal.icon" />
  <ModalNotification ref="modalMissingDataRef" :title="modal.title" :content="modal.content" :icon="modal.icon" :primaryButton="modalMissingData.primary" :secondaryButton="modalMissingData.secondary" :onUserDismiss="closeModalMissingData"/>
  <ModalFormulaExplain ref="formulaExplainModal" />
  <ModalImage ref="modalImageRef" :title="modal.title" :imageSrc="modal.content"/>
  <ModalShap ref="modalShapRef" :imageSrc="modal.content" :shapImportance="modal.shap_importance" :columns="preview_data.columns"/>
  <ModalLime ref="modalLimeRef" :imageSrc="modal.content" :lime_example_0="modal.lime_example_0" :columns="preview_data.columns"/>
</template>

<script>
import axios from 'axios'
import ModalNotification from "@/components/ModalNotification.vue"
import ModalFormulaExplain from "@/components/ModalFormulaExplain.vue"
import ModalImage from "@/components/ModalImage.vue"
import ModalShap from "@/components/ModalShap.vue"
import ModalLime from "@/components/ModalLime.vue"
import { Collapse } from 'bootstrap'
import { toRaw } from 'vue'
import DataPreprocessing from '@/components/DataPreprocessing.vue'

export default {
  components: {
    ModalNotification,
    ModalFormulaExplain,
    ModalImage,
    ModalShap,
    ModalLime,
    DataPreprocessing,
  },
  data() {
    return {
      modelOptions: {
        xgb: this.$t('lblXgb'),
        lightgbm: this.$t('lblLightGBM'),
        random_forest: this.$t('lblRandomForest'),
        logistic_regression: this.$t('lblLogisticRegression'),
        tabnet: this.$t('lblTabNet'),
        mlp: this.$t('lblMultiLayerPerceptron'),
        catboost: this.$t('lblCatBoost'),
        adaboost: this.$t('lblAdaBoost'),
        svm: this.$t('lblSvm'),
      },
      preview_data: {
        columns: [],
        preview: [],
        total_rows: 0,
        total_columns: 0
      },
      summary: {},
      missing_cords: [],
      missing_header:[],
      missing_methods: [],
      missing_options: [
        { value: 'min', label: 'lblMin' },
        { value: 'max', label: 'lblMax' },
        { value: 'median', label: 'lblMedian' },
        { value: 'mean', label: 'lblMean' },
        { value: 'mode', label: 'lblMode' },
        { value: 'skip', label: 'lblFillSkip' },
        { value: 'zero', label: 'lblZero'},
      ],
      rfPenaltyOptions: {
        l1: 'l1',
        l2: 'l2',
        elasticnet: 'elasticnet',
        none: 'none'
      },
      mlpActivactionOptions: {
        relu: 'relu',
        tanh: 'tanh',
        logistic: 'logistic'
      },
      svmKernelOptions:{
        linear: 'linear',
        poly: 'poly',
        rbf: 'rbf',
        sigmoid: 'sigmoid',
        precomputed: 'precomputed'
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
          n_iter_no_change: '50'
        },
        catboost: {
          iterations: '500',
          learning_rate: '0.009',
          depth: '6',
        },
        adaboost: {
          n_estimators: '100',
          learning_rate: '1.0',
          depth: '3',
        },
        svm: {
          C: '1.0',
          kernel: 'rbf'
        },
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
      imageLoss: null,
      imageAccuracy: null,
      imageShap: null,
      imageLime: null,
      errors: {}, // 檢核用
      errors_preprocess: {},
      fileOptions: [],
      isUnmounted: false, // 防止跳轉後，API執行完仍繼續執行js，造成錯誤
      rules: [], // 預處理用
      unhandledMissingColumns: []
    }
  },
  created() {
    this.updateTestSize()
    this.updateFileExtension()
    this.listFiles()
  },
  mounted() {
    window.addEventListener('beforeunload', this.handleBeforeUnload)
  },
  beforeUnmount() {
    window.removeEventListener('beforeunload', this.handleBeforeUnload)
    this.isUnmounted = true
  },
  computed: {
    modalMissingData() {
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
    },
    rows() {
      const result = [];
      for (let i = 0; i < this.missing_header.length; i += 4) {
        result.push(this.missing_header.slice(i, i + 4));
      }
      return result;
    },
  },
  watch: {
    "selected.split_strategy"() {
      if (this.selected.split_strategy == 'train_test_split') {
        this.selected.split_value = '0.8'
      } else if (this.selected.split_strategy == 'k_fold') {
        this.selected.split_value = '5'
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
      this.selected.label_column = ''
      if (this.selected.data != '') {
        this.previewTab(true)
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

    initPreviewData() {
      this.preview_data = {
        columns: [],
        preview: [],
        total_rows: 0,
        total_columns: 0
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

    async listFiles() {
      this.loading = true
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

    async deleteMissingData() {
      // 關閉 modal
      if (this.$refs.modalMissingDataRef) {
        this.$refs.modalMissingDataRef.closeModal()
      }
      this.loading = true

      // 從 message (['K3', 'O6']) 切割座標
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

      // delete-Tabular-Rows 成功才會執行 preview-tabular
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/delete-tabular-rows`, {
          file_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data}`, // upload/
          rows: rowsToDelete
        })
        if (response.data.status == "success") {
          await this.previewTab(true)
        } else if (response.data.status == "error") {
          this.modal.title = this.$t('lblError')
          this.modal.content = response.data.message
          this.modal.icon = 'error'
          this.openModalNotification()
          this.initPreviewData()
          this.selected.data = ''
        }
      } catch (error) {
        this.modal.title = this.$t('lblError')
        this.modal.content = error
        this.modal.icon = 'error'
        this.openModalNotification()
        this.initPreviewData()
        this.selected.data = ''
      }
      this.loading = false
    },

    closeModalMissingData() {
      this.initPreviewData()
      this.selected.data = ''
      if (this.$refs.modalMissingDataRef) {
        this.$refs.modalMissingDataRef.closeModal()
      }
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
        this.errors.model_type = this.$t('msgValRequired')
        isValid = false
      }

      // Parameters
      if (this.selected.model_type === "xgb") {
        if (!this.selected.xgb.n_estimators || !this.isInt(this.selected.xgb.n_estimators)) {
          this.errors.n_estimators = this.$t('msgValIntOnly')
          isValid = false
        }
        if (!this.selected.xgb.learning_rate || !this.isFloat(this.selected.xgb.learning_rate)) {
          this.errors.learning_rate = this.$t('msgValFloatOnly')
          isValid = false
        }
        if (!this.selected.xgb.max_depth || !this.isInt(this.selected.xgb.max_depth)) {
          this.errors.max_depth = this.$t('msgValIntOnly')
          isValid = false
        }
      } else if (this.selected.model_type === "lightgbm") {
        if (!this.selected.lgbm.n_estimators || !this.isInt(this.selected.lgbm.n_estimators)) {
          this.errors.n_estimators = this.$t('msgValIntOnly')
          isValid = false
        }
        if (!this.selected.lgbm.learning_rate || !this.isFloat(this.selected.lgbm.learning_rate)) {
          this.errors.learning_rate = this.$t('msgValFloatOnly')
          isValid = false
        }
        if (!this.selected.lgbm.max_depth || !this.isInt(this.selected.lgbm.max_depth)) {
          this.errors.max_depth = this.$t('msgValIntOnly')
          isValid = false
        }
        if (!this.selected.lgbm.num_leaves || !this.isInt(this.selected.lgbm.num_leaves)) {
          this.errors.num_leaves = this.$t('msgValIntOnly')
          isValid = false
        }
      } else if (this.selected.model_type === "random_forest") {
        if (!this.selected.rf.n_estimators || !this.isInt(this.selected.rf.n_estimators)) {
          this.errors.n_estimators = this.$t('msgValIntOnly')
          isValid = false
        }
        if (!this.selected.rf.max_depth || !this.isInt(this.selected.rf.max_depth)) {
          this.errors.max_depth = this.$t('msgValIntOnly')
          isValid = false
        }
        if (!this.selected.rf.random_state || !this.isInt(this.selected.rf.random_state)) {
          this.errors.random_state = this.$t('msgValIntOnly')
          isValid = false
        }
        if (!this.selected.rf.n_jobs || !this.isInt(this.selected.rf.n_jobs)) {
          this.errors.n_jobs = this.$t('msgValIntOnly')
          isValid = false
        }
      } else if (this.selected.model_type === "logistic_regression") {
        if (!this.selected.lr.penalty) {
          this.errors.penalty = this.$t('msgValRequired')
          isValid = false
        }
        if (!this.selected.lr.C || !this.isFloat(this.selected.lr.C)) {
          this.errors.C = this.$t('msgValFloatOnly')
          isValid = false
        }
        if (!this.selected.lr.max_iter || !this.isInt(this.selected.lr.max_iter)) {
          this.errors.max_iter = this.$t('msgValIntOnly')
          isValid = false
        }
      } else if (this.selected.model_type === "tabnet") {
        if (!this.selected.tabnet.batch_size || !this.isInt(this.selected.tabnet.batch_size)) {
          this.errors.batch_size = this.$t('msgValIntOnly')
          isValid = false
        }
        if (!this.selected.tabnet.max_epochs || !this.isInt(this.selected.tabnet.max_epochs)) {
          this.errors.max_epochs = this.$t('msgValIntOnly')
          isValid = false
        }
        if (!this.selected.tabnet.patience || !this.isInt(this.selected.tabnet.patience)) {
          this.errors.patience = this.$t('msgValIntOnly')
          isValid = false
        }
      } else if (this.selected.model_type === "mlp") {
        if (!this.selected.mlp.hidden_layer_1 || !this.isInt(this.selected.mlp.hidden_layer_1)) {
          this.errors.hidden_layer_1 = this.$t('msgValIntOnly')
          isValid = false
        }
        if (this.selected.mlp.hidden_layer_2 && !this.isInt(this.selected.mlp.hidden_layer_2)) {
          this.errors.hidden_layer_2 = this.$t('msgValIntOnly')
          isValid = false
        }
        if (this.selected.mlp.hidden_layer_3) {
          if (!this.selected.mlp.hidden_layer_2) {
            this.errors.hidden_layer_3 = this.$t('msgValRequireLayer2')
            isValid = false
          }
          if (!this.isInt(this.selected.mlp.hidden_layer_3)) {
            this.errors.hidden_layer_3 = this.$t('msgValIntOnly')
            isValid = false
          }
        }
        if (!this.selected.mlp.activation) {
          this.errors.activation = this.$t('msgValRequired')
          isValid = false
        }
        if (!this.selected.mlp.learning_rate_init || !this.isFloat(this.selected.mlp.learning_rate_init)) {
          this.errors.learning_rate_init = this.$t('msgValFloatOnly')
          isValid = false
        }
        if (!this.selected.mlp.max_iter || !this.isInt(this.selected.mlp.max_iter)) {
          this.errors.max_iter = this.$t('msgValIntOnly')
          isValid = false
        }
        if (!this.selected.mlp.n_iter_no_change || !this.isInt(this.selected.mlp.n_iter_no_change)) {
          this.errors.n_iter_no_change = this.$t('msgValIntOnly')
          isValid = false
        }
      } else if (this.selected.model_type === "catboost") {
        if (!this.selected.catboost.iterations || !this.isInt(this.selected.catboost.iterations)) {
          this.errors.iterations = this.$t('msgValIntOnly')
          isValid = false
        }
        if (!this.selected.catboost.learning_rate || !this.isFloat(this.selected.catboost.learning_rate)) {
          this.errors.learning_rate = this.$t('msgValFloatOnly')
          isValid = false
        }
        if (!this.selected.catboost.depth || !this.isInt(this.selected.catboost.depth)) {
          this.errors.depth = this.$t('msgValIntOnly')
          isValid = false
        }
      } else if (this.selected.model_type === "adaboost") {
        if (!this.selected.adaboost.n_estimators || !this.isInt(this.selected.adaboost.n_estimators)) {
          this.errors.n_estimators = this.$t('msgValIntOnly')
          isValid = false
        }
        if (!this.selected.adaboost.learning_rate || !this.isFloat(this.selected.adaboost.learning_rate)) {
          this.errors.learning_rate = this.$t('msgValFloatOnly')
          isValid = false
        }
        if (!this.selected.adaboost.depth || !this.isInt(this.selected.adaboost.depth)) {
          this.errors.depth = this.$t('msgValIntOnly')
          isValid = false
        }
      } else if (this.selected.model_type === "svm") {
        if (!this.selected.svm.C || !this.isFloat(this.selected.svm.C)) {
          this.errors.C = this.$t('msgValFloatOnly')
          isValid = false
        }
        if (!this.selected.svm.kernel) {
          this.errors.kernel = this.$t('msgValRequired')
          isValid = false
        }
      }

      // File Selection (data)
      if (!this.selected.data) {
        this.errors.data = this.$t('msgValRequired')
        isValid = false
      }

      // Outcome Column (label_column)
      if (!this.selected.label_column) {
        this.errors.label_column = this.$t('msgValRequired')
        isValid = false
      }

      // Model Saved as (model_name)
      if (!this.selected.model_name) {
        this.errors.model_name = this.$t('msgValRequired')
        isValid = false
      }

      return isValid
    },

    async previewTab(showMissingModal) {
      this.initPreviewData()
      this.missing_cords = []
      this.missing_header = []
      this.missing_methods = []
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/preview-tabular`, {
          file_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data}`, // upload/
        })
        if (response.data.status == "success") {
          this.preview_data = response.data.preview_data
          this.summary = response.data.summary
          this.missing_cords = response.data.missing_cords
          this.missing_header = response.data.missing_header
          if (this.missing_cords && this.missing_cords.length > 0 && showMissingModal) {
            this.modal.title = this.$t('lblError')
            this.modal.content = this.$t('msgMissingDataFound') + response.data.missing_cords
            this.modal.icon = 'error'
            this.openModalNotification()
          }
        // } else if (response.data.status == "errorMissing") {
        //   this.modal.title = this.$t('lblError')
        //   this.modal.content = this.$t('msgMissingDataFound') + response.data.message + '\n' + this.$t('msgConfirmDeleteRows')
        //   this.modal.icon = 'error'
        //   this.openModalMissingData()
        } else if (response.data.status == "error") {
          this.modal.title = this.$t('lblError')
          this.modal.content = response.data.message
          this.modal.icon = 'error'
          this.initPreviewData()
          this.selected.data = ''
          this.openModalNotification()
        }
      } catch (error) {
        this.modal.title = this.$t('lblError')
        this.modal.content = error
        this.modal.icon = 'error'
        this.openModalNotification()
        this.initPreviewData()
        this.selected.data = ''
      }
      this.loading = false
    },
    
    async runTrain() {
      if (!this.validateForm()) {
        return
      }

      this.loading = true

      // 檢查 label 是否只有一種 class
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/check-label-uniqueness`, {
          file_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data}`, // upload/
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

      // preprocessing
      if (this.unhandledMissingColumns.length > 0) {
        this.modal.title = this.$t('lblError')
        this.modal.content = this.$t('msgMissingNotHandled') + this.unhandledMissingColumns.join(', ')
        this.modal.icon = 'error'
        this.openModalNotification()
        this.loading = false
        return
      } else if(this.rules.length != 0) {
        try {
          const response = await axios.post(`${process.env.VUE_APP_API_URL}/preprocess`, {
            file_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data}`, // upload/
            rules: this.rules
          })
          if (response.data.status == "success") {
            this.modal.title = this.$t('lblDataPreprocessing')
            this.modal.content = response.data.message
            this.modal.icon = 'info'
            this.openModalNotification()
            await this.previewTab(false)
            // this.$refs.preprocessor.resetExceptSkip()
            this.loading = true // previewTab() 會取消 loading = =
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
      }

      // train
      try {
        this.output = null
        let api = ''
        let payload = {
          file_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data}`, // upload/
          label_column: this.selected.label_column,
          split_strategy: this.selected.split_strategy,
          split_value: this.selected.split_value,
          model_name: this.selected.model_name,
          username: sessionStorage.getItem('username'),
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
          payload["n_iter_no_change"] = this.selected.mlp.n_iter_no_change
        } else if (this.selected.model_type == "catboost") {
          api = "run-train-catboost"
          payload["iterations"] = this.selected.catboost.iterations
          payload["learning_rate"] = this.selected.catboost.learning_rate
          payload["depth"] = this.selected.catboost.depth
        } else if (this.selected.model_type == "adaboost") {
          api = "run-train-adaboost"
          payload["n_estimators"] = this.selected.adaboost.n_estimators
          payload["learning_rate"] = this.selected.adaboost.learning_rate
          payload["depth"] = this.selected.adaboost.depth
        } else if (this.selected.model_type == "svm") {
          api = "run-train-svm"
          payload["C"] = this.selected.svm.C
          payload["kernel"] = this.selected.svm.kernel
        } else {
          this.output = {
            "status": "error",
            "message": "Unsupported model type"
          }
          return
        }

        console.log(payload)

        // 提示 Random Forest 很耗時
        if (this.selected.model_type == 'random_forest') {
          this.modal.title = this.$t('lblRandomForest')
          this.modal.content = this.$t('msgTrainLonger')
          this.modal.icon = 'info'
          this.openModalNotification()
        }

        const response = await axios.post(`${process.env.VUE_APP_API_URL}/${api}`, payload)
        if (this.isUnmounted) return // 若頁面已離開就不要繼續處理
        this.output = response.data
        if (this.output.status == 'success') {
          // 顯示用，讓切換 split_strategy 不會拿掉 results
          this.output.split_strategy = this.selected.split_strategy
          this.modal.title = this.$t('lblSuccess')
          this.modal.content = this.$t('msgTrainingCompleted')
          this.modal.icon = 'success'
          this.openModalNotification()
          if (this.selected.split_strategy == "train_test_split") {
            this.imageRoc = `data:image/png;base64,${this.output.roc}`
            this.imageLoss = `data:image/png;base64,${this.output.loss_plot}`
            this.imageAccuracy = `data:image/png;base64,${this.output.accuracy_plot}`
            this.imageShap = `data:image/png;base64,${this.output.shap_plot}`
            this.imageLime = `data:image/png;base64,${this.output.lime_plot}`
          }
        } else if (this.output.status == 'error') {
          this.modal.title = this.$t('lblError')
          this.modal.content = this.output.message
          this.modal.icon = 'error'
          this.openModalNotification()
          this.output = null
        }
      } catch (error) {
        if (this.isUnmounted) return // 頁面已離開就忽略錯誤處理
        this.output = {
          "status": "error",
          "message": error,
        }
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

    // 處理缺失值
    async preprocess() {
      if (!this.validatePreprocess()) {
        return
      }

      this.loading = true
      try {
        const raw = toRaw(this.missing_methods)
        // 防呆修復：若是 array+屬性混用的 proxy，就重建成 dict
        let fixedDict = {}
        Object.keys(raw).forEach(key => {
          if (isNaN(Number(key))) {
            fixedDict[key] = raw[key]
          }
        })

        const response = await axios.post(`${process.env.VUE_APP_API_URL}/preprocess`, {
          file_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data}`, // upload/
          missing_methods: fixedDict
        })
        if (response.data.status == "success") {
          const skipped = Object.keys(fixedDict).filter(col => fixedDict[col] === 'skip')
          this.modal.title = this.$t('lblSuccess')
          this.modal.icon = 'success'
          this.modal.content = this.$t('msgFinishHandleMissing')
          if (skipped.length > 0) {
            this.modal.content += '\n' + this.$t('msgSkipWarning', { columns: skipped.join(', ') })
          }
          this.openModalNotification()
          this.loading = false
          await this.previewTab(false)
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

    validatePreprocess() {
      let isValid = true
      this.errors_preprocess = {}
      for (const header of this.missing_header) {
        const value = this.missing_methods[header]
        if (!value) {
          this.errors_preprocess[header] = this.$t('msgValRequired')
          isValid = false
        }
      }
      return isValid
    },

    handleRuleUpdate(newRules) {
      this.rules = newRules
    },

    onUnhandledMissingUpdate(cols) {
      this.unhandledMissingColumns = cols
    }
  },
}
</script>
