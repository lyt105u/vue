<template>
  <div class="about">
    <h1>{{ $t('lblStacking') }}</h1>
    <h6 class="text-body-secondary">{{ $t('msgPredictDescription') }}</h6>
  </div>

  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <form class="row g-3" @submit.prevent="runTrain" style="margin-top: 16px">
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
              </tfoot>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- 缺失值處理 -->
    <template v-if="missing_cords && missing_cords.length > 0">
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
        <!-- ✅ 只有第一列顯示按鈕 -->
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
    </template>

    <!-- Predict Column -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblPredictionColumn') }}</label>
      <div class="col-sm-8">
        <input v-model="selected.label_column" class="form-control" type="text" :disabled="loading">
        <div v-if="errors.label_column" class="text-danger small">{{ errors.label_column }}</div>
      </div>
    </div>

    <!-- Base Model -->
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">{{ $t('lblBaseModel') }}</label>
      <div class="col-sm-8">
        <div v-for="model in modelOptions" :key="model.value" class="form-check form-check-inline">
          <input
            type="checkbox"
            :id="model.value"
            :value="model.value"
            v-model="selected.base_models"
            class="form-check-input"
            :disabled="loading"
          />
          <label :for="model.value" class="form-check-label">{{ $t(model.label) }}</label>
        </div>
        <p v-if="errors.base_models" class="text-danger small">{{ errors.base_models }}</p>
      </div>
    </div>

    <!-- Meta Model -->
    <div class="row mb-3">
      <label class="col-sm-3 col-form-label">{{ $t('lblMetaModel') }}</label>
      <div class="col-sm-8">
        <select class="form-select" aria-label="Small select example" v-model="selected.meta_model" :disabled="loading">
          <option v-for="model in modelOptions" :key="model.value" :value="model.value">
            {{ $t(model.label) }}
          </option>
        </select>
        <div v-if="errors.meta_model" class="text-danger small">{{ errors.meta_model }}</div>
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

  <!-- hr -->
  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <!-- Results 標題 -->
  <div v-if="output" class="about d-flex align-items-center gap-2" style="padding-bottom:12px;">
    <h3 class="mb-0 d-flex align-items-center">{{ $t('lblPredictionResult') }}</h3>
    <button v-if="!loading" @click="downloadReport" type="button" class="btn btn-outline-primary">
      <i class="fa fa-download me-1"></i>{{ $t('lblDownload') }}
    </button>
    <button v-if="loading" class="btn btn-outline-primary" type="button" disabled>
      <span class="spinner-border spinner-border-sm" aria-hidden="true"></span>
    </button>
  </div>

  <!-- output -->
  <div v-if="output" class="accordion" id="accordionExample">
    <div class="accordion-item">
      <h2 class="accordion-header" @click="toggleCollapseResult('meta')">
        <button class="accordion-button collapsed" type="button">
          {{ $t('lblMetaModel') }} ( {{ $t(modelOptions.find(opt => opt.value === output.meta_model)?.label || '') }} )
        </button>
      </h2>
      <div class="accordion-collapse collapse" ref="meta">
        <div class="accordion-body">
          <div class="row row-cols-1 row-cols-md-3 mb-3 text-center">
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
                              <td>{{ output.meta_results.confusion_matrix.true_positive }}</td>
                              <td>{{ output.meta_results.confusion_matrix.false_negative }}</td>
                            </tr>
                            <tr>
                              <td>{{ output.meta_results.confusion_matrix.false_positive }}</td>
                              <td>{{ output.meta_results.confusion_matrix.true_negative }}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                    <li>Accuracy : {{ output.meta_results.metrics.accuracy.toFixed(2) }}%</li>
                    <li>Recall : {{ output.meta_results.metrics.recall.toFixed(2) }}%</li>
                    <li>Precision : {{ output.meta_results.metrics.precision.toFixed(2) }}%</li>
                    <li>F1_score : {{ output.meta_results.metrics.f1_score.toFixed(2) }}%</li>
                  </ul>
                </div>
              </div>
            </div>
            <!-- recall 列表 -->
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
                              <td>{{ output.meta_results[recall.key].true_positive }}</td>
                              <td>{{ output.meta_results[recall.key].false_negative }}</td>
                            </tr>
                            <tr>
                              <td>{{ output.meta_results[recall.key].false_positive }}</td>
                              <td>{{ output.meta_results[recall.key].true_negative }}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                    <li>Recall: {{ output.meta_results[recall.key].recall.toFixed(2) }}%</li>
                    <li>Specificity: {{ output.meta_results[recall.key].specificity.toFixed(2) }}%</li>
                    <li>Precision: {{ output.meta_results[recall.key].precision.toFixed(2) }}%</li>
                    <li>NPV: {{ output.meta_results[recall.key].npv.toFixed(2) }}%</li>
                    <li>F1 Score: {{ output.meta_results[recall.key].f1_score.toFixed(2) }}%</li>
                    <li>F2 Score: {{ output.meta_results[recall.key].f2_score.toFixed(2) }}%</li>
                    <li>Accuracy: {{ output.meta_results[recall.key].accuracy.toFixed(2) }}%</li>
                  </ul>
                </div>
              </div>
            </div>
            <!-- ROC 曲線 -->
            <div class="col">
              <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalImage($t('lblRocCurve'), `data:image/png;base64,${output.meta_results.roc}`)" style="cursor: pointer;">
                <div class="card-header py-3">
                  <h4 class="my-0 fw-normal">{{ $t('lblRocCurve') }}</h4>
                </div>
                <img :src="`data:image/png;base64,${output.meta_results.roc}`" :alt="$t('lblRocCurve')" />
              </div>
            </div>

            <!-- Loss 曲線 -->
            <div v-if="output.meta_results.loss_plot" class="col">
              <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalImage('Loss', `data:image/png;base64,${output.meta_results.loss_plot}`)" style="cursor: pointer;">
                <div class="card-header py-3">
                  <h4 class="my-0 fw-normal">Loss</h4>
                </div>
                <img :src="`data:image/png;base64,${output.meta_results.loss_plot}`" alt="Loss" />
              </div>
            </div>

            <!-- Accuracy 曲線 -->
            <div v-if="output.meta_results.accuracy_plot" class="col">
              <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalImage('Accuracy', `data:image/png;base64,${output.meta_results.accuracy_plot}`)" style="cursor: pointer;">
                <div class="card-header py-3">
                  <h4 class="my-0 fw-normal">Accuracy</h4>
                </div>
                <img :src="`data:image/png;base64,${output.meta_results.accuracy_plot}`" alt="Accuracy" />
              </div>
            </div>

            <!-- SHAP -->
            <div v-if="!output.meta_results.shap_error" class="col">
              <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalShap(`data:image/png;base64,${output.meta_results.shap_plot}`, output.meta_results.shap_importance)" style="cursor: pointer;">
                <div class="card-header py-3">
                  <h4 class="my-0 fw-normal">{{ $t('lblShap') }}</h4>
                </div>
                <img :src="`data:image/png;base64,${output.meta_results.shap_plot}`" :alt="$t('lblShap')" />
              </div>
            </div>

            <!-- LIME -->
            <div v-if="!output.meta_results.lime_error" class="col">
              <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalLime(`data:image/png;base64,${output.meta_results.lime_plot}`, output.meta_results.lime_example_0)" style="cursor: pointer;">
                <div class="card-header py-3">
                  <h4 class="my-0 fw-normal">{{ $t('lblLime') }}</h4>
                </div>
                <img :src="`data:image/png;base64,${output.meta_results.lime_plot}`" :alt="$t('lblLime')" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-for="base_model in output.base_models" :key="base_model" class="accordion-item">
      <h2 class="accordion-header" @click="toggleCollapseResult(`collapse-${base_model}`)">
        <button class="accordion-button collapsed" type="button">
          {{ $t('lblBaseModel') }} ( {{ $t(modelOptions.find(opt => opt.value === base_model)?.label || '') }} )
        </button>
      </h2>
      <div class="accordion-collapse collapse" :ref="`collapse-${base_model}`">
        <div class="accordion-body">
          <div class="row row-cols-1 row-cols-md-3 mb-3 text-center">
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
                              <td>{{ output.base_results[base_model]?.confusion_matrix.true_positive }}</td>
                              <td>{{ output.base_results[base_model]?.confusion_matrix.false_negative }}</td>
                            </tr>
                            <tr>
                              <td>{{ output.base_results[base_model]?.confusion_matrix.false_positive }}</td>
                              <td>{{ output.base_results[base_model]?.confusion_matrix.true_negative }}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                    <li>Accuracy : {{ output.base_results[base_model]?.metrics.accuracy.toFixed(2) }}%</li>
                    <li>Recall : {{ output.base_results[base_model]?.metrics.recall.toFixed(2) }}%</li>
                    <li>Precision : {{ output.base_results[base_model]?.metrics.precision.toFixed(2) }}%</li>
                    <li>F1_score : {{ output.base_results[base_model]?.metrics.f1_score.toFixed(2) }}%</li>
                  </ul>
                </div>
              </div>
            </div>
            <!-- recall 列表 -->
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
                              <td>{{ output.base_results[base_model][recall.key].true_positive }}</td>
                              <td>{{ output.base_results[base_model][recall.key].false_negative }}</td>
                            </tr>
                            <tr>
                              <td>{{ output.base_results[base_model][recall.key].false_positive }}</td>
                              <td>{{ output.base_results[base_model][recall.key].true_negative }}</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                    <li>Recall: {{ output.base_results[base_model][recall.key].recall.toFixed(2) }}%</li>
                    <li>Specificity: {{ output.base_results[base_model][recall.key].specificity.toFixed(2) }}%</li>
                    <li>Precision: {{ output.base_results[base_model][recall.key].precision.toFixed(2) }}%</li>
                    <li>NPV: {{ output.base_results[base_model][recall.key].npv.toFixed(2) }}%</li>
                    <li>F1 Score: {{ output.base_results[base_model][recall.key].f1_score.toFixed(2) }}%</li>
                    <li>F2 Score: {{ output.base_results[base_model][recall.key].f2_score.toFixed(2) }}%</li>
                    <li>Accuracy: {{ output.base_results[base_model][recall.key].accuracy.toFixed(2) }}%</li>
                  </ul>
                </div>
              </div>
            </div>
            <!-- ROC 曲線 -->
            <div class="col">
              <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalImage($t('lblRocCurve'), `data:image/png;base64,${output.base_results[base_model]?.roc}`)" style="cursor: pointer;">
                <div class="card-header py-3">
                  <h4 class="my-0 fw-normal">{{ $t('lblRocCurve') }}</h4>
                </div>
                <img :src="`data:image/png;base64,${output.base_results[base_model]?.roc}`" :alt="$t('lblRocCurve')" />
              </div>
            </div>

            <!-- Loss 曲線 -->
            <div v-if="output.base_results[base_model]?.loss_plot" class="col">
              <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalImage('Loss', `data:image/png;base64,${output.base_results[base_model]?.loss_plot}`)" style="cursor: pointer;">
                <div class="card-header py-3">
                  <h4 class="my-0 fw-normal">Loss</h4>
                </div>
                <img :src="`data:image/png;base64,${output.base_results[base_model]?.loss_plot}`" alt="Loss" />
              </div>
            </div>

            <!-- Accuracy 曲線 -->
            <div v-if="output.base_results[base_model]?.accuracy_plot" class="col">
              <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalImage('Accuracy', `data:image/png;base64,${output.base_results[base_model]?.accuracy_plot}`)" style="cursor: pointer;">
                <div class="card-header py-3">
                  <h4 class="my-0 fw-normal">Accuracy</h4>
                </div>
                <img :src="`data:image/png;base64,${output.base_results[base_model]?.accuracy_plot}`" alt="Accuracy" />
              </div>
            </div>

            <!-- SHAP -->
            <div v-if="!output.base_results[base_model]?.shap_error" class="col">
              <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalShap(`data:image/png;base64,${output.base_results[base_model]?.shap_plot}`, output.base_results[base_model]?.shap_importance)" style="cursor: pointer;">
                <div class="card-header py-3">
                  <h4 class="my-0 fw-normal">{{ $t('lblShap') }}</h4>
                </div>
                <img :src="`data:image/png;base64,${output.base_results[base_model]?.shap_plot}`" :alt="$t('lblShap')" />
              </div>
            </div>

            <!-- LIME -->
            <div v-if="!output.base_results[base_model]?.lime_error" class="col">
              <div class="card mb-4 rounded-3 shadow-sm"  @click="openModalLime(`data:image/png;base64,${output.base_results[base_model]?.lime_plot}`, output.base_results[base_model]?.lime_example_0)" style="cursor: pointer;">
                <div class="card-header py-3">
                  <h4 class="my-0 fw-normal">{{ $t('lblLime') }}</h4>
                </div>
                <img :src="`data:image/png;base64,${output.base_results[base_model]?.lime_plot}`" :alt="$t('lblLime')" />
              </div>
            </div>
          </div>
        </div>
      </div>
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
  <ModalFormulaExplain ref="formulaExplainModal" />
  <ModalImage ref="modalImageRef" :title="modal.title" :imageSrc="modal.content"/>
  <ModalShap ref="modalShapRef" :imageSrc="modal.content" :shapImportance="modal.shap_importance" :columns="preview_data.columns"/>
  <ModalLime ref="modalLimeRef" :imageSrc="modal.content" :lime_example_0="modal.lime_example_0" :columns="preview_data.columns"/>
</template>

<script>
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import axios from 'axios';
import ModalNotification from "@/components/ModalNotification.vue"
import ModalFormulaExplain from "@/components/ModalFormulaExplain.vue"
import ModalImage from "@/components/ModalImage.vue"
import ModalShap from "@/components/ModalShap.vue"
import ModalLime from "@/components/ModalLime.vue"
import { Collapse } from 'bootstrap'
import { toRaw } from 'vue'

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
        base_models: [],
        data_name: '',
        label_column: '',
        meta_model: '',
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
      errors_preprocess: {},
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
        { value: 'skip', label: 'lblSkip' },
        { value: 'zero', label: 'lblZero'},
      ],
      modelOptions: [
        { value: 'xgb', label: 'lblXgb' },
        { value: 'lgbm', label: 'lblLightGBM' },
        { value: 'rf', label: 'lblRandomForest' },
        { value: 'lr', label: 'lblLogisticRegression' },
        { value: 'tabnet', label: 'lblTabNet' },
        { value: 'mlp', label: 'lblMultiLayerPerceptron' },
      ],
      fileOptions: [],
      isUnmounted: false, // 防止跳轉後，API執行完仍繼續執行js，造成錯誤
      recallLevels: [
        { level: 80, key: 'recall_80' },
        { level: 85, key: 'recall_85' },
        { level: 90, key: 'recall_90' },
        { level: 95, key: 'recall_95' }
      ],
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
    rows() {
      const result = [];
      for (let i = 0; i < this.missing_header.length; i += 4) {
        result.push(this.missing_header.slice(i, i + 4))
      }
      return result
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
    "selected.data_name"() {
      if (this.selected.data_name != '') {
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
        event.returnValue = '' // 必需，讓瀏覽器顯示警示對話框
      }
    },

    // list both model and tabular files
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

    initPreviewData() {
      this.preview_data = {
        columns: [],
        preview: [],
        total_rows: 0,
        total_columns: 0
      }
    },

    async previewTab(showMissingModal) {
      this.initPreviewData()
      this.missing_cords = []
      this.missing_header = []
      this.missing_methods = []
      this.loading = true
      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/preview-tabular`, {
          file_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data_name}`, // upload/
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

    toggleCollapseResult(refName) {
      const refRaw = this.$refs[refName]
      const collapseEl = Array.isArray(refRaw) ? refRaw[0] : refRaw

      if (!collapseEl) return

      const buttonEl = collapseEl.previousElementSibling?.querySelector('.accordion-button')
      if (!buttonEl) return

      let instance = Collapse.getInstance(collapseEl)
      if (!instance) {
        instance = new Collapse(collapseEl, { toggle: false })
      }

      instance.toggle()
      buttonEl.classList.toggle('collapsed')
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
          await this.previewTab(true)
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

      // File Selection
      if (!this.selected.data_name) {
        this.errors.data_name = this.$t('msgValRequired')
        isValid = false
      }
      // Models
      if (this.selected.base_models.length < 2) {
        this.errors.base_models = this.$t('msgValSelect2')
        isValid = false
      }
      // Outcome Column
      if (!this.selected.label_column) {
        this.errors.label_column = this.$t('msgValRequired')
        isValid = false
      }
      // Meta Model
      if (!this.selected.meta_model) {
        this.errors.meta_model = this.$t('msgValRequired')
        isValid = false
      }

      return isValid
    },

    async runTrain() {
      if (!this.validateForm()) {
        return
      }

      this.loading = true
      this.output = null

      try {
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/run-train-stacking`,
          {
            base_models: this.selected.base_models,
            data_name: `upload/${sessionStorage.getItem('username')}/${this.selected.data_name}`, // upload/
            label_column: this.selected.label_column,
            meta_model: this.selected.meta_model,
            username: sessionStorage.getItem('username'),
          },
        )
        if (this.isUnmounted) return // 若頁面已離開就不要繼續處理
        if (response.data.status == "success") {
          this.output = response.data
          this.modal.title = this.$t('lblPredictionCompleted')
          this.modal.icon = 'success'
          this.modal.content = response.data
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
        const response = await axios.post(`${process.env.VUE_APP_API_URL}/download-report`, this.output, {
          responseType: 'blob' // 關鍵：支援二進位檔案格式
        })

        // 從 Content-Disposition 擷取檔案名稱
        let filename = 'report.zip' // 預設檔名
        const disposition = response.headers['content-disposition']
        if (disposition && disposition.includes('filename=')) {
          const match = disposition.match(/filename="?([^"]+)"?/)
          if (match) {
            filename = decodeURIComponent(match[1]) // 使用後端提供的檔名（如：report_20250627_160500.zip）
          }
        }

        const blob = response.data
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = filename
        a.click()
        URL.revokeObjectURL(url)
      } catch (err) {
        this.modal.title = this.$t('lblError')
        this.modal.content = err
        this.modal.icon = 'error'
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
          file_path: `upload/${sessionStorage.getItem('username')}/${this.selected.data_name}`, // upload/
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
  },
}
</script>
