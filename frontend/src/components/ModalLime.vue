<template>
  <div class="modal fade" ref="modalRef" tabindex="-1" aria-labelledby="modalImageLabel" aria-hidden="true">
    <div class="modal-dialog modal-lg">
      <div class="modal-content">
        <div class="modal-header">
          <h1 class="modal-title fs-5" id="modalImageLabel">{{ $t('lblLime') }}</h1>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div class="mb-3">
            <small>
              {{ $t('msgLimeExplanation1') }}
              <br>
              {{ $t('msgLimeExplanation2') }}
            </small>
          </div>
          <div class="text-center">
            <img :src="imageSrc" :alt="$t('lblLime')" class="img-fluid" />
          </div>
          <div>
            <!-- <small>
              The values below represent the average absolute SHAP value for each feature across all samples.
              A higher value means the feature tends to have a stronger influence on the model predictions.
            </small> -->
            <table class="table table-striped">
              <thead>
                <tr>
                  <th scope="col">{{ $t('lblCondition') }}</th>
                  <th scope="col">{{ $t('lblFeature') }}</th>
                  <th scope="col">{{ $t('lblContribution') }}</th>
                  <th scope="col">{{  $t('lblEffect') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(item, index) in limeParsed" :key="index">
                  <td><code>{{ item.condition }}</code></td>
                  <td>{{ item.columnName }}</td>
                  <td>{{ item.weight.toFixed(4) }}</td>
                  <td :style="{ color: item.weight > 0 ? 'green' : 'red' }">
                    {{ item.weight > 0 ? $t('lblIncreasePrediction') : $t('lblDecreasePrediction') }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">{{ $t('lblClose') }}</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { Modal } from "bootstrap";
  
export default {
  props: {
    imageSrc: {
      type: String,
      required: true
    },
    lime_example_0: {
      type: Array,
      default: () => [],  // 預設值為空陣列
      required: true
    },
    columns: {
      type: Array,
      required: true,
    },
  },
  data() {
    return {
      modalInstance: null,
    }
  },
  mounted() {
    if (this.$refs.modalRef) {
      this.modalInstance = new Modal(this.$refs.modalRef);
    }
  },
  computed: {
    limeParsed() {
      if (!Array.isArray(this.lime_example_0)) return []

      // 工具函式：轉義正則特殊字元
      const escapeRegExp = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')

      return this.lime_example_0.map(([condition, weight]) => {
        let columnName = '(Unknown)';

        for (const col of this.columns) {
          // 判斷 col 是否是英文或包含數字
          const isEnglish = /^[\w\s\-()]+$/.test(col);  // 英文、數字、底線、括號等

          const regex = isEnglish
            ? new RegExp(`\\b${escapeRegExp(col)}\\b`)
            : new RegExp(escapeRegExp(col));  // 中文直接整體比對

          if (regex.test(condition)) {
            columnName = col;
            break;
          }
        }

        return { condition, weight, columnName }
      })
    }
  },
  methods: {
    openModal() {
      if (this.modalInstance) {
        this.modalInstance.show();
      }
    },
  },
};
</script>
  
<style scoped>
.img-fluid {
  max-height: 80vh;
  object-fit: contain;
}
</style>  
