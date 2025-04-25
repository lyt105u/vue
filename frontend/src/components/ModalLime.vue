<template>
  <div class="modal fade" ref="modalRef" tabindex="-1" aria-labelledby="modalImageLabel" aria-hidden="true">
    <div class="modal-dialog modal-lg">
      <div class="modal-content">
        <div class="modal-header">
          <h1 class="modal-title fs-5" id="modalImageLabel">SHAP</h1>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div class="mb-3">
            <small>
              The following plot and table explain how individual feature conditions contributed to the model’s prediction for the first sample.
              LIME builds a local interpretable model to approximate the complex model around this instance.
            </small>
          </div>
          <div class="text-center">
            <img :src="imageSrc" alt="SHAP" class="img-fluid" />
          </div>
          <div>
            <!-- <small>
              The values below represent the average absolute SHAP value for each feature across all samples.
              A higher value means the feature tends to have a stronger influence on the model predictions.
            </small> -->
            <table class="table table-striped">
              <thead>
                <tr>
                  <th scope="col">Condition</th>
                  <th scope="col">Feature</th>
                  <th scope="col">Contribution</th>
                  <th scope="col">Effect</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(item, index) in limeParsed" :key="index">
                  <td><code>{{ item.condition }}</code></td>
                  <td>{{ item.columnName }}</td>
                  <td>{{ item.weight.toFixed(4) }}</td>
                  <td :style="{ color: item.weight > 0 ? 'green' : 'red' }">
                    {{ item.weight > 0 ? 'Increases prediction' : 'Decreases prediction' }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
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
      return this.lime_example_0.map(([condition, weight]) => {
        const match = condition.match(/feature_(\d+)/);
        const index = match ? parseInt(match[1]) : null;
        const columnName = index !== null && this.columns[index] ? this.columns[index] : '(Unknown)';
        return { condition, weight, columnName };
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
