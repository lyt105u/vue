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
              The SHAP plot below shows how each feature contributed to the model’s predictions across all test samples.
              Each dot represents a single sample. The horizontal position indicates how much a feature increased or decreased the model output.
            </small>
          </div>
          <div class="text-center">
            <img :src="imageSrc" alt="SHAP" class="img-fluid" />
          </div>
          <div>
            <small>
              The values below represent the average absolute SHAP value for each feature across all samples.
              A higher value means the feature tends to have a stronger influence on the model predictions.
            </small>
            <table class="table table-striped">
              <thead>
                <tr>
                  <th scope="col">#</th>
                  <th scope="col">Feature</th>
                  <th scope="col">Average SHAP Value</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="item in sortedShapImportance" :key="item.key">
                  <td>{{ item.key }}</td>
                  <td>{{ item.columnName }}</td>
                  <td>{{ item.value.toFixed(4) }}</td>
                </tr>
              </tbody>
            </table>
            <!-- <ul>
              <li v-for="(value, feature) in sortedShapImportance" :key="feature">
                Feature <strong>{{ feature }}</strong> has an average contribution of 
                <strong>{{ value.toFixed(4) }}</strong> to the prediction.
              </li>
            </ul> -->
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
    shapImportance: {
      type: Object,
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
    sortedShapImportance() {
      return Object.entries(this.shapImportance)
        .map(([key, value]) => {
          const index = parseInt(key.split('_')[1]) // e.g., "feature_3" -> 3
          const columnName = this.columns[index] || key
          return { key, columnName, value }
        })
        .sort((a, b) => b.value - a.value)
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
