<template>
  <div class="modal fade" ref="modalRef" tabindex="-1" aria-labelledby="modalImageLabel" aria-hidden="true">
    <div class="modal-dialog modal-lg">
      <div class="modal-content">
        <div class="modal-header">
          <h1 class="modal-title fs-5" id="modalImageLabel">{{ $t('lblShap') }}</h1>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div class="mb-3">
            <small>
              {{ $t('msgShapExplain1') }}
              {{ $t('msgShapExplain2') }}
            </small>
          </div>
          <div class="text-center">
            <img :src="imageSrc" alt="SHAP" class="img-fluid" />
          </div>
          <div>
            <small>
              {{ $t('msgShapExplain3') }}
              {{ $t('msgShapExplain4') }}
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
                  <td><code>{{ item.key }}</code></td>
                  <td>{{ item.columnName }}</td>
                  <td>{{ item.value.toFixed(4) }}</td>
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
