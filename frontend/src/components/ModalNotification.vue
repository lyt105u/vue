<template>
  <div class="modal fade" ref="modalRef" tabindex="-1" aria-labelledby="finishTrainingModalLabel" aria-hidden="true">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h1 class="modal-title fs-5">{{ title }}</h1>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div class="bd-example-snippet bd-code-snippet">
            <div class="bd-example m-0 border-0">
              {{ content }}
            </div>
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
    title: {
      type: String,
      required: true,
    },
    content: {
      type: String,
      required: true,
    },
  },
  data() {
    return {
      modalInstance: null,
    };
  },
  mounted() {
    if (this.$refs.modalRef) {
      this.modalInstance = new Modal(this.$refs.modalRef);
    }
  },
  methods: {
    openModal() {
      if (this.modalInstance) {
        this.modalInstance.show();
      } else {
        console.error("Modal instance is not initialized.");
      }
    },
  },
};
</script>
