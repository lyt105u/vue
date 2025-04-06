<template>
    <div class="modal fade" ref="modalRef" tabindex="-1" aria-labelledby="modalImageLabel" aria-hidden="true">
      <div class="modal-dialog modal-lg">
        <div class="modal-content">
          <div class="modal-header">
            <h1 class="modal-title fs-5" id="modalImageLabel">{{ title }}</h1>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body text-center">
            <img :src="imageSrc" :alt="title" class="img-fluid" />
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
        required: true
      },
      imageSrc: {
        type: String,
        required: true
      }
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
  