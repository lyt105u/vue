<template>
  <div class="modal fade" ref="modalRef" tabindex="-1" aria-labelledby="finishTrainingModalLabel" aria-hidden="true">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h1 class="modal-title fs-5 d-flex align-items-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              :fill="iconPath.fill"
              class="me-2"
              viewBox="0 0 16 16"
            >
              <path :d="iconPath.path" />
            </svg>
            {{ title }}
          </h1>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div class="bd-example-snippet bd-code-snippet">
            <div class="bd-example m-0 border-0" style="white-space: pre-line;">
              {{ content }}
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <!-- <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button> -->
          <button
            v-if="secondaryButton"
            type="button"
            class="btn btn-secondary"
            @click="handleSecondaryClick"
            :data-bs-dismiss="secondaryButton.dismiss ? 'modal' : null"
          >
            {{ secondaryButton.text || $t('lblClose') }}
          </button>
          <button
            v-if="primaryButton"
            type="button"
            class="btn btn-primary"
            @click="handlePrimaryClick"
          >
            {{ primaryButton.text || $t('lblOk') }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { Modal } from "bootstrap"

export default {
  // 只有 Close 按鈕（預設）：
  // <Modal
  //   :title="'完成'"
  //   :content="'是否下載結果？'"
  //   :primaryButton="{ text: '下載', onClick: downloadResult }"
  // />

  // 加上 primary（含點擊功能）：
  // <Modal
  //   :title="'完成'"
  //   :content="'是否下載結果？'"
  //   :primaryButton="{ text: '下載', onClick: downloadResult }"
  // />

  // 自訂 secondary（不關閉 modal）：
  // <FinishTrainingModal
  //   :title="'提醒'"
  //   :content="'請確認操作'"
  //   :primaryButton="{ text: '確定', onClick: handleConfirm }"
  //   :secondaryButton="{ text: '取消', dismiss: false, onClick: cancelAction }"
  // />

  props: {
    title: {
      type: String,
      required: true,
    },
    content: {
      type: String,
      required: true,
    },
    icon: {
      type: String,
      default: 'success',
      validator: (value) => ['error', 'info', 'success'].includes(value),
    },
    primaryButton: {
      type: Object,
      default: null, // e.g. { text: 'OK', onClick: function }
    },
    secondaryButton: {
      type: Object,
      default: () => ({
        text: this.$t('lblClose'),
        dismiss: true,
      }),
    },
  },
  data() {
    return {
      modalInstance: null,
    }
  },
  mounted() {
    if (this.$refs.modalRef) {
      this.modalInstance = new Modal(this.$refs.modalRef)
    }
  },
  computed: {
    iconPath() {
      switch (this.icon) {
        case 'success':
          return {
            path: 'M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zm-3.97-3.03a.75.75 0 0 0-1.08.022L7.477 9.417 5.384 7.323a.75.75 0 0 0-1.06 1.06L6.97 11.03a.75.75 0 0 0 1.079-.02l3.992-4.99a.75.75 0 0 0-.01-1.05z',
            fill: 'green',
          }
        case 'info':
          return {
            path: 'M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm.93-9.412-1 4.705c-.07.34.029.533.304.533.194 0 .487-.07.686-.246l-.088.416c-.287.346-.92.598-1.465.598-.703 0-1.002-.422-.808-1.319l.738-3.468c.064-.293.006-.399-.287-.47l-.451-.081.082-.381 2.29-.287zM8 5.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z',
            fill: 'blue',
          }
        case 'error':
        default:
          return {
            path: 'M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z',
            fill: 'red',
          }
      }
    }
  },
  methods: {
    openModal() {
      if (this.modalInstance) {
        this.modalInstance.show()
      } else {
        console.error("Modal instance is not initialized.")
      }
    },
    closeModal() {
      if (this.modalInstance) {
        this.modalInstance.hide()
      }
    },
    handlePrimaryClick() {
      if (this.primaryButton?.onClick) {
        this.primaryButton.onClick()
      } else if (this.modalInstance) {
        this.modalInstance.hide() // 預設關閉 modal
      }
    },
    handleSecondaryClick() {
      if (this.secondaryButton?.onClick) {
        this.secondaryButton.onClick()
      }
    },
  },
};
</script>
