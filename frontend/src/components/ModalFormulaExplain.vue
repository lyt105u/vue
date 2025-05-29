<template>
    <div class="modal fade" ref="modalRef" tabindex="-1" aria-labelledby="modalFormulaExplainLabel" aria-hidden="true">
      <div class="modal-dialog">
        <div class="modal-content">
          <div class="modal-header">
            <h1 class="modal-title fs-5" id="modalFormulaExplainLabel">{{ $t('lblConfusionMatrix') }} & {{ $t('lblMetricsExplanation') }}</h1>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <div class="bd-example-snippet bd-code-snippet">
              <div class="bd-example m-0 border-0">
                <table class="table table-sm table-bordered text-center">
                  <thead>
                    <tr>
                      <th scope="col" colspan="2">{{ $t('lblConfusionMatrix') }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>True Positive (TP)</td>
                      <td>False Negative (FN)</td>
                    </tr>
                    <tr>
                      <td>False Positive (FP)</td>
                      <td>True Negative (TN)</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
  
            <div class="text-center" v-for="formula in formulas" :key="formula.label">
              <div class="formula">
                <div>
                  {{ formula.label }} =
                  <span class="fraction">
                    <span class="numerator">{{ formula.numerator }}</span>
                    <span class="denominator">{{ formula.denominator }}</span>
                  </span>
                </div>
              </div>
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
    data() {
      return {
        modalInstance: null,
        formulas: [
          { label: "Recall", numerator: "TP", denominator: "TP + FN" },
          { label: "Specificity", numerator: "TN", denominator: "TN + FP" },
          { label: "Precision", numerator: "TP", denominator: "TP + FP" },
          { label: "Negative Predictive Value (NPV)", numerator: "TN", denominator: "TN + FN" },
          { label: "F1-score", numerator: "2 × Precision × Recall", denominator: "Precision + Recall" },
          { label: "F2-score", numerator: "5 × Precision × Recall", denominator: "4 × Precision + Recall" },
          { label: "Accuracy", numerator: "TP + TN", denominator: "TP + TN + FP + FN" },
        ],
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
        }
      },
    },
  };
  </script>
  
  <style scoped>
  .formula {
    display: inline-block;
    font-size: 1rem;
  }
  
  .fraction {
    display: inline-block;
    text-align: center;
    vertical-align: middle;
  }
  
  .numerator {
    display: block;
    border-bottom: 1px solid black;
    padding-bottom: 0.2rem;
  }
  
  .denominator {
    display: block;
    padding-top: 0.2rem;
  }
  </style>
  