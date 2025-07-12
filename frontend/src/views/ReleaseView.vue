<template>
  <!-- Release Note -->
  <div class="about">
    <h1>{{ $t('lblReleaseNote') }}</h1>
  </div>

  <!-- hr -->
  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <div class="accordion" id="accordionExample">
    <!-- v0.9 -->
    <div class="accordion-item">
      <h2 class="accordion-header" @click="toggleCollapse('collapse_0_9')">
        <button class="accordion-button collapsed" type="button">
          v0.9
        </button>
      </h2>
      <div class="accordion-collapse collapse" ref="collapse_0_9">
        <div class="accordion-body">
          <div>
            <span class="badge rounded-pill text-bg-success">Feature</span>
            Settings page
          </div>
          <div>
            <span class="badge rounded-pill text-bg-success">Feature</span>
            Login
          </div>
          <div>
            <span class="badge rounded-pill text-bg-danger">Fix</span>
            Show feature names correctly in SHAP and LIME images.
          </div>
          <div>
            <span class="badge rounded-pill text-bg-info">Enhance</span>
            Check uniqueness of training data's true label column.
          </div>
        </div>
      </div>
    </div>

    <!-- v0.8 -->
    <div class="accordion-item">
      <h2 class="accordion-header" @click="toggleCollapse('collapse_0_8')">
        <button class="accordion-button collapsed" type="button">
          v0.8
        </button>
      </h2>
      <div class="accordion-collapse collapse" ref="collapse_0_8">
        <div class="accordion-body">
          <span class="badge rounded-pill text-bg-success">Feature</span>
          Download result report.
        </div>
      </div>
    </div>
  </div>

  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
</template>


<script>
import { Collapse } from 'bootstrap'

export default {
  name: 'PreviewAccordion',
  methods: {
    toggleCollapse(refName) {
      const collapseEl = this.$refs[refName]
      const buttonEl = collapseEl.previousElementSibling.querySelector('.accordion-button')

      let instance = Collapse.getInstance(collapseEl)
      if (!instance) {
        instance = new Collapse(collapseEl, { toggle: false })
      }

      instance.toggle()
      buttonEl.classList.toggle('collapsed')
    }
  }
}
</script>
