<template>
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
    <!-- 第一筆 -->
    <div class="accordion-item">
      <h2 class="accordion-header" @click="toggleCollapse('collapseOne')">
        <button class="accordion-button collapsed" type="button">
          v0.8
        </button>
      </h2>
      <div class="accordion-collapse collapse" ref="collapseOne">
        <div class="accordion-body">
          Implement the download result function.
        </div>
      </div>
    </div>

    <!-- 第二筆 -->
    <!-- <div class="accordion-item">
      <h2 class="accordion-header" @click="toggleCollapse('collapseTwo')">
        <button class="accordion-button collapsed" type="button">
          v0.7
        </button>
      </h2>
      <div class="accordion-collapse collapse" ref="collapseTwo">
        <div class="accordion-body">
          Added model explanation modal for SHAP and LIME
        </div>
      </div>
    </div> -->
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
