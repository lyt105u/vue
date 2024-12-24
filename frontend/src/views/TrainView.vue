<template>
  <div class="about">
    <h1>Train</h1>
  </div>
  <form class="row g-3" @submit.prevent="runTrain">
    <div class="col-md-6">
      <label for="inputEmail4" class="form-label">Model</label>
      <select class="form-select" aria-label="Small select example" v-model="arg1">
        <option value="catboost">Cat Boost</option>
        <option value="lightgbm">lightGBM</option>
        <option value="random_forest">Random Forest</option>
        <option value="xgb">XGB</option>
      </select>
    </div>
    <div class="col-md-6">
      <label for="inputPassword4" class="form-label">Data</label>
      <select class="form-select" aria-label="Small select example" v-model="arg2">
        <option value="cg_train_data.csv">長庚訓練</option>
      </select>
    </div>
    <div class="col-12">
      <button v-if="!loading" type="submit" class="btn btn-primary">Train</button>
      <button v-if="loading" class="btn btn-primary" type="button" disabled>
        <span class="spinner-border spinner-border-sm" aria-hidden="true"></span>
      </button>
    </div>
  </form>
  <div v-if="output">
    <h3>Output:</h3>
    <p>{{ output }}</p>
  </div>

  <!-- modal -->
  <div class="modal fade" id="exampleModal" tabindex="-1" aria-labelledby="exampleModalLabel" aria-hidden="true">
    <div class="modal-dialog">
      <div class="modal-content">
        <div class="modal-header">
          <h1 class="modal-title fs-5" id="exampleModalLabel">Modal title</h1>
          <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div class="bd-example-snippet bd-code-snippet">
            <div class="bd-example m-0 border-0">
              Model trained successfully!
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
import axios from 'axios';
import { Modal } from 'bootstrap';
export default {
  data() {
    return {
      arg1: '',
      arg2: '',
      output: '',
      loading: false,
    };
  },
  methods: {
    async runTrain() {
      this.loading = true
      this.output = null
      // try {
      //   const response = await fetch('http://127.0.0.1:5000/run-train', {
      //     method: 'POST',
      //     headers: { 'Content-Type': 'application/json' },
      //     body: JSON.stringify({ arg1: this.arg1, arg2: this.arg2 }),
      //   });
      //   // const result = await response.json();
      //   // if (result.status === 'success') {
      //   //   this.output = result.output;
      //   // } else {
      //   //   this.output = `Error: ${result.output}`;
      //   // }
      //   if (!response.ok) {
      //     throw new Error(`HTTP error! status: ${response.status}`);
      //   }
      //   this.output = await response.json();
      //   console.log(response)
      // } catch (error) {
      //   console.error('Error:', error);
      //   this.output = null;
      // }
      try {
        const response = await axios.post('http://127.0.0.1:5000/run-train', {
          arg1: this.arg1,
          arg2: this.arg2,
        });
        this.output = response.data;
      } catch (error) {
        console.error(error);
        this.output = { status: 'fail', error: '無法連接後端服務' };
      }
      this.loading = false
      this.openModal()
    },
    openModal() {
      const modalElement = document.getElementById('exampleModal');
      if (modalElement) {
        const modalInstance = new Modal(modalElement);
        modalInstance.show();
      } else {
        console.error('Modal element not found');
      }
    },
  },
};
</script>
