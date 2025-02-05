<template>
  <div class="about">
    <h1>Predict</h1>
  </div>

  <div class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <form class="row g-3" @submit.prevent="runPredict" style="margin-top: 16px">
    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Trained model</label>
      <div class="col-sm-8">
        <select class="form-select" aria-label="Small select example" v-model="selected.model_path">
          <option v-for="data in modelNames" :key="data" :value="data">{{ data }}</option>
        </select>
      </div>
    </div>

    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Prediction Type</label>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.mode" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1" value="file">
          <label class="form-check-label" for="gridRadios1">
            File Prediction
          </label>
        </div>
      </div>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.mode" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1" value="input">
          <label class="form-check-label" for="gridRadios1">
            Manual Input
          </label>
        </div>
      </div>
    </div>

    <template v-if="selected.mode=='file'">
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label">File Selection</label>
        <div class="col-sm-8">
          <select class="form-select" aria-label="Small select example" v-model="selected.data_path">
            <option v-for="data in xlsxNames" :key="data" :value="data">{{ data }}</option>
          </select>
        </div>
      </div>

      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label">Results Saved as</label>
        <div class="col-sm-8">
          <div class="input-group">
            <input v-model="selected.output_name" class="form-control" type="text">
            <span class="input-group-text">{{ watched.file_extension }}</span>
          </div>
        </div>
      </div>
    </template>

    <template v-if="selected.mode=='input'">
      <div class="row mb-3" v-for="(row, rowIndex) in rows" :key="rowIndex">
        <!-- 第一行顯示，其他行保持空白，排版用 -->
        <label for="inputEmail3" class="col-sm-3 col-form-label">
          {{ rowIndex === 0 ? "Manual Input" : "" }}
        </label>

        <div v-for="(field, fieldIndex) in row" :key="`${rowIndex}-${fieldIndex}`" class="col-sm-2">
          <div class="form-floating">
            <input
              v-model="selected.input_values[rowIndex * 4 + fieldIndex]"
              type="text" 
              class="form-control" 
              :id="`floatingInput-${rowIndex}-${fieldIndex}`" 
              :placeholder="`Field ${rowIndex * 4 + fieldIndex + 1}`" 
            />
            <label :for="`floatingInput-${rowIndex}-${fieldIndex}`">
              {{ rowIndex * 4 + fieldIndex + 1 }}
            </label>
          </div>
        </div>
      </div>
    </template>
    
    <div class="col-12">
      <button v-if="!loading" type="submit" class="btn btn-primary">Predict</button>
      <button v-if="loading" class="btn btn-primary" type="button" disabled>
        <span class="spinner-border spinner-border-sm" aria-hidden="true"></span>
      </button>
    </div>
  </form>

  <div v-if="output" class="bd-example-snippet bd-code-snippet">
    <div class="bd-example m-0 border-0">
      <hr>
    </div>
  </div>

  <div v-if="output" class="about">
    <h3>
      Results
    </h3>
    {{ notificationMsg }}
  </div>

  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
  <ModalNotification ref="modalNotification" title="Training Complete" :content="notificationMsg" />
</template>

<script>
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import axios from 'axios';
import ModalNotification from "@/components/ModalNotification.vue"

export default {
  components: {
    ModalNotification,
  },
  data() {
    return {
      modelNames: '',
      xlsxNames: '',
      selected: {
        model_path: '',
        mode: 'file',
        data_path: '',
        output_name: '',
        input_values: [],
      },
      watched: {
        file_extension: '',
      },
      output: '',
      notificationMsg: '',
      loading: false,
    };
  },
  created() {
    this.fetchData();
  },
  mounted() {},
  computed: {
    rows() {
      const result = [];
      for (let i = 0; i < this.selected.input_values.length; i += 4) {
        result.push(this.selected.input_values.slice(i, i + 4));
      }
      return result;
    },
  },
  watch: {
    "selected.model_path"() {
      this.selected.data_path = ''
      this.selected.output_name = ''
      this.selected.input_name = []
      this.output = ''
      this.getFieldNumber()
    },
    "selected.data_path"() {
        if (this .selected.data_path.endsWith(".csv")) {
          this.watched.file_extension = ".csv"
        } else if (this.selected.data_path.endsWith(".xlsx")) {
          this.watched.file_extension = ".xlsx"
        } else {
          this.watched.file_extension = ""
        }
    },
    "selected.mode"() {
      this.output = ''
    }
  },
  methods: {
    async fetchData() {
      // 拿 trained model names
      try {
        const response = await axios.post('http://127.0.0.1:5000/fetch-data', {
          param: 'model'
        });
        if (response.data.status == "success") {
          this.modelNames = response.data.files
        }
      } catch (error) {
        console.error("fetchData error: " + error)
        this.modelNames = { status: 'fail', error: '無法連接後端服務' };
      }

      // 拿 xlsx names
      try {
        const response = await axios.post('http://127.0.0.1:5000/fetch-data', {
          param: 'data/predict'
        });
        if (response.data.status == "success") {
          this.xlsxNames = response.data.files
        }
      } catch (error) {
        console.error("fetchData error: " + error)
        this.xlsxNames = { status: 'fail', error: '無法連接後端服務' };
      }
    },

    async getFieldNumber() {
      if (this.selected.model_path) {
        try {
          const response = await axios.post('http://127.0.0.1:5000/get-fieldNumber', {
            param: this.selected.model_path
          });
          if (response.data.status == "success") {
            this.selected.input_values = Array(response.data.field_count).fill("");
          }
        } catch (error) {
          console.error("fetchData error: " + error)
          this.modelNames = { status: 'fail', error: '無法連接後端服務' };
        }
      }
    },

    async runPredict() {
      this.loading = true
      this.output = null

      try {
        const response = await axios.post('http://127.0.0.1:5000/run-predict', this.selected)
        this.output = response.data
        this.openModalNotification()
      } catch (error) {
        console.error('Error:', error);
        this.output = {
          status: 'error',
          message: error.response?.data?.message || error.message,
        };
      }

      this.loading = false
    },

    openModalNotification() {
      if (this.$refs.modalNotification) {
        if (this.selected.mode == 'file') {
          this.notificationMsg = this.output.message
        } else if (this.selected.mode == 'input') {
          this.notificationMsg = this.output.message[0]
        }
        this.$refs.modalNotification.openModal();
      } else {
        console.error("ModalNotification component not found.");
      }
    },
  },
};
</script>
