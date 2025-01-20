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
        <select class="form-select" aria-label="Small select example" v-model="selected.model">
          <option v-for="data in modelNames" :key="data" :value="data">{{ data }}</option>
        </select>
      </div>
    </div>

    <div class="row mb-3">
      <label for="inputEmail3" class="col-sm-3 col-form-label">Prediction Type</label>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.method" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1" value="file">
          <label class="form-check-label" for="gridRadios1">
            File Prediction
          </label>
        </div>
      </div>
      <div class="col-sm-4">
        <div class="form-check">
          <input v-model="selected.method" class="form-check-input" type="radio" name="gridRadios" id="gridRadios1" value="input">
          <label class="form-check-label" for="gridRadios1">
            Manual Input
          </label>
        </div>
      </div>
    </div>

    <template v-if="selected.method=='file'">
      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label">File Selection</label>
        <div class="col-sm-8">
          <select class="form-select" aria-label="Small select example" v-model="selected.file">
            <option v-for="data in xlsxNames" :key="data" :value="data">{{ data }}</option>
          </select>
        </div>
      </div>

      <div class="row mb-3">
        <label for="inputEmail3" class="col-sm-3 col-form-label">Results Saved as</label>
        <div class="col-sm-8">
          <div class="input-group">
            <input v-model="selected.model_name" class="form-control" type="text">
            <span class="input-group-text">{{ watched.file_extension }}</span>
          </div>
        </div>
      </div>
    </template>

    <template v-if="selected.method=='input'">
      <div class="row mb-3" v-for="(row, rowIndex) in rows" :key="rowIndex">
        <!-- 第一行顯示，其他行保持空白，排版用 -->
        <label for="inputEmail3" class="col-sm-3 col-form-label">
          {{ rowIndex === 0 ? "Manual Input" : "" }}
        </label>

        <div v-for="(field, fieldIndex) in row" :key="`${rowIndex}-${fieldIndex}`" class="col-sm-2">
          <div class="form-floating">
            <input
              v-model="fields[rowIndex * 4 + fieldIndex]"
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
  </div>

  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

</template>

<script>
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import axios from 'axios';

export default {
  components: {},
  data() {
    return {
      modelNames: '',
      xlsxNames: '',
      selected: {
        model: '',
        method: 'file',
        file: '',
      },
      watched: {
        file_extension: '',
      },
      fields: [],
      output: '',
      imageData: null,
      loading: false,
    };
  },
  created() {
    this.fetchData();
  },
  mounted() {
    // Initialize Tooltip
    // const tooltipTrigger = this.$refs.tooltipTrigger;
    // new bootstrap.Tooltip(tooltipTrigger);
  },
  computed: {
    rows() {
      const result = [];
      for (let i = 0; i < this.fields.length; i += 4) {
        result.push(this.fields.slice(i, i + 4));
      }
      return result;
    },
  },
  watch: {
    "selected.model"() {
      this.getFieldNumber()
    },
    "selected.file"() {
        if (this .selected.file.endsWith(".csv")) {
          this.watched.file_extension = ".csv"
        } else if (this.selected.file.endsWith(".xlsx")) {
          this.watched.file_extension = ".xlsx"
        } else {
          this.watched.file_extension = ""
        }
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
      if (this.selected.model) {
        try {
          const response = await axios.post('http://127.0.0.1:5000/get-fieldNumber', {
            param: this.selected.model
          });
          if (response.data.status == "success") {
            this.fields = Array(response.data.field_count).fill("");
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
        const response = await fetch('http://127.0.0.1:5000/run-predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ arg1: this.selected.model, arg2: this.selected.file }),
        });
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        this.output = await response.json();

        this.imageData = `data:image/png;base64,${this.output.roc}`;

      } catch (error) {
        console.error('Error:', error);
        this.output = null;
      }
      this.loading = false;
    },
  },
};
</script>
