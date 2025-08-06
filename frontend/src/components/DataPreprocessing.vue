<template>
  <div>
    <!-- 選擇欄位 -->
    <div class="mb-3">
      <select v-model="selectedColumn" class="form-select" :disabled="loading">
        <option value="" disabled>{{ $t('lblSelectColumn') }}</option>
        <option v-for="col in columns" :key="col" :value="col">
          {{ col }}
          <span v-if="missingColumns.includes(col)">{{ $t('lblContainsMissingValue') }}</span>
        </option>
      </select>
    </div>

    <!-- 規則設定 -->
    <div v-if="selectedColumn" class="card p-3 mb-3">
      <!-- 預期條件類型 -->
      <div class="mb-2">
        <label class="form-label">{{ $t('lblExpectationType') }}</label>
        <select v-model="expectType" class="form-select" :disabled="loading">
          <option value="condition">{{ $t('lblConditionValidation') }}</option>
          <option value="not_missing">{{ $t('lblHasValue') }}</option>
        </select>
      </div>

      <!-- 條件運算與值（僅限 condition） -->
      <div v-if="expectType === 'condition'" class="mb-2">
        <label class="form-label">{{ $t('lblCondition') }}</label>
        <div class="input-group">
          <select v-model="expectCondition" class="form-select" style="max-width: 120px;" :disabled="loading">
            <option value=">=">&ge;</option>
            <option value="<=">&le;</option>
            <option value="!=">&ne;</option>
          </select>
          <input v-model="expectValue" class="form-control" :placeholder="$t('msgCustomValueEx')" :disabled="loading" />
        </div>
        <div v-if="errors.expectValue" class="text-danger small">{{ errors.expectValue }}</div>
      </div>

      <!-- fallback 設定 -->
      <div class="mb-2">
        <label class="form-label">{{ $t('lblFallback') }}</label>
        <select v-model="fallbackType" class="form-select mb-2" :disabled="loading">
          <option v-for="opt in fallbackOptions" :key="opt.value" :value="opt.value">
            {{ $t(opt.label) }}
          </option>
        </select>
        <input
          v-if="fallbackType === 'custom'"
          v-model="fallbackValue"
          class="form-control"
          :placeholder="$t('msgCustomValueEx')"
          :disabled="loading"
        />
        <div v-if="errors.fallbackValue" class="text-danger small">{{ errors.fallbackValue }}</div>
      </div>

      <!-- 按鈕 -->
      <div class="d-flex justify-content-end gap-2 mt-2">
        <button @click="resetInputs" class="btn btn-outline-secondary" type="button" :disabled="loading">
          <i class="fa fa-times me-1"></i> {{ $t('lblCancel') }}
        </button>
        <button @click="addRule" class="btn btn-outline-primary" type="button" :disabled="loading">
          <i class="fa fa-plus me-1"></i> {{ $t('lblAddRule') }}
        </button>
      </div>
    </div>

    <!-- 規則表格 -->
    <div v-if="rules.length > 0">
      <h5>{{ $t('lblConfigValRules') }}</h5>
      <div class="table-responsive card card-body">
        <table class="table caption">
          <caption>{{ $t('msgExpectFallback') }}</caption>
          <thead>
            <tr>
              <th>{{ $t('lblFeature') }}</th>
              <th>{{ $t('lblExpectation') }}</th>
              <th>{{ $t('lblFallback') }}</th>
              <th>{{ $t('lblActions') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(rule, index) in sortedRules" :key="index">
              <td>{{ rule.column }}</td>
              <td>
                <template v-if="rule.expect_type === 'condition'">
                  {{ rule.expect_condition }} {{ rule.expect_value }}
                </template>
                <template v-else>
                  {{ $t('lblHasValue') }}
                </template>
              </td>
              <td>
                <template v-if="rule.fallback_type === 'custom'">
                  {{ rule.fallback_value }}
                </template>
                <template v-else>
                  {{ $t('lblFill' + capitalize(rule.fallback_type)) }}
                </template>
              </td>
              <td>
                <button class="btn btn-sm btn-outline-danger" @click="removeRule(index)" type="button" :disabled="loading">
                  <i class="fa fa-trash me-1"></i> {{ $t('lblDelete') }}
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- 未處理缺值提示 -->
    <div v-if="unhandledMissingColumns.length > 0" class="alert alert-warning mt-3">
      {{ $t('msgMissingNotHandled') }} <strong>{{ unhandledMissingColumns.join(', ') }}</strong>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DataPreprocessing',
  props: {
    columns: Array,
    missingColumns: Array,
    loading: {
      type: Boolean,
      default: false
    },
  },
  data() {
    return {
      selectedColumn: '',
      expectType: 'condition',
      expectCondition: '>=',
      expectValue: '',
      fallbackType: '',
      fallbackValue: '',
      rules: [],
      errors: {},
      fallbackOptions: [
        { value: 'min', label: 'lblFillMin' },
        { value: 'max', label: 'lblFillMax' },
        { value: 'mean', label: 'lblFillMean' },
        { value: 'median', label: 'lblFillMedian' },
        { value: 'mode', label: 'lblFillMode' },
        { value: 'custom', label: 'lblFillCustom' },
        { value: 'drop', label: 'lblFillDrop' },
        { value: 'skip', label: 'lblFillSkip' },
      ]
    }
  },
  computed: {
    sortedRules() {
      return this.rules.slice().sort((a, b) => {
        return this.columns.indexOf(a.column) - this.columns.indexOf(b.column)
      })
    },
    unhandledMissingColumns() {
      const handled = this.rules
        .filter(r => r.expect_type === 'not_missing')
        .map(r => r.column)
      return this.missingColumns.filter(c => !handled.includes(c))
    }
  },
  created() {
    this.$emit('update:unhandled', this.unhandledMissingColumns)
  },
  methods: {
    resetInputs() {
      this.selectedColumn = ''
      this.expectType = 'condition'
      this.expectCondition = '>='
      this.expectValue = ''
      this.fallbackType = ''
      this.fallbackValue = ''
      this.errors = {}
    },

    validateForm() {
      this.errors = {}
      let isValid = true
      if (!this.selectedColumn) isValid = false
      if (this.expectType === 'condition' && !this.expectValue) {
        this.errors.expectValue = this.$t('msgValRequired')
        isValid = false
      }
      if (!this.fallbackType) {
        this.errors.fallbackValue = this.$t('msgValRequired')
        isValid = false
      }
      if (this.fallbackType === 'custom' && !this.fallbackValue) {
        this.errors.fallbackValue = this.$t('msgValRequired')
        isValid = false
      }
      if (this.expectType === 'condition') {
        const newCond = this.expectCondition
        const newVal = parseFloat(this.expectValue)
        if (!isNaN(newVal)) {
          const existing = this.rules.filter(
            r => r.column === this.selectedColumn && r.expect_type === 'condition'
          )
          for (const r of existing) {
            const cond = r.expect_condition
            const val = parseFloat(r.expect_value)
            if (cond === '<=' && newCond === '>=' && newVal > val) {
              this.errors.expectValue = this.$t('msgValGreaterThanMax', { max: val })
              isValid = false
              break
            }
            if (cond === '>=' && newCond === '<=' && newVal < val) {
              this.errors.expectValue = this.$t('msgValLessThanMin', { min: val })
              isValid = false
              break
            }
          }
        }
      }
      return isValid
    },

    addRule() {
      if (!this.validateForm()) {
        return
      }
      const rule = {
        column: this.selectedColumn,
        expect_type: this.expectType,
        fallback_type: this.fallbackType,
      }
      if (this.expectType === 'condition') {
        rule.expect_condition = this.expectCondition
        rule.expect_value = this.expectValue
      }
      if (this.fallbackType === 'custom') {
        rule.fallback_value = isNaN(Number(this.fallbackValue))
          ? this.fallbackValue
          : Number(this.fallbackValue)
      }
      // 移除相同 column + expect_type + expect_condition 的 rule
      this.rules = this.rules.filter(r => {
        if (r.column !== rule.column) return true
        if (rule.expect_type === 'condition' && r.expect_type === 'condition') {
          return r.expect_condition !== rule.expect_condition
        }
        if (rule.expect_type === 'not_missing' && r.expect_type === 'not_missing') {
          return false
        }
        return true
      })
      this.rules.push(rule)
      // emit 更新給父層
      this.$emit('update:rules', this.rules)
      this.$emit('update:unhandled', this.unhandledMissingColumns)
      this.resetInputs()
    },

    removeRule(index) {
      this.rules.splice(index, 1)
      // emit 更新給父層
      this.$emit('update:rules', this.rules)
      this.$emit('update:unhandled', this.unhandledMissingColumns)
    },
    
    capitalize(str) {
      return str.charAt(0).toUpperCase() + str.slice(1)
    },
    
    resetExceptSkip() {
      this.rules = this.rules.filter(
        r => r.expect_type === 'not_missing' && r.fallback_type === 'skip'
      )
      this.$emit('update:rules', this.rules)
      this.$emit('update:unhandled', this.unhandledMissingColumns)
    },
  }
}
</script>
