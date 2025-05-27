import "bootstrap/dist/css/bootstrap.css"
import { createApp } from 'vue'
import App from './App.vue'
import VueSidebarMenu from 'vue-sidebar-menu'
import 'vue-sidebar-menu/dist/vue-sidebar-menu.css'
import i18n from './i18n'

createApp(App).use(router).use(VueSidebarMenu).use(i18n).mount('#app')

import "bootstrap/dist/js/bootstrap.js"
import router from './router'
