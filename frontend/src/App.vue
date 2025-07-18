<template>
  <sidebar-menu
    :key="$i18n.locale" 
    v-model:collapsed="collapsed"
    :menu="menu"
    :theme="selectedTheme"
    :show-one-child="true"
    @update:collapsed="onToggleCollapse"
    @item-click="onItemClick"
  />
  <div v-if="isOnMobile && !collapsed" class="sidebar-overlay" @click="collapsed = true" />
  <div id="demo" :class="[{ collapsed: collapsed }, { onmobile: isOnMobile }]">
    <div class="demo">
      <div class="container">
        <router-view />
      </div>
    </div>
  </div>
</template>

<script>
import { SidebarMenu } from 'vue-sidebar-menu'
import 'vue-sidebar-menu/dist/vue-sidebar-menu.css'

import { h } from 'vue'

export default {
  name: "App",
  components: {
    SidebarMenu,
  },
  data() {
    return {
      collapsed: false,
      selectedTheme: 'white-theme',
      isOnMobile: false,
      menu: [],
    }
  },

  mounted() {
    this.onResize()
    window.addEventListener('resize', this.onResize)
  },

  created() {
    this.buildMenu()
    const savedLocale = sessionStorage.getItem('locale')
    if (savedLocale) {
      this.$i18n.locale = savedLocale
    }
  },

  beforeUnmount() {
    sessionStorage.removeItem('username')
    sessionStorage.removeItem('locale')
  },
  
  watch: {
    '$i18n.locale'() {
      this.buildMenu()
    }
  },

  methods: {
    buildMenu() {
      const separator = h('hr', {
        style: {
          borderColor: 'gray',
          margin: '20px',
        },
      })

      this.menu = [
        {
          header: this.$t('lblUserName') + ": " + (sessionStorage.getItem('username') || ''),
          hiddenOnCollapse: true
        },
        {
          href: '/',
          title: this.$t('lblHome'),
          icon: 'fa fa-home'
        },
        {
          href: '/settings',
          title: this.$t('lblSettings'),
          icon: 'fa fa-cog'
        },
        { component: separator },
        {
          href: '/upload',
          title: this.$t('lblUpload'),
          icon: 'fa fa-upload'
        },
        {
          href: '/train',
          title: this.$t('lblTrain'),
          icon: 'fa fa-cogs'
        },
        {
          title: this.$t('lblPredict'),
          icon: 'fa fa-line-chart',
          child: [
            {
              href: '/evaluate',
              title: this.$t('lblModelEvaluation'),
              icon: 'fa fa-book',
            },
            {
              href: '/predict',
              title: this.$t('lblClinicalPrediction'),
              icon: 'fa fa-stethoscope',
            },
          ]
        },
        { component: separator },
        {
          href: '/release',
          title: this.$t('lblReleaseNote'),
          icon: 'fa fa-sticky-note-o'
        }
      ]
    },

    onToggleCollapse(collapsed) {
      console.log(collapsed)
      console.log('onToggleCollapse')
    },
    onItemClick(event, item) {
      console.log('onItemClick')
      console.log(event)
      console.log(item)
    },
    onResize() {
      if (window.innerWidth <= 767) {
        this.isOnMobile = true
        this.collapsed = true
      } else {
        this.isOnMobile = false
        this.collapsed = false
      }
    }
  }
}
</script>


<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  /* text-align: center; */
  color: #2c3e50;
}

nav {
  padding: 30px;
}

nav a {
  font-weight: bold;
  color: #2c3e50;
}

nav a.router-link-exact-active {
  color: #42b983;
}

@import url('https://fonts.googleapis.com/css?family=Source+Sans+Pro:400,600');

body,
html {
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Source Sans Pro', sans-serif;
  font-size: 18px;
  background-color: #f2f4f7;
  color: #262626;
}

#demo {
  padding-left: 290px;
  transition: 0.3s ease;
}
#demo.collapsed {
  padding-left: 65px;
}
#demo.onmobile {
  padding-left: 65px;
}

.sidebar-overlay {
  position: fixed;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
  background-color: #000;
  opacity: 0.5;
  z-index: 900;
}

.demo {
  padding: 50px;
}

.container {
  max-width: 900px;
}

.v-sidebar-menu .vsm--header {
  text-transform: none !important;
}
</style>
