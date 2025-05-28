<template>
  <!-- <NavBar /> -->
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
  <!-- <div>
    <SideBar />
  </div> -->
  <!-- <nav>
    <router-link to="/">Home</router-link> |
    <router-link to="/about">About</router-link>
  </nav> -->
  <!-- <router-view/> -->
</template>

<script>
// import NavBar from './components/NavBar.vue'
// import SideBar from './components/SideBar.vue'
import { SidebarMenu } from 'vue-sidebar-menu'
import 'vue-sidebar-menu/dist/vue-sidebar-menu.css'

// import { h } from 'vue'
// import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'

// const separator = h('hr', {
//   style: {
//     borderColor: 'rgba(0, 0, 0, 0.1)',
//     margin: '20px',
//   },
// })

// const faIcon = (props) => {
//   return {
//     element: h('div', [h(FontAwesomeIcon, { size: 'lg', ...props })]),
//   }
// }

export default {
  name: "App",
  components: {
    // NavBar,
    // SideBar,
    SidebarMenu,
  },
  computed: {
    menu() {
      return [
        {
          header: this.$t('lblMlas')+'_v0.6',
          hiddenOnCollapse: true
        },
        {
          href: '/',
          title: this.$t('lblHome'),
          icon: 'fa fa-home'
        },
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
          href: '/predict',
          title: this.$t('lblPredict'),
          icon: 'fa fa-line-chart'
        }
      ]
    }
  },
  data() {
    return {
      collapsed: false,
      selectedTheme: 'white-theme',
      isOnMobile: false,
    }
  },

  mounted() {
    this.onResize()
    window.addEventListener('resize', this.onResize)
  },
  methods: {
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
</style>
