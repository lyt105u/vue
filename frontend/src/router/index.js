import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView
  },
  {
    path: '/upload',
    name: 'upload',
    component: () => import('../views/UploadView.vue')
  },
  {
    path: '/train',
    name: 'train',
    component: () => import('../views/TrainView.vue')
  },
  {
    path: '/evaluate',
    name: 'evaluate',
    component: () => import('../views/EvaluateView.vue')
  },
  {
    path: '/predict',
    name: 'predict',
    // route level code-splitting
    // this generates a separate chunk (about.[hash].js) for this route
    // which is lazy-loaded when the route is visited.
    component: () => import(/* webpackChunkName: "about" */ '../views/PredictView.vue')
  },
  {
    path: '/release',
    name: 'release',
    component: () => import('../views/ReleaseView.vue')
  },
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
