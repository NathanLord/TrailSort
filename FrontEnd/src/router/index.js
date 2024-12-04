
import { createRouter, createWebHistory } from 'vue-router'
import HomePage from '../views/HomePage.vue'
import AboutPage from '../views/AboutPage.vue'
import SortPage from '../views/SortPage.vue'
import UserPage from '../views/UserPage.vue'

const routes = [
    { path: '/', component: HomePage, name: 'homePage' },
    { path: '/about', component: AboutPage, name: 'aboutPage' },
    { path: '/sort', component: SortPage, name: 'sortPage' },
    { path: '/user', component: UserPage, name: 'userPage' },
]

const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes,
})

export default router
