import { createRouter, createWebHistory } from 'vue-router'

import HomePage from '../views/HomePage.vue'
import AboutPage from '../views/AboutPage.vue'
import SortPage from '../views/SortPage.vue'
import UserPage from '../views/UserPage.vue'
import BlogPage from '../views/BlogPage.vue'
import BlogEditor from '../views/BlogEditor.vue'

const routes = [
    { path: '/', component: HomePage, name: 'homePage' },
    { path: '/about', component: AboutPage, name: 'aboutPage' },
    { path: '/sort', component: SortPage, name: 'sortPage' },
    { path: '/user', component: UserPage, name: 'userPage' },
    { path: '/blog', component: BlogPage, name: 'blog' },
    { path: '/blog/editor', component: BlogEditor, name: 'blogEditor' },
    
]

const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes,
})

export default router
