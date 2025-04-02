import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../stores/auth';

import HomePage from '../views/HomePage.vue'
import AboutPage from '../views/AboutPage.vue'
import SortPage from '../views/SortPage.vue'
import UserPage from '../views/UserPage.vue'
import BlogPage from '../views/BlogPage.vue'
import BlogPost from '../views/BlogPost.vue';  // Page to show the full content of a single blog post
import BlogEditor from '../views/BlogEditor.vue'

const routes = [
    { path: '/', component: HomePage, name: 'homePage' },
    { path: '/about', component: AboutPage, name: 'aboutPage' },
    { path: '/sort', component: SortPage, name: 'sortPage' },
    { path: '/user', component: UserPage, name: 'userPage' },
    { path: '/blog', component: BlogPage, name: 'blog' },
    { path: '/blog/:id', component: BlogPost, name: 'blog-post' }, // This is the dynamic route
    { path: '/blog/editor', component: BlogEditor, name: 'blogEditor',  meta: { requiresAuth: true } },
    
]

const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes,
})



router.beforeEach((to, from, next) => {
    const authStore = useAuthStore()
    if (to.meta.requiresAuth) {
        if (!authStore.isAuthenticated) {
            alert('Access Denied! Please log in.')
            return next('/') // Redirect unauthorized users
        }
    }
    next()
})

export default router
