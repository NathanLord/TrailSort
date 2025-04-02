<template>
    <div>
        
        <v-container class="fill-height d-flex justify-center align-center">
            <v-row justify="center">
                <v-col cols="12" md="8" lg="6">

                    <v-btn class="mb-4" color="primary" @click="navigateToEditor">Publish Blog</v-btn>

                    <BlogPosts> </BlogPosts>
                
                </v-col>
            </v-row>
        </v-container>
        
    </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import BlogPosts from '../components/BlogPosts.vue';

const router = useRouter()

const navigateToEditor = () => {
    const token = localStorage.getItem('token') // Check if token exists
    if (token) {
        router.push({ name: 'blogEditor' })
    } else {
        alert('Access Denied! You must be logged in.') // Handle unauthorized access
    }
}
</script>