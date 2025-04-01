<template>
    <div>
        <v-btn  color="primary" @click="navigateToEditor">Publish Blog</v-btn>
    </div>
</template>

<script setup>
import { useRouter } from 'vue-router'

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