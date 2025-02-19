<template>
    <div>
        <v-btn  color="primary" @click="navigateToEditor">Publish Blog</v-btn>
    </div>
</template>

<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()

const navigateToEditor = () => {
    router.push({ name: 'blogEditor' })
}
</script>