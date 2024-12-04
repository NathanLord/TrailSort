<template>
    <div>
        <v-container>
            
            <ModelSelect ref="modelSelect" />

            <iframe
                :src="modelSelect?.selectedSource"
                width="100%"
                height="800px"
                style="border: none"
            ></iframe>
            
        </v-container>
    </div>
</template>

<script setup>

    import { ref } from 'vue';
    import ModelSelect from './ModelSelect.vue';

    const modelSelect = ref(null);

    
</script>

<style scoped>

</style>
