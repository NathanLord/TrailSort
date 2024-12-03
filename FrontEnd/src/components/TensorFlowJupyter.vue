<template>
    <div>

        <v-container>
            <!-- Select Dropdown -->
            <v-select
                v-model="selectedSource"
                :items="sources"
                item-title="text" 
                item-value="value" 
                label="Select Model"
                class="mb-4"
                outlined
            ></v-select>

            <!-- Iframe -->
            <iframe
                :src="selectedSource"
                width="100%"
                height="800px"
                style="border: none"
            ></iframe>
        </v-container>

    </div>
</template>

<script setup>

    import { ref } from 'vue';

    const selectedSource = ref('/trail_sort_TF2_display.html');

    const sources = [
        { text: 'Basic Model', value: '/trail_sort_TF2_display.html' },
        { text: 'Larger Model', value: '/trail_sort_TF3Larger_display.html' },
    ];

</script>

<style scoped>

</style>