<template>

    <!-- https://vuetifyjs.com/en/components/selects/#usage -->
    <v-select
        v-model="selectedSource"
        :items="sources"
        item-title="text"
        item-value="value"
        label="Select Model"
        class="mb-4"
        outlined
    ></v-select>

</template>

<script setup>

    import { ref, computed } from 'vue';


    const selectedSource = ref('/iNaturalRawClassification_TF1_display.html');
    const sources = [
        { text: 'Basic Model', value: '/iNaturalRawClassification_TF1_display.html', backEnd: 'trailSortTF1.keras' },
        { text: 'Model with more input layers', value: '/iNaturalRawClassification_TF2MorePixels_display.html', backEnd: 'trailSortTF2MorePixels.keras' },
        { text: 'Model trained on larger dataset', value: '/iNaturalRawClassification_TF3Large_display.html', backEnd: 'trailSortTF3Large.keras' },
        
    ];

    // For defineExpose https://medium.com/@louis.young0420/leveraging-defineexpose-in-vue-3-empowering-component-communication-f4991c1bc997
    // Expose the data from this child component to any parent component
    defineExpose({
        selectedSource,
        selectedBackEnd: computed(() => {
            const selected = sources.find(source => source.value === selectedSource.value);
            return selected ? selected.backEnd : null;
        }),
    });


</script>

<style scoped>

</style>
