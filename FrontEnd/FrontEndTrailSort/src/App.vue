<script setup>
import menuComponent from './components/MenuComponent.vue';
import { RouterView } from 'vue-router';
</script>

<template>
  <v-app>
    <!-- Other components or content -->
    <menuComponent>
      <router-view></router-view>
    </menuComponent>
  </v-app>
</template>

<style scoped>


</style>
