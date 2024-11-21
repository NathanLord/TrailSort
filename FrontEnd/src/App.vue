<script setup>
  import menuComponent from './components/MenuComponent.vue';
  import { RouterView } from 'vue-router';
</script>

<template>
  <v-app>
    <menuComponent>
      <router-view v-slot="{ Component }">
        <transition name="slide" mode="out-in"> <!-- https://www.youtube.com/watch?v=pG5pwDGc3D4-->
          <component :is="Component"/>
        </transition>
      </router-view>
    </menuComponent>
  </v-app>
</template>

<style scoped>

  .fade-enter-from, .fade-leave-to{
    opacity: 0;
  }

  .fade-enter-active, .fade-leave-active{
    transition: opacity 0.3s ease-out;
  }

  .slide-enter-from{
    opacity: 0;
    transform: translateY(500px);
  }

  .slide-leave-to{
    opacity: 0;
    transform: translateY(-500px);
  }

  .slide-enter-active, .slide-leave-active{
    transition: 0.3s ease-out;
  }


</style>
