<template>
  <v-app>
    <v-app-bar app>

      <v-app-bar-nav-icon @click="drawer = !drawer"></v-app-bar-nav-icon>

      <v-toolbar-title  class="text-center flex-grow-1">Trail Sort</v-toolbar-title>

      <router-link to="/user">
        <v-btn icon color="white">
          <v-icon>mdi-account-circle</v-icon>
        </v-btn>
      </router-link>

    </v-app-bar>

    <v-navigation-drawer v-model="drawer" app color="background">
      <v-list>
        <v-list-item v-for="(item, index) in items" :key="index">

          <router-link :to="item.route" class="router-link" exact>

            <v-btn icon>
                <v-icon >{{ item.icon }}</v-icon>
            </v-btn>
            
            <v-list-item-title>{{ item.title }}</v-list-item-title>
            
         </router-link>

        </v-list-item>
      </v-list>
    </v-navigation-drawer>

    <v-main>
      <v-container>
        <slot></slot>
      </v-container>
    </v-main>
  </v-app>
</template>



<script setup>

  import { ref } from 'vue'

  const drawer = ref(false)
  const items = [
    { title: 'Home', icon: 'mdi-home', route: '/' },
    { title: 'About', icon: 'mdi-information', route: '/about' },
    { title: 'Sort', icon: 'mdi-file-document-arrow-right ', route: '/sort' },
    { title: 'Blog', icon: 'mdi-post-outline', route: '/blog' }
  ]
  
</script>


  

  
<style scoped>
  .v-application {
    font-family: 'Roboto', sans-serif;
  }

  /* Make sure the router-link doesn't have underline and text color is white */
  .router-link {
    text-decoration: none;
    color: white; 
    display: flex;
    align-items: center;
  }

  .router-link-exact-active{
    color: rgb(var(--v-theme-orangeDarken1)); /* https://stackoverflow.com/questions/48280990/using-custom-theming-in-vuetify-and-pass-color-variables-to-components */
  }

  .v-btn--variant-elevated, 
  .v-btn--variant-flat {
    background: rgb(var(--v-theme-background)) !important;
  }

  .v-list-item--active .v-btn--variant-elevated,
  .v-list-item--active .v-btn--variant-flat {
    color: rgb(var(--v-theme-orangeDarken1));
  }
    

</style>
  