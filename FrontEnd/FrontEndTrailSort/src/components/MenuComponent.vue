<script setup>

  import { ref } from 'vue'

  const drawer = ref(false)
  const items = [
    { title: 'Home', icon: 'mdi-home', route: '/' },
    { title: 'About', icon: 'mdi-information', route: '/about' },
    { title: 'Sort', icon: 'mdi-file-document-arrow-right ', route: '/sort' },
  ]
</script>

<template>
    <v-app>
      <v-app-bar app>
        <v-app-bar-nav-icon @click="drawer = !drawer"></v-app-bar-nav-icon>
  
        <v-toolbar-title  class="text-center flex-grow-1">Trail Sort</v-toolbar-title>
        
        <!-- Used to have it aligned next to menu button
        <v-spacer></v-spacer>
        -->
        
  
        <v-btn icon>
          <v-icon>mdi-magnify</v-icon>
        </v-btn>
  
        <v-btn icon>
          <v-icon>mdi-bell</v-icon>
        </v-btn>
  
        <v-btn icon>
          <v-icon>mdi-account-circle</v-icon>
        </v-btn>
      </v-app-bar>
  
      <v-navigation-drawer v-model="drawer" app color="background">
        <v-list>
          <v-list-item v-for="(item, index) in items" :key="index">

            <router-link :to="item.route" class="router-link" exact>
              <v-btn icon>
                <v-list-item-icon>
                  <v-icon >{{ item.icon }}</v-icon>
                </v-list-item-icon>
              </v-btn>
              <v-list-item-content>
                <v-list-item-title>{{ item.title }}</v-list-item-title>
              </v-list-item-content>
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
  