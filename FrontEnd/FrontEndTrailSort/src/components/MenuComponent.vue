<script setup>
  import { ref } from 'vue'

  const drawer = ref(false)
  const items = [
    { title: 'Home', icon: 'mdi-home', route: '/' },
    { title: 'About', icon: 'mdi-information', route: '/about' },
    { title: 'Contact', icon: 'mdi-phone', route: '/' },
  ]
</script>

<template>
    <v-app>
      <v-app-bar app>
        <v-app-bar-nav-icon @click="drawer = !drawer"></v-app-bar-nav-icon>
  
        <v-toolbar-title>My Menu</v-toolbar-title>
  
        <v-spacer></v-spacer>
  
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
  
      <v-navigation-drawer v-model="drawer" app>
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
    color: white; /* Sets the text color of the titles to white */
    display: flex;
    align-items: center;
  }

</style>
  