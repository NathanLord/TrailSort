<template>
  <v-app>
    <v-app-bar app>

      <v-app-bar-nav-icon @click="drawer = !drawer"></v-app-bar-nav-icon>

      <!-- <router-link to="/" class="router-link text-center flex-grow-1"> -->
      <!-- Title Wrapper for Centering -->
      <div class="title-container">
        <router-link to="/" class="router-link">
          <v-toolbar-title>Trail Sort</v-toolbar-title>
        </router-link>
      </div>

      <v-spacer></v-spacer>

      <v-switch
        v-model="isDark"
        :label="isDark ? 'Dark Mode' : 'Light Mode'"
        @change="toggleTheme"
        inset
      ></v-switch>

      

      <router-link to="/user">
        <v-btn icon >
          <v-icon class="user-icon" >mdi-account-circle</v-icon>
        </v-btn>
      </router-link>

    </v-app-bar>

    <v-navigation-drawer v-model="drawer" app color="background"  temporary @click:outside="onClickOutside">
      <v-list>
        <v-list-item v-for="(item, index) in items" :key="index">

          <router-link :to="item.route" class="router-link" exact>

            <v-btn icon class="icon-btn">
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

  import { ref, watchEffect } from 'vue'
  import { useTheme } from 'vuetify'

  const drawer = ref(false)
  const items = [
    { title: 'Home', icon: 'mdi-home', route: '/' },
    { title: 'About', icon: 'mdi-information', route: '/about' },
    { title: 'Sort', icon: 'mdi-file-document-arrow-right ', route: '/sort' },
    { title: 'Blog', icon: 'mdi-post-outline', route: '/blog' }
  ]

  

  const theme = useTheme()
  const isDark = ref(theme.global.current.value.dark)

  const toggleTheme = () => {
    theme.global.name.value = theme.global.current.value.dark ? 'light' : 'dark'
  }

  watchEffect(() => {
    isDark.value = theme.global.current.value.dark
  })

  function onClickOutside() {
    console.log('onClickOutside called')
    console.log('drawer value:', drawer.value)
    if (drawer.value) {
      console.log('minimizing drawer')
      drawer.value = false
    }
  }

  
  
</script>


  

  
<style scoped>

  .title-container {
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
  }

  .v-application {
    font-family: 'Roboto', sans-serif;
  }

  .user-icon {
    color: rgb(var(--v-theme-on-surface)) !important;
  }

  .icon-btn {
    box-shadow: none !important;
    outline: none !important;
  }

  /* Make sure the router-link doesn't have underline and text color is white */
  .router-link {
    text-decoration: none;
    display: flex;
    align-items: center;
    color: rgb(var(--v-theme-on-surface));
  }

  .router-link-exact-active{
    color: rgb(var(--v-theme-primary)); /* https://stackoverflow.com/questions/48280990/using-custom-theming-in-vuetify-and-pass-color-variables-to-components */
  }

  .v-btn--variant-elevated, 
  .v-btn--variant-flat {
    background: rgb(var(--v-theme-background)) !important;
  }

  .v-list-item--active .v-btn--variant-elevated,
  .v-list-item--active .v-btn--variant-flat {
    color: rgb(var(--v-theme-primary));
  }
    

</style>
  