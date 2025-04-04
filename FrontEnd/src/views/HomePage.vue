<template>
    <v-container class="fill-height d-flex justify-center align-center">
      <v-row justify="center">
        <v-col cols="12" md="8" lg="6">


          <h1 class="text-center">Home</h1>

          <v-row class="my-4">
            <v-col cols="12">
              <v-img src="/wormBurglar.png" alt="Worm Burglar" max-height="200" />
            </v-col>
          </v-row>

        
          <p>Welcome to my Trail Sort project.</p>
          <p>
            
            The goal of the project was to train a model to be able to help me sort through trail camera images I have. 
            Go to the Sort Page to upload a zip file of images and receive sorted images back.
          </p>
  
          <BlogPosts> </BlogPosts>
        </v-col>
      </v-row>
    </v-container>
  </template>
  
  <script setup>
  import BlogPosts from '../components/BlogPosts.vue';
  </script>
  
  <style scoped>

  </style>
  