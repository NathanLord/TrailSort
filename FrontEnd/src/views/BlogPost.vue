<template>

  <div>

    <!-- Preview of the Full Blog / Article -->
    <v-container class="d-flex justify-center">
        <v-row justify="center">
          <v-col cols="12" md="8" lg="6">

            <!-- Check if blog data is available -->
            <div v-if="blog">
              <!-- Display Blog Title -->
              <div v-html="blog.title"></div>

              <div class="mb-8">
                <p>By: {{ blog.author }}</p>
                <p>{{ blog.date }}</p>
              </div>

              <!-- Display Image if Available -->
              <v-img v-if="blog.image" :src="blog.image" contain max-height="800px" width="100%"></v-img>

              <!-- Display the Blog Content -->
              <div class="mt-8" v-html="blog.content"></div>
            </div>

            <!-- Display loading or error message if no blog data -->
            <div v-else>
              <p v-if="error">{{ error }}</p>
              <p v-else>Loading...</p>
            </div>

          </v-col>
        </v-row>
      </v-container>

  </div>
      

  </template>
  
  <script setup>
  import { ref, onMounted } from 'vue';
  import { useRoute } from 'vue-router';  // Import to access route params
  
  const route = useRoute();  // Get the current route
  const blog = ref(null);  // To store the blog post data
  const error = ref(null);  // Optional: To store any error messages

  const backendUrl = import.meta.env.VITE_BACKEND_URL
  

  const fetchBlogPost = async (id) => {
        try {
            const headers = {
              'Content-Type': 'application/json',
            };

            // Perform GET request to retrieve blog posts
            const response = await fetch(`${backendUrl}/blog/${id}`, {
              method: 'GET',
              headers: headers,
            });

            // Handle the response and check if it's successful
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Parse the response JSON and map to desired format
            const data = await response.json();
            
            // If blog data exists, assign it to the blog ref
            if (data) {
              // Assuming `data` is the full blog post object
              blog.value = {
                id: data.id,
                title: data.title,
                content: data.content,
                image: data.image ? `data:image/png;base64,${data.image}` : data.image, // Handle base64 image if needed
                author: data.author,
                date: data.date,
              };
            } else {
              // Handle case when no data is returned
              error.value = 'Blog post not found.';
            }
            

        } catch (error) {
            // Handle errors and set the error message
            console.error('Error retrieving blog posts:', error);
            error.value = 'Failed to fetch blog posts. Please try again later.';
        }
    };
  
  onMounted(() => {
    const blogId = route.params.id;  // Get the ID from the route params
    fetchBlogPost(blogId);  // Fetch the full blog post by ID
  });
  </script>
  