<template>
    <v-container fluid>
      <v-row>
        <v-col
          v-for="(card, index) in blogPosts"
          :key="index"
          cols="12"
          sm="6"
          md="4"
        >
          <v-hover v-slot="{ isHovering, props }">
            <v-card
              class="mx-auto"
              max-width="344"
              v-bind="props"
              @click="goToBlogPost(card.id)"
              :style="{ backgroundColor: 'transparent', boxShadow: 'none', border: 'none' }"
            >
                <!--
                <v-img :src="card.image"></v-img>
                -->

                <v-img v-if="card.image" :src="card.image" height="200" cover></v-img>
              
  
              <v-card-text>
                <h2 class="text-h6 text-primary">
                  {{ card.shortTitle }}
                </h2>
                <p>By: {{ card.author }}</p>
                <p>{{ card.date }} </p>
              </v-card-text>
  
              
  
              <v-overlay
                :model-value="!!isHovering"
                class="align-center justify-center"
                scrim="primary"
                contained
              >

                <!-- View Button -->
                <!--
                <v-btn variant="flat">View</v-btn>
                -->
                

              </v-overlay>
            </v-card>
          </v-hover>
        </v-col>
      </v-row>
    </v-container>
  </template>
  
<script setup>
    import { ref, onMounted } from 'vue';
    import { useRouter } from 'vue-router'; 

    const router = useRouter();

    const backendUrl = import.meta.env.VITE_BACKEND_URL

    const blogPosts = ref([]);
    const error = ref(null);

    const cards = [
    {
        title: 'Magento Forests',
        content: 'Travel to the best outdoor experience on planet Earth. A vacation you will never forget!',
        image: 'https://cdn.vuetifyjs.com/images/cards/forest-art.jpg'
    },
    {
        title: 'Desert Oasis',
        content: 'Enjoy the golden sunsets and unique landscapes of the desert.',
        image: 'https://cdn.vuetifyjs.com/images/cards/sunshine.jpg'
    },
    {
        title: 'Mountain Escape',
        content: 'Find peace and quiet among the peaks and fresh mountain air.',
        image: 'https://cdn.vuetifyjs.com/images/cards/house.jpg'
    },
    // Add more cards as needed
    ];

    const goToBlogPost = (id) => {
      //console.log('Navigating to blog post with ID:', id);
      router.push({ name: 'blog-post', params: { id } });  // Navigate to the blog post page with the id
    };

    const removeHtmlTags = (htmlString) => {
        const div = document.createElement('div');
        div.innerHTML = htmlString;
        return div.textContent || div.innerText || "";
    };

    const removeHtmlTagsWithNewlines = (htmlString) => {
        // Add a newline before any closing HTML tags
        return htmlString
            .replace(/<\/[^>]+>/g, '\n$&'); // Add newline before every closing tag
    };

    const getFirstH2Text = (htmlString) => {
      const match = htmlString.match(/<h2[^>]*>(.*?)<\/h2>/);
      return match ? match[1].trim() : ''; // Return the content or an empty string if not found
    };




    // get to retrive blog posts from database
    const retrieveBlogs = async () => {
        try {
            const headers = {
              'Content-Type': 'application/json',
            };

            // Perform GET request to retrieve blog posts
            const response = await fetch(`${backendUrl}/blog/retrieve`, {
              method: 'GET',
              headers: headers,
            });

            // Handle the response and check if it's successful
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Parse the response JSON and map to desired format
            const data = await response.json();
            // Log the response to inspect the structure
            //console.log('Received data:', data);

            // Check if the data is an array before calling map
            if (Array.isArray(data)) {
                blogPosts.value = data.map((blog) => {
                    return {
                    id: blog.id,
                    shortTitle: removeHtmlTags(blog.title), 
                    shortContent: getFirstH2Text(blog.content),
                    title: blog.title,
                    content: blog.content, 
                    image: blog.image ? `data:image/png;base64,${blog.image}` : null, // Handle base64 image
                    author: blog.author,
                    date: blog.date
                    };
                });
            } else {
                throw new Error('Expected an array but received something else');
            }

        } catch (error) {
            // Handle errors and set the error message
            console.error('Error retrieving blog posts:', error);
            error.value = 'Failed to fetch blog posts. Please try again later.';
        }
    };

    // Fetch blog posts when component is mounted
    onMounted(() => {
        retrieveBlogs();
    });


</script>
  