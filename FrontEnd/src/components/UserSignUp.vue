<template>

    <!-- https://codesandbox.io/p/sandbox/0q4kvj8n0l?file=%2Fsrc%2Fcomponents%2FLogin.vue%3A2%2C4-37%2C12 -->
    <div class="full-height">
          <v-container>
            <v-row class="d-flex  justify-center" style="height: 100vh;">
              <v-col cols="12" md="6" lg="4">
                <v-card>
                  <v-toolbar color="orange darken-2">
                    <v-toolbar-title>Sign Up Form</v-toolbar-title>
                  </v-toolbar>
                  <v-card-text>
                    <v-form>
                      <v-text-field
                        prepend-icon="person"
                        name="username"
                        label="UserName"
                        type="text"
                        v-model="username"
                      ></v-text-field>
                      <v-text-field
                        id="password"
                        prepend-icon="lock"
                        name="password"
                        label="Password"
                        type="password"
                        v-model="password"
                      ></v-text-field>
                    </v-form>
                  </v-card-text>
                  <v-card-actions>
                    <v-spacer></v-spacer>
                    <v-btn color="orange darken-2" @click="signup" :disabled="username.length < 5 || password.length < 5">Sign Up</v-btn>
                  </v-card-actions>
                </v-card>
              </v-col>
            </v-row>
          </v-container>
    </div>

  </template>
  

<script setup>
import { ref } from 'vue';
const backendUrl = import.meta.env.VITE_BACKEND_URL;

const username = ref('');
const password = ref('');
const isLoading = ref(false);
const errorMessage = ref('');

const signup = async () => {

    isLoading.value = true;
    if (!username.value || !password.value) {
        errorMessage.value = 'Please fill out all fields.';
        return;
    }

    try {
    const response = await fetch(`${backendUrl}/user/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
        username: username.value,
        password: password.value,
        }),
    });

    if (!response.ok) {
        const errorData = await response.json();
        console.error('SignUp error:', errorData);
        throw new Error(errorData.error || 'SignUp failed');
    }

    const data = await response.json();
    console.log('SignUp successful:', data);

    // Redirect or perform actions upon successful login
    // e.g., store token, navigate to another page

    } catch (error) {
        console.error('Error SignUp in:', error);
        errorMessage.value = error.message || 'An error occurred during SignUp.';
    } finally {
        isLoading.value = false;
    }
};

</script>

<style scoped>

</style>