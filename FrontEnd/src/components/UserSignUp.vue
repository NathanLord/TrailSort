<!-- https://codesandbox.io/p/sandbox/0q4kvj8n0l?file=%2Fsrc%2Fcomponents%2FLogin.vue%3A2%2C4-37%2C12 -->

<template>
  <div class="full-height">
    <v-container>
      <v-row class="d-flex justify-center" style="min-height: 100vh;">
        <v-col cols="12" md="6" lg="4">
          <v-card v-if="!isAuthenticated">
            <v-toolbar color="orange darken-2">
              <v-toolbar-title>{{ isSignUp ? "Sign Up Form" : "Login Form" }}</v-toolbar-title>
            </v-toolbar>

            <v-card-text>
              <v-form>
                <v-text-field
                  prepend-icon="person"
                  name="username"
                  label="Username"
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

              <!-- Error Message (Login/Signup Failure) -->
              <v-alert v-if="errorMessage" type="error" dismissible>{{ errorMessage }}</v-alert>
            </v-card-text>

            <!-- Main Action Button (Login/Signup) -->
            <v-card-actions class="d-flex justify-center">
              <v-btn
                color="orange darken-2"
                @click="isSignUp ? signup() : login()"
                :disabled="username.length < 5 || password.length < 5"
              >
                {{ isSignUp ? "Sign Up" : "Login" }}
              </v-btn>
            </v-card-actions>

            <!-- Below Action Buttons (Toggle between SignUp/Login) -->
            <v-card-actions class="d-flex justify-center mt-3">
              <v-btn text @click="toggleForm">
                {{ isSignUp ? "Already have an account? Login" : "Don't have an account? Sign Up" }}
              </v-btn>
            </v-card-actions>
          </v-card>

          <!-- Welcome Message (For Authenticated Users) -->
          <v-card v-else>
            <v-card-title>Welcome, you are logged in!</v-card-title>
            <v-card-actions class="d-flex justify-center">
              <v-btn color="red darken-2" @click="logout">Sign Out</v-btn>
            </v-card-actions>
          </v-card>
        </v-col>
      </v-row>
    </v-container>
  </div>
</template>
  

<script setup>
import { ref, computed } from 'vue';
const backendUrl = import.meta.env.VITE_BACKEND_URL;

const username = ref('');
const password = ref('');
const isLoading = ref(false);
const errorMessage = ref('');
const isSignUp = ref(true);
const isAuthenticated = ref(!!localStorage.getItem('token'));


const toggleForm = () => {
  username.value = '';
  password.value = '';
  isSignUp.value = !isSignUp.value;
};

const signup = async () => {

  isLoading.value = true;
  errorMessage.value = '';

  if (!username.value || !password.value) {
      errorMessage.value = 'Please fill out all fields.';
      return;
  }
  console.log(username.value);

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
  isSignUp.value = false;
  username.value = '';
  password.value = '';


  } catch (error) {
      console.error('Error SignUp in:', error);
      errorMessage.value = error.message || 'An error occurred during SignUp.';
  } finally {
      isLoading.value = false;
  }
  };


  const login = async () => {

  errorMessage.value = '';
  console.log('Login', { username: username.value, password: password.value });

  isLoading.value = true;
  if (!username.value || !password.value) {
      errorMessage.value = 'Please fill out all fields.';
      return;
  }
  console.log(username.value);

  try {
  const response = await fetch(`${backendUrl}/user/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
      username: username.value,
      password: password.value,
      }),
  });

  if (!response.ok) {
      const errorData = await response.json();
      console.error('Login error:', errorData);
      throw new Error(errorData.error || 'Login failed');
  }

  const data = await response.json();

  if (data.token) {
    // Store JWT token in localStorage
    localStorage.setItem('token', data.token);
    console.log('Login successful:', data);
    isAuthenticated.value = true;
    username.value = '';
    password.value = '';
  } else {
    throw new Error('Token not received');
  }




  } catch (error) {
      console.error('Error logining in:', error);
      errorMessage.value = error.message || 'An error occurred during Login.';
  } finally {
      isLoading.value = false;
  }


};



const logout = () => {
  const token = localStorage.getItem('token');
  console.log(token);
  localStorage.removeItem('token');  
  console.log('Logged out successfully');
  isAuthenticated.value = false;
};

</script>

<style scoped>

</style>