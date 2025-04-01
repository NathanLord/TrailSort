<template>
    <div class="full-height">
            <v-container>
                    <v-row class="d-flex justify-center" style="min-height: 100vh;">
                            <v-col cols="12" md="6" lg="4">
                                    <v-card v-if="!authStore.isAuthenticated">
                                            <v-toolbar color="orange darken-2">
                                                    <v-toolbar-title>{{ isSignUp ? "Sign Up Form" : "Login Form" }}</v-toolbar-title>
                                            </v-toolbar>

                                            <v-card-text>
                                                    <v-form>
                                                            <v-text-field
                                                                    v-if="isSignUp"
                                                                    prepend-icon="mdi-account"
                                                                    name="firstName"
                                                                    label="First Name"
                                                                    type="text"
                                                                    v-model="firstName"
                                                            ></v-text-field>
                                                            
                                                            <v-text-field
                                                                    v-if="isSignUp"
                                                                    prepend-icon="mdi-account"
                                                                    name="lastName"
                                                                    label="Last Name"
                                                                    type="text"
                                                                    v-model="lastName"
                                                            ></v-text-field>
                                                            
                                                            <v-text-field
                                                                    v-if="isSignUp"
                                                                    prepend-icon="mdi-email"
                                                                    name="email"
                                                                    label="Email"
                                                                    type="email"
                                                                    v-model="email"
                                                            ></v-text-field>

                                                            <v-text-field
                                                                    prepend-icon="mdi-account"
                                                                    name="username"
                                                                    label="Username"
                                                                    type="text"
                                                                    v-model="username"
                                                            ></v-text-field>
                                                            <v-text-field
                                                                    id="password"
                                                                    prepend-icon="mdi-lock"
                                                                    name="password"
                                                                    label="Password"
                                                                    type="password"
                                                                    v-model="password"
                                                            ></v-text-field>
                                                    </v-form>

                                                    <!-- Error Message -->
                                                    <v-alert v-if="errorMessage" type="error" dismissible>{{ errorMessage }}</v-alert>
                                            </v-card-text>

                                            <!--Button for SignUp or Login -->
                                            <v-card-actions class="d-flex justify-center">
                                                    <v-btn
                                                            color="orange darken-2"
                                                            @click="isSignUp ? signup() : login()"
                                                            :disabled="isSignUp ? isSignUpDisabled() : isLoginDisabled()"
                                                    >
                                                            {{ isSignUp ? "Sign Up" : "Login" }}
                                                    </v-btn>
                                            </v-card-actions>

                                            <!-- Swap between SignUp and Login-->
                                            <v-card-actions class="d-flex justify-center mt-3">
                                                    <v-btn text @click="toggleForm">
                                                            {{ isSignUp ? "Already have an account? Login" : "Don't have an account? Sign Up" }}
                                                    </v-btn>
                                            </v-card-actions>
                                    </v-card>

                                    <!-- Already logged in with JWT-->
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
import { ref } from 'vue';
import { useAuthStore } from '../stores/auth';

const authStore = useAuthStore();
const backendUrl = import.meta.env.VITE_BACKEND_URL;


const firstName = ref('');
const lastName = ref('');
const email = ref('');
const username = ref('');
const password = ref('');
const isLoading = ref(false);
const errorMessage = ref('');
const isSignUp = ref(true);

const toggleForm = () => {
    firstName.value = '';
    lastName.value = '';
    email.value = '';
    username.value = '';
    password.value = '';
    isSignUp.value = !isSignUp.value;
};

const isSignUpDisabled = () => {
    return (
            !firstName.value ||
            !lastName.value ||
            !email.value ||
            !username.value ||
            !password.value ||
            username.value.length < 5 ||
            password.value.length < 5
    );
};

const isLoginDisabled = () => {
    return username.value.length < 5 || password.value.length < 5;
};

const signup = async () => {
    isLoading.value = true;
    errorMessage.value = '';

    if (isSignUpDisabled()) {
            errorMessage.value = 'Please fill out all fields.';
            return;
    }

    try {
            const response = await fetch(`${backendUrl}/user/signup`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                            firstName: firstName.value,
                            lastName: lastName.value,
                            email: email.value,
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
            firstName.value = '';
            lastName.value = '';
            email.value = '';
            username.value = '';
            password.value = '';
    } catch (error) {
            console.error('Error during SignUp:', error);
            errorMessage.value = error.message || 'An error occurred during SignUp.';
    } finally {
            isLoading.value = false;
    }
};

const login = async () => {
    errorMessage.value = '';

    isLoading.value = true;
    if (isLoginDisabled()) {
            errorMessage.value = 'Please fill out all fields.';
            return;
    }

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
                    authStore.setToken(data.token);
                    username.value = '';
                    password.value = '';
            } else {
                    throw new Error('Token not received');
            }
    } catch (error) {
            console.error('Error during Login:', error);
            errorMessage.value = error.message || 'An error occurred during Login.';
    } finally {
            isLoading.value = false;
    }
};

const logout = () => {
    authStore.logout();
};
</script>

<style scoped></style>
