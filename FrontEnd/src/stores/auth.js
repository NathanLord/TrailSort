// src/stores/auth.js
import { defineStore } from 'pinia';
import { ref, computed, watch } from 'vue';

export const useAuthStore = defineStore('auth', () => {
    const token = ref(localStorage.getItem('token') || null);
    const user = ref(JSON.parse(localStorage.getItem('user')) || null);
    const firstName = ref(localStorage.getItem('first_name') || '');
    const lastName = ref(localStorage.getItem('last_name') || '');
    const role = ref(localStorage.getItem('role') || '');

    const isAuthenticated = computed(() => !!token.value);

    const setToken = (newToken) => {
        token.value = newToken;
        localStorage.setItem('token', newToken);  // Store token in localStorage
    };

    const setUserDetails = (first, last) => {
        firstName.value = first;
        lastName.value = last;
        localStorage.setItem('first_name', first);
        localStorage.setItem('last_name', last);
    };

    // Set user data (including first_name, last_name, and role)
    const setUser = (userData) => {
        user.value = userData;
        localStorage.setItem('user', JSON.stringify(userData)); // Store user in localStorage
        localStorage.setItem('role', userData.role); // Store role separately for easy access
    };

    const logout = () => {
        token.value = null;
        user.value = null;
        firstName.value = '';
        lastName.value = '';
        role.value = '';
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        localStorage.removeItem('first_name');
        localStorage.removeItem('last_name');
        localStorage.removeItem('role');
    };

    return { token, user, firstName, lastName, isAuthenticated, setToken, setUser, setUserDetails, logout };
});
