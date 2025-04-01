// src/stores/auth.js
import { defineStore } from 'pinia';
import { ref, computed, watch } from 'vue';

export const useAuthStore = defineStore('auth', () => {
    const token = ref(localStorage.getItem('token') || null);
    const user = ref(JSON.parse(localStorage.getItem('user')) || null);

    const isAuthenticated = computed(() => !!token.value);

    const setToken = (newToken) => {
        token.value = newToken;
        localStorage.setItem('token', newToken);  // Store token in localStorage
    };

    const setUser = (userData) => {
        user.value = userData;
        localStorage.setItem('user', JSON.stringify(userData)); // Store user in localStorage
    };

    const logout = () => {
        token.value = null;
        user.value = null;
        localStorage.removeItem('token');
        localStorage.removeItem('user');
    };

    return { token, user, isAuthenticated, setToken, setUser, logout };
});
