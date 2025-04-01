import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import vuetify from './plugins/vuetify'
import router from './router'


// Create Vue app
const pinia = createPinia()
const app = createApp(App)


app.use(pinia)
// Use Vuetify
app.use(vuetify)


// Use Router
app.use(router)

// Mount the app
app.mount('#app')