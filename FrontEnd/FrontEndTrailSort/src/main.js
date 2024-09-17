import { createApp } from 'vue'

import vuetify from './plugins/vuetify'
import App from './App.vue'
import router from './router'


// Create Vue app
const app = createApp(App)

// Use Vuetify
app.use(vuetify)

// Use Router
app.use(router)

// Mount the app
app.mount('#app')