// plugins/vuetify.js
import 'vuetify/styles'
import '@mdi/font/css/materialdesignicons.css' // Import MDI icons
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'

// Vuetify instance
const vuetify = createVuetify({
    components,
    directives,
    theme: {
        defaultTheme: 'dark', // Set dark theme globally
        themes: {
            dark: {
                colors: {
                    background: '#121212',
                    surface: '#212121',
                    //primary: '#BB86FC',\
                    primary: '#FB8C00',
                    secondary: '#03DAC6',
                    error: '#CF6679',

                    // https://vuetifyjs.com/en/styles/colors/#material-colors
                    // Custom orange colors
                    orangeDarken1: '#FB8C00',
                    orangeDarken2: '#F57C00',
                    orangeDarken3: '#EF6C00',
                    orangeDarken4: '#E65100',
                },
            },
        },
    },
    icons: {
        iconfont: 'mdi', // Use Material Design Icons
    },
})

export default vuetify

