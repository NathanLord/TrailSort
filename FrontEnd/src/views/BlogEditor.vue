<script setup>
import { ref, onMounted, nextTick  } from 'vue'
import { QuillyEditor } from 'vue-quilly'
import Quill from 'quill'
import 'quill/dist/quill.snow.css'

const backendUrl = import.meta.env.VITE_BACKEND_URL;

// Expose Quill globally
window.Quill = Quill

// Title model and options (limited formatting)
const titleModel = ref('')
const titleEditorRef = ref(null)
const titleOptions = ref({
  theme: 'snow',
  modules: {
    toolbar: [[{ header: '1' }, 'blockquote']], // Limited formatting
  },
  placeholder: 'Enter Title...',
  readOnly: false
})

// Content model and options (comprehensive)
const contentModel = ref('')
const contentEditorRef = ref(null)
const options = ref({
  theme: 'snow',
  modules: {
    toolbar: [
      [{ font: [] }, { size: [] }],
      ['bold', 'italic', 'underline', 'strike'],
      [{ color: [] }, { background: [] }],
      [{ script: 'super' }, { script: 'sub' }],
      [{ header: '1' }, { header: '2' }, 'blockquote', 'code-block'],
      [{ list: 'ordered' }, { list: 'bullet' }, { indent: '-1' }, { indent: '+1' }],
      ['direction', { align: [] }],
      ['link', 'image', 'video', 'formula'],
      ['clean']
    ],
    imageResize: {
      modules: ['Resize', 'DisplaySize', 'Toolbar']
    },
  },
  placeholder: 'Insert text here ...',
  readOnly: false
})

// Initialization on mount
onMounted(async () => {
  // Patch for style attributor
  window.Quill.imports.parchment.Attributor.Style = window.Quill.imports.parchment.StyleAttributor
  
  // Dynamically import image resize module
  const QuillImageResize = await import('quill-image-resize-module')
  
  // Register the image resize module
  window.Quill.register('modules/imageResize', QuillImageResize.default, { silent: true })
  
  // Initialize the title editor
  if (titleEditorRef.value) {
    titleEditorRef.value.initialize(window.Quill)
  }

  // Initialize the content editor
  if (contentEditorRef.value) {
    contentEditorRef.value.initialize(window.Quill)
  }



})

// Publish function (placeholder)
const publishModel = async () => {
  try {
    const titleData = titleModel.value;
    const contentData = contentModel.value;
    const headers = {
      'Content-Type': 'application/json',
    };

    // Handle both model and title for this post
    const response = await fetch(`${backendUrl}/blog/editor`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ title: titleData, content: contentData }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log(data);
  } catch (error) {
    console.error('Error publishing model:', error);
  }
}
</script>

<template>
  <div>


    <h3>Title Editor</h3>
    <QuillyEditor
      ref="titleEditorRef"
      v-model="titleModel"
      :options="titleOptions"
    />

    <h3>Content Editor</h3>
    <QuillyEditor
      ref="contentEditorRef"
      v-model="contentModel"
      :options="options"
    />

    <v-container class="d-flex justify-center">
      <v-row justify="center">
        <v-col cols="12" md="8" lg="6">

          <div  v-html="titleModel"></div>
          <div  v-html="contentModel"></div>

          <!-- <div class="ql-editor" v-html="contentModel"></div> -->



        </v-col>
      </v-row>
    </v-container>



    <div class="publish-section">
      <v-btn color="primary" @click="publishModel">Publish</v-btn>
    </div>



    


  </div>
</template>

<style scoped>

.ql-editor {
  margin: 0;         /* Remove any margins */
  padding: 10px;     /* Optional: Adjust padding as needed */
  line-height: 1.5;  /* Optional: Adjust line height for better spacing */
}

.publish-section {
  margin-top: 20px;
  text-align: center;
}
</style>