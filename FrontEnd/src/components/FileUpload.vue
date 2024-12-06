
<template>
    <h2>File Upload</h2>
    <v-container>


        <ModelSelect ref="modelSelect" />


        <!-- https://vuetifyjs.com/en/components/file-inputs/#usage -->
        <v-file-input
            label="Upload Trail Camera Photos"
            variant="solo-inverted"
            prepend-icon="mdi-camera"
            v-model="file"
            @change="onFileChange"
            accept=".zip"
            :loading="isLoading"
            :disabled="isLoading"
        />
        <!-- https://vuetifyjs.com/en/components/alerts/#icon -->
        <v-alert v-if="showError" type="error" dismissible v-model:show="showError">
            {{ errorMessage }}
        </v-alert>

        <v-btn 
            @click="uploadFile"
            color="orangeDarken2" 
            :loading="isLoading" 
            :disabled="!file || isLoading"
        >
            Upload
        </v-btn>

        <v-btn
            v-if= "downloadUrl"
            @click = "downloadFile"
            color = orangeDarken2
        >
            Download
        </v-btn>



    </v-container>
</template>
  
<script setup>

    import { ref } from 'vue';
    import { useRouter } from 'vue-router';

    import ModelSelect from './ModelSelect.vue';

    const backendUrl = import.meta.env.VITE_BACKEND_URL;

    const router = useRouter();

    const file = ref(null);
    const isLoading = ref(false);
    const downloadUrl = ref(null);
    const filename = ref('');
    const errorMessage = ref('');
    const showError = ref(false);

    const modelSelect = ref(null);


    const onFileChange = () => {
        console.log("Selected file:", file.value);
    };

    const uploadFile = async () => {
        if (!file.value) return;

        isLoading.value = true;
        errorMessage.value = '';
        showError.value = false;

        // Add the file
        const formData = new FormData();
        formData.append('file', file.value);

        // Add the type of model
        const backend = modelSelect.value?.selectedBackEnd;
        console.log("Selected Model", backend);
        if (backend) {
            formData.append('model_type', backend);
        }

        // Retrieve the token from localStorage
        const token = localStorage.getItem('token');
        //console.log(token);


        // Check if token is null and redirect to userPage
        if (!token) {
            router.push({ name: 'userPage' }); 
            return; // Exit
        }

        try {
            const response = await fetch(`${backendUrl}/sort`, {
                method: 'POST',
                headers: {
                'Authorization': `Bearer ${token}`,
                },
                body: formData,
            });
            

            if (!response.ok) {
                const errorData = await response.json();
                console.error("Server error:", errorData);
                errorMessage.value = errorData.message || JSON.stringify(errorData) || 'Upload failed';
                throw new Error(errorMessage.value || 'Upload failed');
            }

            // Get filename
            const contentDisposition = response.headers.get('Content-Disposition');
            const filenameMatch = contentDisposition && contentDisposition.match(/filename="?([^"]*)"?/);
            filename.value = filenameMatch ? filenameMatch[1] : 'sorted_images.zip';

            // Handle ZIP file download
            const blob = await response.blob();
            
            // Verify blob size
            if (blob.size === 0) {
                throw new Error('Received empty file from server');
            }

            // Create link to download blob which is the zip folder
            downloadUrl.value = window.URL.createObjectURL(blob);;
            
            // Reset file input
            file.value = null;


        } catch (error) {
            console.error("Error uploading file:", error);
            errorMessage.value = error.message || 'An error occurred during the file upload.';
            showError.value = true;
        } finally {
            isLoading.value = false;
        }

    };


    // https://stackoverflow.com/questions/54771261/trying-to-send-a-zip-from-the-backend-to-the-frontend
    const downloadFile = () => {
        if (downloadUrl.value) {
            const link = document.createElement('a');
            link.href = downloadUrl.value;
            link.download = filename.value;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Revoke object URL to free memory
            window.URL.revokeObjectURL(downloadUrl.value);
            downloadUrl.value = null; // hide button
        }
    };

</script>
  
<style scoped>
  
</style>
  