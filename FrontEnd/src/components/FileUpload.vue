
<template>
    <h2>File Upload</h2>
    <v-container>
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
    const backendUrl = import.meta.env.VITE_BACKEND_URL;

    const file = ref(null);
    const isLoading = ref(false);
    const downloadUrl = ref(null);
    const filename = ref('');

    const onFileChange = () => {
        console.log("Selected file:", file.value);
    };

    const uploadFile = async () => {
        if (!file.value) return;

        isLoading.value = true;

        const formData = new FormData();
        formData.append('file', file.value);

        // Retrieve the token from localStorage
        const token = localStorage.getItem('token');
        console.log(token);

        try {
            const response = await fetch(`${backendUrl}/sort`, {
                method: 'POST',
                headers: {
                'Authorization': `Bearer ${token}`, // Include the JWT token in the Authorization header
                },
                body: formData,
            });
            

            if (!response.ok) {
                const errorData = await response.json();
                console.error("Server error:", errorData);
                throw new Error(errorData.error || 'Upload failed');
            }

            // Get filename from Content-Disposition header if available
            const contentDisposition = response.headers.get('Content-Disposition');
            const filenameMatch = contentDisposition && contentDisposition.match(/filename="?([^"]*)"?/);
            filename.value = filenameMatch ? filenameMatch[1] : 'sorted_images.zip';

            // Handle ZIP file download
            const blob = await response.blob();
            
            // Verify blob size
            if (blob.size === 0) {
                throw new Error('Received empty file from server');
            }

            downloadUrl.value = window.URL.createObjectURL(blob);;
            
            // Reset file input
            file.value = null;


        } catch (error) {
            console.error("Error uploading file:", error);
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
  