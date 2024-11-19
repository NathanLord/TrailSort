
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
    </v-container>
</template>
  
<script setup>
    import { ref } from 'vue';
    const backendUrl = import.meta.env.VITE_BACKEND_URL;

    const file = ref(null);
    const isLoading = ref(false);

    const onFileChange = () => {
        console.log("Selected file:", file.value);
    };

    const uploadFile = async () => {
        if (!file.value) return;

        isLoading.value = true;

        const formData = new FormData();
        formData.append('file', file.value);

        try {
            const response = await fetch(`${backendUrl}/sort`, {
                method: 'POST',
                body: formData,
                //credentials: 'include',
            });
            

            if (!response.ok) {
                const errorData = await response.json();
                console.error("Server error:", errorData);
                throw new Error(errorData.error || 'Upload failed');
            }

            // Get filename from Content-Disposition header if available
            const contentDisposition = response.headers.get('Content-Disposition');
            const filenameMatch = contentDisposition && contentDisposition.match(/filename="?([^"]*)"?/);
            const filename = filenameMatch ? filenameMatch[1] : 'sorted_images.zip';

            // Handle ZIP file download
            const blob = await response.blob();
            
            // Verify blob size
            if (blob.size === 0) {
                throw new Error('Received empty file from server');
            }

            const downloadUrl = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = filename;
            
            // Trigger download
            document.body.appendChild(link);
            link.click();
            
            // Cleanup
            document.body.removeChild(link);
            window.URL.revokeObjectURL(downloadUrl);
            
            // Reset file input
            file.value = null;


        } catch (error) {
            console.error("Error uploading file:", error);
        } finally {
            isLoading.value = false;
        }

    };
</script>
  
<style scoped>
  
</style>
  