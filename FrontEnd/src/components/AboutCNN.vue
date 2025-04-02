<template>
    <v-container>
      <!-- Title and Description -->
      <v-row class="my-4">
        <v-col>
          <h2 class="text-center">About CNN</h2>
          <p class="text-center">
            How do convolutional neural networks work? How are they able to identify and classify images?
          </p>
        </v-col>
      </v-row>

      <!-- CNN Architecture -->
      <v-row class="my-4">
      <v-col>
        <h2 >Model Overview</h2>
        <p >
          Here, we explain the structure of the model, which is divided into two main parts: Feature Extraction and Classification.
        </p>
      </v-col>
    </v-row>

    <!-- Explanation -->
    <v-row class="my-4">
      <v-col cols="12">
        <p class="mb-4">
          The model is divided into two key sections: the **Feature Extraction** part and the **Classification** part.
        </p>

        <p class="mb-4">
          <strong>Feature Extraction:</strong> This part of the model is responsible for analyzing the raw image data and learning the important features. It includes the **convolutional layers** and **pooling layers**:
        </p>
        
        <ul class="mb-4">
          <li><strong>Convolutional Layers:</strong> These layers are responsible for detecting basic patterns in the image, such as edges, textures, and shapes, by applying filters to the input data.</li>
          <li><strong>Pooling Layers:</strong> After convolution, pooling layers reduce the size of the feature maps produced, helping to preserve the key information while lowering computational cost.</li>
        </ul>

        <p class="mb-4 pl-6" >
          As the image data passes through these layers, the model learns more complex and abstract features, eventually building up a detailed understanding of the image.
        </p>

        <p class="mb-4">
          <strong>Classification:</strong> Once the image features have been extracted, the model moves to the classification phase, which consists of the **flattening** and **dense layers**:
        </p>

        <ul class="mb-4">
          <li><strong>Flattening:</strong> The 2D feature maps from the convolutional and pooling layers are flattened into a 1D vector, making them suitable for input into the dense layers.</li>
          <li><strong>Dense Layers:</strong> These layers take the flattened features and learn how to combine them in ways that lead to the final prediction, helping the model classify the image into the appropriate category.</li>
        </ul>

        <p class="mb-4 pl-6">
          Finally, the model outputs its prediction through the **output layer**, which represents the classification result based on the features learned during the classification part of the model.
        </p>
      </v-col>
    </v-row>
  
      <!-- Image Overview -->
      <v-row class="my-4">
        <v-col cols="12">
          <v-img src="/cnnOverview.jpg" alt="CNN Overview" />
        </v-col>
      </v-row>



      <v-row class="my-4">
      <v-col>
        <h2 >Hidden Layers</h2>
        <p >
          General idea behind how we can classify an image with math
        </p>
      </v-col>
    </v-row>
  
      <!-- Model Explanation -->
      <v-row class="my-4">
        <v-col cols="12">
          <p class="mb-4">
            We can now examine the hidden layers of the model. The core idea here is that the image is passed through a convolutional layer, where the model learns to identify and extract important features from the image. In this layer, a set of filters is applied to the image, and these filters are responsible for detecting various aspects such as edges, textures, and patterns. The convolutional layer is defined by several parameters: the number of filters, the size of the filter (also known as the kernel), padding, and the activation function.
          </p>
          
          <p class="mb-4">
            The number of filters determines how many different feature maps the layer will generate, and the kernel size defines the dimensions of the filter used to process the image. Padding is set to ensure that the output feature map has the same spatial dimensions as the input, maintaining consistency in size. The convolution operation involves sliding the kernel over the image, performing element-wise multiplication at each location, where the kernel values are multiplied by the pixel values in the image patch it is covering. The results are summed, a bias is added, and the result is passed through an activation function, such as the Rectified Linear Unit (ReLU), which replaces any negative values with zero.
          </p>
          <p class="mb-4">
            After the convolution operation, a pooling layer (often max pooling) is applied. This layer reduces the size of the feature map while attempting to retain the most significant features identified in the convolutional layer. The pooling operation reduces the dimensionality of the data, lowering the computational cost but still preserving important details. This process is repeated across multiple layers, progressively extracting higher-level features while reducing the size of the data at each step.
          </p>
        </v-col>
      </v-row>
  
      <!-- Explanation Image -->
      <v-row class="my-4">
        <v-col cols="12">
          <v-img src="/cnnExplained.png" alt="CNN Explained" class="mb-4" />
        </v-col>
      </v-row>


      <h1>Code Behind the Models</h1>
      <p>Select which model's Jupyter Notebook you want to look at.</p>

      <TensorFlowJupyter></TensorFlowJupyter>


    </v-container>
  </template>
  
  <script setup>
  
    import TensorFlowJupyter from './TensorFlowJupyter.vue';
  </script>
  
  <style lang="scss" scoped>
     /* Center text for headers and paragraphs */
  .text-center {
    text-align: center;
  }

  /* Ensure images take up the full width */
  v-img {
    width: 100%;
    max-width: 100%;
  }

  .my-4 {
    margin-top: 16px;
    margin-bottom: 16px;
  }


  /* Margin bottom for paragraphs */

  /* Optional: Style for lists */
  ul {
    padding-left: 20px;
  }
  </style>
  