# %%
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# %% [markdown]
# The project follows this tutorial https://www.tensorflow.org/tutorials/images/classification 
# 
# This project uses a dataset of animal photos. The dataset contains five sub-directories, one per class:
# 
# location = D:\datasets\iNaturalCleaned
# 
# ```
# D:\datasets\iNaturalCleaned\blackBear
# D:\datasets\iNaturalCleaned\coyote
# D:\datasets\iNaturalCleaned\ruffedGrouse
# D:\datasets\iNaturalCleaned\turkey
# D:\datasets\iNaturalCleaned\whitetailDeer
# ```

# %%
# Specify the location of your dataset
data_dir = pathlib.Path(r'D:\datasets\iNaturalCleaned_temp')

# Count the number of images
image_count = len(list(data_dir.glob('*/*.jpg')))
print(f"Total number of images: {image_count}")

# Example: Load and display an image from the 'blackBear' folder
black_bears = list(data_dir.glob('blackBear/*'))
PIL.Image.open(str(black_bears[0]))

# %%
PIL.Image.open(str(black_bears[1]))

# %% [markdown]
# ## Load data using a Keras utility
# 
# Next, load these images off disk using the helpful `tf.keras.utils.image_dataset_from_directory` utility. This will take you from a directory of images on disk to a `tf.data.Dataset` in just a couple lines of code. If you like, you can also write your own data loading code from scratch by visiting the [Load and preprocess images](../load_data/images.ipynb) tutorial.

# %% [markdown]
# ### Create a dataset

# %% [markdown]
# Define some parameters for the loader:

# %%
batch_size = 32
img_height = 256
img_width = 256

# %% [markdown]
# It's good practice to use a validation split when developing your model. Use 80% of the images for training and 20% for validation.

# %%
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# %%
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# %% [markdown]
# You can find the class names in the `class_names` attribute on these datasets. These correspond to the directory names in alphabetical order.

# %%
class_names = train_ds.class_names
print(class_names)

# %% [markdown]
# ## Visualize the data
# 

# %% [markdown]
# Here are the first nine images from the training dataset:

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# %% [markdown]
# ## Configure the dataset for performance
# 
# Make sure to use buffered prefetching, so you can yield data from disk without having I/O become blocking. These are two important methods you should use when loading data:
# 
# - `Dataset.cache` keeps the images in memory after they're loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.
# - `Dataset.prefetch` overlaps data preprocessing and model execution while training.
# 
# Interested readers can learn more about both methods, as well as how to cache data to disk in the *Prefetching* section of the [Better performance with the tf.data API](../../guide/data_performance.ipynb) guide.

# %%
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %% [markdown]
# ## Standardize the data

# %% [markdown]
# The RGB channel values are in the `[0, 255]` range. This is not ideal for a neural network; in general you should seek to make your input values small.
# 
# Here, you will standardize values to be in the `[0, 1]` range by using `tf.keras.layers.Rescaling`:

# %%
normalization_layer = layers.Rescaling(1./255)

# %% [markdown]
# There are two ways to use this layer. You can apply it to the dataset by calling `Dataset.map`:

# %%
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# %% [markdown]
# ## A basic Keras model
# 
# ### Create the model
# 
# The Keras [Sequential](https://www.tensorflow.org/guide/keras/sequential_model) model consists of three convolution blocks (`tf.keras.layers.Conv2D`) with a max pooling layer (`tf.keras.layers.MaxPooling2D`) in each of them. There's a fully-connected layer (`tf.keras.layers.Dense`) with 128 units on top of it that is activated by a ReLU activation function (`'relu'`).

# %%
num_classes = len(class_names)

model = Sequential([
    layers.Input(shape=(img_height, img_width, 3)),  # Define input shape here
    layers.Rescaling(1./255),  # Rescale pixel values to [0, 1]
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# %% [markdown]
# ### Compile the model
# 
# For this tutorial, choose the `tf.keras.optimizers.Adam` optimizer and `tf.keras.losses.SparseCategoricalCrossentropy` loss function. To view training and validation accuracy for each training epoch, pass the `metrics` argument to `Model.compile`.

# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# %% [markdown]
# ### Model summary
# 
# View all the layers of the network using the Keras `Model.summary` method:

# %%
model.summary()

# %% [markdown]
# ### Train the model

# %% [markdown]
# Train the model for 10 epochs with the Keras `Model.fit` method:

# %%
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# %% [markdown]
# The increase in accuracy but decrease in validation accuracy is said to be a sign of overfitting or can also be a sign that the dataset is to complex for the size of the dataset that uou are using to train it.
# 
# Solutions
# 
# - Increase dataset size (artifically using data augmenation if can not get mroe data)
# - Stop early when validation accuracy starts to dip

# %% [markdown]
# ## Visualize training results
# 
# Create plots of the loss and accuracy on the training and validation sets:

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %% [markdown]
# ## Overfitting

# %% [markdown]
# In the plots above, the training accuracy is increasing linearly over time, whereas validation accuracy stalls around 50% in the training process. Also, the difference in accuracy between training and validation accuracy is noticeable—a sign of [overfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit).
# 
# When there are a small number of training examples, the model sometimes learns from noises or unwanted details from training examples—to an extent that it negatively impacts the performance of the model on new examples. This phenomenon is known as overfitting. It means that the model will have a difficult time generalizing on a new dataset.
# 
# There are multiple ways to fight overfitting in the training process. In this tutorial, you'll use *data augmentation* and add *dropout* to your model.

# %% [markdown]
# ## Data augmentation

# %% [markdown]
# Overfitting generally occurs when there are a small number of training examples. [Data augmentation](./data_augmentation.ipynb) takes the approach of generating additional training data from your existing examples by augmenting them using random transformations that yield believable-looking images. This helps expose the model to more aspects of the data and generalize better.
# 
# You will implement data augmentation using the following Keras preprocessing layers: `tf.keras.layers.RandomFlip`, `tf.keras.layers.RandomRotation`, and `tf.keras.layers.RandomZoom`. These can be included inside your model like other layers, and run on the GPU.

# %%
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# %% [markdown]
# Visualize a few augmented examples by applying data augmentation to the same image several times:

# %%
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")

# %% [markdown]
# ## Dropout
# 
# Another technique to reduce overfitting is to introduce [dropout](https://developers.google.com/machine-learning/glossary#dropout_regularization) regularization to the network.
# 
# When you apply dropout to a layer, it randomly drops out (by setting the activation to zero) a number of output units from the layer during the training process. Dropout takes a fractional number as its input value, in the form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20% or 40% of the output units randomly from the applied layer.
# 
# Create a new neural network with `tf.keras.layers.Dropout` before training it using the augmented images:

# %% [markdown]
# Overfitting could be seen by visualizing the training accuracy and the validation accuracy of models after training. Overfitting is when “the accuracy of models on the validation data would peak after training for a number of epochs and then stagnate or start decreasing.” (https://www.tensorflow.org/tutorials/keras/overfit_and_underfit). To help with overfitting a common tactic is using dropout. Dropout randomly selects a number of nodes / neurons and sets them to zero to deactivate them. For example, with ‘layers.Dropout(0.2),’ 20% of the neurons in the layer are randomly dropped during each training step, meaning 80% of the neurons remain active. This helps prevents overfitting by randomly deactivating neurons during training which forces the model to not rely on any specific set of nodes. That helps generalize better and encourages the model to learn more robust features that are not dependent on the presence of any specific node.

# %%
num_classes = len(class_names)

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])

# %% [markdown]
# ## Compile and train the model

# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# %%
model.summary()

# %%
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# %% [markdown]
# ## Visualize training results
# 
# After applying data augmentation and `tf.keras.layers.Dropout`, there is less overfitting than before, and training and validation accuracy are closer aligned:

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# %% [markdown]
# ## Predict on new data

# %% [markdown]
# Use your model to classify an image that wasn't included in the training or validation sets.

# %% [markdown]
# Note: Data augmentation and dropout layers are inactive at inference time.

# %% [markdown]
# Test out different animal images here

# %%
import os

# Specify the location of your dataset
image_path = pathlib.Path(r'D:\datasets\iNaturalCleaned')

test_image = list(image_path.glob('blackBear/*'))
test_image_path = (test_image[5654])
PIL.Image.open(str(test_image[5654]))

# %%
img = tf.keras.utils.load_img(
    test_image_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# %%
import os

# Specify the location of your dataset
image_path = pathlib.Path(r'D:\datasets\iNaturalCleaned')

test_image = list(image_path.glob('whitetailDeer/*'))
test_image_path = (test_image[5601])
PIL.Image.open(str(test_image[5601]))

# %%

img = tf.keras.utils.load_img(
    test_image_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# %%
import os

# Specify the location of your dataset
image_path = pathlib.Path(r'D:\datasets\iNaturalCleaned')

test_image = list(image_path.glob('ruffedGrouse/*'))
test_image_path = (test_image[5656])
PIL.Image.open(str(test_image[5656]))

# %%
img = tf.keras.utils.load_img(
    test_image_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# %%
import os

# Specify the location of your dataset
image_path = pathlib.Path(r'D:\datasets\iNaturalCleaned')

test_image = list(image_path.glob('turkey/*'))
test_image_path = (test_image[5655])
PIL.Image.open(str(test_image[5655]))

# %%
img = tf.keras.utils.load_img(
    test_image_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# %%
import os

# Specify the location of your dataset
image_path = pathlib.Path(r'D:\datasets\iNaturalCleaned')

test_image = list(image_path.glob('coyote/*'))
test_image_path = (test_image[5604])
PIL.Image.open(str(test_image[5604]))

# %%
img = tf.keras.utils.load_img(
    test_image_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

# %% [markdown]
# ## Save base model

# %%
# Save the model
model.save(r'C:\Users\kdlor\Documents\Documents\projects\trainingTrailSort\trailSortTF2MorePixels.keras')



# %% [markdown]
# ## Running Model

# %% [markdown]
# ### Load the Model

# %%
import tensorflow as tf

# Load the model
savedModel = tf.keras.models.load_model(r'C:\Users\kdlor\Documents\Documents\projects\trainingTrailSort\trailSortTF2MorePixels.keras')
savedModel.summary()


# %% [markdown]
# ### Prepare Input Data
# 
# You need to preprocess the input data (images) similarly to how you did when training the model. This typically includes resizing the image and normalizing pixel values.

# %%
from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_preprocess_image(img_path, target_size):
    # Load the image
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    return img_array


# %% [markdown]
# ### Predictions

# %%
import os
import pathlib
import PIL

img_height = 256
img_width = 256
# Define the target size as per your model's input shape
target_size = (img_height, img_width)  # Set this according to your model

# Specify the location of your dataset
image_path = pathlib.Path(r'D:\datasets\iNaturalCleaned')

test_image = list(image_path.glob('blackBear/*'))
test_image_path = (test_image[5654])
PIL.Image.open(str(test_image[5654]))



# %%
# class_names = 0,1,2,3,4, - blackBear,coyote,ruffedGrouse,turkey,whitetailDeer
class_names = ["blackBear", "coyote", "ruffedGrouse", "turkey", "whitetailDeer"]

img = tf.keras.utils.load_img(
    test_image_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = savedModel.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)


