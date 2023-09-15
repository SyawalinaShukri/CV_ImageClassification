#%%
#1. Import packages
import patoolib
import os,datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers,optimizers,losses,callbacks,applications,models

#%%
#2. Data loading
_URL = 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/5y9wdsg2zt-2.zip'
path_to_zip = tf.keras.utils.get_file('5y9wdsg2zt-2.zip', origin=_URL, extract=True)
parent_dir = os.path.dirname(path_to_zip)
path_to_rar = os.path.join(parent_dir, 'Concrete Crack Images for Classification.rar')

directory_path = os.path.join(os.path.dirname(path_to_zip), 'cracks_and_no_cracks_filtered')
os.makedirs(directory_path, exist_ok=True)


# Specify the path to your .rar file
rar_file_path = path_to_rar

# Specify the destination directory for extraction
extract_dir = directory_path

# Use patoolib to extract the .rar file
patoolib.extract_archive(rar_file_path, outdir=extract_dir)

# Define batch size and image size for the dataset
BATCH_SIZE = 64
IMG_SIZE = (160, 160)

# Create a dataset from the extracted folders
data = tf.keras.utils.image_dataset_from_directory(
    directory_path,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
)

#%%
#3.0 Split dataset
train_size = int(len(data)*0.7)
val_size = int(len(data)*0.2) #to test model performance, during training phase
test_size = int(len(data)*0.1) #to test model performance, after training phase

#%%
train_dataset = data.take(train_size)
validation_dataset = data.skip(train_size).take(val_size)
test_dataset = data.skip(train_size+val_size).take(test_size)

# %%
#4. Inspect some data examples
class_names = data.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# %%
#5. Converting the tensorflow datasets into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
# %%
#6. Create a Sequential 'model' for data augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))
# %%
#7. Repeatedly apply data augmentation on a single image
for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')
# %%
#8. Define a layer for data normalization/rescaling
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
# %%
"""
The plan:

data augmentation > preprocess input > transfer learning model
"""

IMG_SHAPE = IMG_SIZE + (3,)
#(A) Load the pretrained model using keras.applications module
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
#Display summary of the model
base_model.summary()
#Display model structure
keras.utils.plot_model(base_model)
# %%
#(B) Freeze the entire feature extractor
base_model.trainable = False
# %%
#Display the model summary to show that most parameters are non-trainable
base_model.summary()
# %%
#(C) Create global average pooling layer
global_avg = layers.GlobalAveragePooling2D()
#(D) Create the output layer
output_layer = layers.Dense(len(class_names),activation='softmax')
#(E) Build the entire pipeline using functional API
#a. Input
inputs = keras.Input(shape=IMG_SHAPE)
#b. Data augmentation model
x = data_augmentation(inputs)
#c. Data rescaling layer
x = preprocess_input(x)
#d. Transfer learning feature extractor
x = base_model(x,training=False)
#e. Final extracted features
x = global_avg(x)
#f. Classification layer
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)
#g. Build the model
model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()
# %%
#10. Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
# %%
#Create TensorBoard callback object
base_log_path = r"tensorboard_logs\transfer_learning_tutorial"
log_path = os.path.join(base_log_path,datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_path)
# %%
#11. Model training
#Evaluate the model before training
loss0,acc0 = model.evaluate(test_dataset)
print("Evaluation before training:")
print("Loss = ", loss0)
print("Accuracy = ",acc0)
# %%
#12. Proceed with model training
early_stopping = callbacks.EarlyStopping(patience=2)
EPOCHS = 10
history = model.fit(train_dataset,validation_data=validation_dataset,epochs=EPOCHS,callbacks=[tb,early_stopping])
# %%
#13. Further fine tune the model
# Let's take a look to see how many layers are in the base model
base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

# %%
# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False
# %%
#14. Compile the model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
# %%
#15. Model training
fine_tune_epoch = 10
total_epoch = EPOCHS + fine_tune_epoch
history_fine = model.fit(train_dataset,validation_data=validation_dataset,epochs=total_epoch,initial_epoch=history.epoch[-1],callbacks=[tb,early_stopping])
# %%
#Evaluate the model after training
loss1,acc1 = model.evaluate(test_dataset)
print("Evaluation After Training:")
print("Loss = ",loss1)
print("Accuracy = ",acc1)
# %%
#16. Deployment
#(A) Retrieve a batch of images from the test set and perform predictions
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
# %%
prediction_index = np.argmax(predictions,axis=1)
#(B) Create a label map for the classes
label_map = {i:names for i,names in enumerate(class_names)}
prediction_label = [label_map[i] for i in prediction_index]
label_class_list = [label_map[i] for i in label_batch]
# %%
plt.figure(figsize=(10,10))
for i in range(9):
  ax = plt.subplot(3,3,i+1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(f"Label: {label_class_list[i]}, Prediction: {prediction_label[i]}")
  plt.axis('off')
  plt.grid('off')

# %%
# 17.Save the model
model.save(os.path.join('models', 'assesment1_model.h5'))
