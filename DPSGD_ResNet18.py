import numpy as np
import pandas as pd
import os

# Set environment variables BEFORE importing TensorFlow
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Disable parallel operations that cause NodeDef errors
os.environ['TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS'] = '1'
os.environ['TF_ENABLE_DETERMINISTIC_GPU'] = '1'

from tensorflow import keras
from keras import layers, models
from keras.layers import Conv2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
import pickle
import random
from classification_models.tfkeras import Classifiers
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report


########################### --- Seed Value Setting ---####################################
# Comprehensive seed setting
seed_value = 42

# Set all random seeds
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Configure TensorFlow threading
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Enable deterministic operations (with error handling)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print(f"Warning: Could not enable op determinism: {e}")

########################### --- GPU MEMORY LIMIT TO 10 GB ---####################################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 10)]
            )
    except RuntimeError as e:
        print(e)

############################ ---Hyperparameters---  ############################################################


learning_rate = 0.0001
batch_size = 32
num_epochs = 100
l2_norm_clip = 1.0  # Gradient clipping threshold
noise_multiplier = 1.1  # Amount of noise



######################### --- Function to load images with deterministic ordering --- ##############################

def load_images(InputImage, GroundTruth, img_size):
    images = []
    labels = []
    
    # Sort the dataframe to ensure consistent ordering
    GroundTruth_sorted = GroundTruth.sort_values('image').reset_index(drop=True)
    
    for i, row in GroundTruth_sorted.iterrows():
        img_path = os.path.join(InputImage, row["image"] + ".jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path).resize((img_size, img_size))
            img = np.asarray(img) / 255.0
            images.append(img)
            labels.append(row.iloc[1:].values.astype(np.float32))
        else:
            print(f"Missing image: {img_path}")
    return np.array(images), np.array(labels)

################################### ----Training_Input--- ############################################################
Training_Input = "/home/user2/prasun/ISIC2018_Task3_Training_Input/"
Training_GroundTruth = pd.read_csv("/home/user2/prasun/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")

IMG_SIZE = 224
x_train, y_train = load_images(Training_Input, Training_GroundTruth, IMG_SIZE)

print("Training image shape:", x_train.shape)
print("Training labels shape:", y_train.shape)

plt.imshow(x_train[20])
plt.axis('off')
plt.title("Sample Image")
plt.show()

################################## ----Validation_Input--- #############################################################

Validation_Input = "/home/user2/prasun/ISIC2018_Task3_Validation_Input/"
Validation_GroundTruth = pd.read_csv("/home/user2/prasun/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv")

x_val, y_val = load_images(Validation_Input, Validation_GroundTruth, IMG_SIZE)
print("Validation image shape:", x_val.shape)
print("Validation labels shape:", y_val.shape)

###############################  ----Load ResNet18 base model ---  ########################################################

ResNet18, preprocess_input = Classifiers.get('resnet18')

# Reset seeds before model creation
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

base_model = ResNet18(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=False
)

base_model.trainable = True

# Create model with explicit seed for each layer
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(64, 
          activation='relu', 
          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value),
          bias_initializer='zeros'),
    Dropout(0.4, seed=seed_value),
    Dense(28, 
          activation='relu', 
          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value + 1),
          bias_initializer='zeros'),
    Dropout(0.5, seed=seed_value + 1),
    Dense(7, 
          activation='softmax', 
          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value + 2),
          bias_initializer='zeros')
])




################################################################################################################
# Training setup
history = {
    'train_loss': [],
    'train_accuracy': [],
    'val_loss': [],
    'val_accuracy': []
}

# Create optimizer with deterministic behavior
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
train_accuracy = tf.keras.metrics.CategoricalAccuracy()

# Compile model
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Manual training loop for full control
def train_epoch(x_data, y_data, batch_size):
    """Train for one epoch with deterministic batching"""
    epoch_loss = 0.0
    train_accuracy.reset_state()
    
    # Create deterministic batches (no shuffling)
    num_samples = len(x_data)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        x_batch = x_data[i:end_idx]
        y_batch = y_data[i:end_idx]
        
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_fn(y_batch, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        
        
        # Clip gradients
        clipped_gradients = []
        for grad in gradients:
            grad_norm = tf.norm(grad)
            clipped_grad = grad * (l2_norm_clip / tf.maximum(grad_norm, l2_norm_clip))
            clipped_gradients.append(clipped_grad)

        # Add Gaussian noise
        noisy_gradients = [g + tf.random.normal(shape=g.shape, stddev=noise_multiplier * l2_norm_clip) for g in clipped_gradients]

        # Apply gradients
        optimizer.apply_gradients(zip(noisy_gradients, model.trainable_variables))

        
        train_accuracy.update_state(y_batch, predictions)
        epoch_loss += float(loss)
    
    avg_loss = epoch_loss / num_batches
    avg_acc = float(train_accuracy.result())
    
    return avg_loss, avg_acc

# Training loop
print("Starting training...")
with open("dpsgd_resnet18.txt", "w") as f:
    for epoch in range(num_epochs):
        # Reset random state for consistent behavior
        tf.random.set_seed(seed_value + epoch)
        np.random.seed(seed_value + epoch)
        
        # Train for one epoch
        avg_train_loss, avg_train_acc = train_epoch(x_train, y_train, batch_size)
        
        # Validation evaluation
        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0, batch_size=batch_size)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(avg_train_acc)
        history['val_loss'].append(float(val_loss))
        history['val_accuracy'].append(float(val_acc))
        
        # Logging
        log_line = (
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        print(log_line)
        f.write(log_line + "\n")



with open('historyfinaldp50.pkl', 'wb') as f:
    pickle.dump(history, f)
       
with open('historyfinaldp50.pkl', 'rb') as f:
    loaded_history = pickle.load(f)
       
      
##################################   ---Test evaluation--- ##########################################################################
 

Test_Input = "/home/user2/prasun/ISIC2018_Task3_Test_Input/"
Test_GroundTruth = pd.read_csv("/home/user2/prasun/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv")

x_test, y_test = load_images(Test_Input, Test_GroundTruth, IMG_SIZE)
print("Test image shape:", x_test.shape)
print("Test labels shape:", y_test.shape)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Get predicted probabilities
y_pred_probs = model.predict(x_test)

# Convert probabilities to predicted class indices
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Convert one-hot encoded true labels to class indices
y_true_classes = np.argmax(y_test, axis=1)

# Generate classification report
report = classification_report(y_true_classes, y_pred_classes, digits=4)
print("Classification Report:\n", report)








