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
# DP-Adam hyperparameters
learning_rate = 0.0001
batch_size = 20
l2_norm_clip = 1.0  # Clipping threshold for DP
beta_1, beta_2 = 0.9, 0.999  # Adam momentum terms
epsilon_1 = 1e-8  # Smoothing term
noise_multiplier = 8.1 
num_epochs = 40

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
   
    Dense(28, 
          activation='relu', 
          kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value + 1),
          bias_initializer='zeros'),
    Dropout(0.2, seed=seed_value + 1),
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

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
train_accuracy = tf.keras.metrics.CategoricalAccuracy()


# Initialize DP-Adam moving averages using `tf.Variable`
m_t = [tf.Variable(tf.zeros_like(var), trainable=False) for var in model.trainable_variables]
v_t = [tf.Variable(tf.zeros_like(var), trainable=False) for var in model.trainable_variables]

@tf.function(input_signature=[
    tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),  # x_batch
    tf.TensorSpec(shape=(None, 7), dtype=tf.float32),  # y_batch
    tf.TensorSpec(shape=(), dtype=tf.int32)  # t (integer scalar)
])
def train_step(x_batch, y_batch, t):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)

    # Clip gradients
    clipped_gradients = []
    for g in gradients:
        norm = tf.norm(g)
        clipped_gradients.append(g * (l2_norm_clip / tf.maximum(norm, l2_norm_clip)))

    # Add Gaussian noise
    noisy_gradients = [
        g + tf.random.normal(shape=g.shape, stddev=noise_multiplier * l2_norm_clip)
        for g in clipped_gradients
    ]

    # Adam momentum & variance updates
    new_m_t, new_v_t = [], []
    for i in range(len(noisy_gradients)):
        mt_new = beta_1 * m_t[i] + (1 - beta_1) * noisy_gradients[i]
        vt_new = beta_2 * v_t[i] + (1 - beta_2) * tf.square(noisy_gradients[i])

        # Fix TypeError by casting t to float32
        m_hat = mt_new / (1 - beta_1 ** tf.cast(t, tf.float32))
        v_hat = vt_new / (1 - beta_2 ** tf.cast(t, tf.float32))
        noisy_gradients[i] = m_hat / (tf.sqrt(v_hat) + epsilon_1)

        new_m_t.append(mt_new)
        new_v_t.append(vt_new)

    # Apply gradients
    optimizer.apply_gradients(zip(noisy_gradients, model.trainable_variables))

    train_accuracy.update_state(y_batch, predictions)
    return new_m_t, new_v_t, loss  # Return updated values
    


t = 1  # Adam time step counter
with open("dpadam_resnet18_1.txt", "w") as f:
    for epoch in range(num_epochs):
        train_accuracy.reset_state()  # Reset accuracy at start of each epoch
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
      
            # Call train_step
            new_m_t, new_v_t, loss = train_step(x_batch, y_batch, tf.convert_to_tensor(t, dtype=tf.int32))

            # Convert before assigning to prevent SymbolicTensor errors
            for j in range(len(m_t)):
                m_t[j].assign(tf.convert_to_tensor(new_m_t[j], dtype=tf.float32))
                v_t[j].assign(tf.convert_to_tensor(new_v_t[j], dtype=tf.float32))

            t += 1  # Increment time step
      
        epoch_accuracy = train_accuracy.result().numpy()
      
        
   
    # Evaluate accuracy
        model.compile(loss=loss_fn, metrics=['accuracy'])
        Val_loss, Val_acc = model.evaluate(x_val, y_val)
        
        
        history['train_loss'].append(loss.numpy())
        history['train_accuracy'].append(epoch_accuracy)
        history['val_loss'].append(Val_loss)
        history['val_accuracy'].append(Val_acc)
     
        
        log_line = (    
             f"Epoch {epoch+1}/{num_epochs}, "   
             f"Train Loss: {loss.numpy():.4f}, Train Acc: {epoch_accuracy:.4f}, "
             f"Val Loss: {Val_loss:.4f}, Val Acc: {Val_acc:.4f}"
             )
        print(log_line)
        f.write(log_line + "\n")




with open('dpadam_resnet18_1.pkl', 'wb') as f:
    pickle.dump(history, f)
       
with open('dpadam_resnet18_1.pkl', 'rb') as f:
    loaded_history = pickle.load(f)
       
      
      
##################################   ---Test evaluation--- ##########################################################################
 

Test_Input = "/home/user2/prasun/ISIC2018_Task3_Test_Input/"
Test_GroundTruth = pd.read_csv("/home/user2/prasun/ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv")

x_test, y_test = load_images(Test_Input, Test_GroundTruth, IMG_SIZE)
print("Test image shape:", x_test.shape)
print("Test labels shape:", y_test.shape)

with open("dpadam_resnet18_1.txt", "a") as f:
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
  #print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


    log_line = (      
             f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}"
             )
    print(log_line)
    f.write(log_line + "\n")


model.save('DPADAM_ResNet18_1.keras')






