# train_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os

# --- 1. Configuration ---
DATASET_PATH = "dataset"
MODEL_PATH = "mask_detector.h5"
INIT_LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 32

# --- 2. Load and Preprocess Data ---
print("[INFO] Loading images...")
data = []
labels = []

categories = ["with_mask", "without_mask"]

for category in categories:
    path = os.path.join(DATASET_PATH, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        
        data.append(image)
        labels.append(category)

# One-hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# --- 3. Data Augmentation ---
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# --- 4. Build the Model (Transfer Learning) ---
# Load MobileNetV2, leaving off the head
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Construct the new head for our classifier
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel) # 2 classes: with/without mask

# Place the head on top of the base model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze the layers in the base model so they aren't trained
for layer in baseModel.layers:
    layer.trainable = False

# --- 5. Compile and Train the Model ---
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCH_SIZE,
    epochs=EPOCHS
)

# --- 6. Save the Model ---
print(f"[INFO] Saving mask detector model to {MODEL_PATH}...")
model.save(MODEL_PATH)                   # <-- NEW LINE (removed save_format)
print("[INFO] Training complete!")