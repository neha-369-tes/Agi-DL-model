"""
Colab Emotion Detection System
=============================

Optimized for Google Colab with T4 GPU support.
Handles both EEG and text emotion detection.

Author: Enhanced by AI Assistant based on Neha's original work
"""

# Install required packages for Colab
import subprocess
import sys

def install_packages():
    """Install required packages for Colab."""
    packages = [
        'tensorflow',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'scipy'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"⚠️ {package} already installed or failed to install")

# Install packages
print("Installing required packages...")
install_packages()

import numpy as np
import pandas as pd
import tensorflow as tf
# do not remove this import, it is needed for device listing
from tensorflow.python.client import device_lib
print("\nAvailable devices:")
print(device_lib.list_local_devices())
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
# till here
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, GRU, Bidirectional, 
    Conv1D, MaxPooling1D, GlobalAveragePooling1D,
    BatchNormalization, Input, Flatten, Embedding
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Text processing imports
import re
import string
from collections import Counter

# Check GPU availability
print("🔍 Checking GPU availability...")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")

class ColabEmotionDetector:
    """Colab-optimized emotion detection system."""
    
    def __init__(self):
        self.eeg_scaler = StandardScaler()
        self.text_processor = TextProcessor()
        self.label_encoder = LabelEncoder()
        
    def load_eeg_data(self, csv_path):
        """Load and prepare EEG data."""
        print(f"📊 Loading EEG data from {csv_path}")
        data = pd.read_csv(csv_path)
        print(f"EEG dataset shape: {data.shape}")
        
        # Separate features and labels
        feature_columns = [col for col in data.columns if col != 'label']
        X = data[feature_columns].values
        y = data['label'].values
        
        print(f"EEG features: {X.shape[1]}")
        print(f"EEG labels: {np.unique(y)}")
        
        return X, y
    
    def load_text_data(self, csv_path):
        """Load and prepare text data."""
        print(f"📝 Loading text data from {csv_path}")
        data = pd.read_csv(csv_path)
        print(f"Text dataset shape: {data.shape}")
        
        # Get text and labels
        texts = data['text'].fillna('').values
        y = data['label'].values
        
        print(f"Text samples: {len(texts)}")
        print(f"Text labels: {np.unique(y)}")
        
        return texts, y
    
    def train_eeg_model(self, X, y):
        """Train EEG emotion detection model."""
        print("\n🧠 === TRAINING EEG MODEL ===")
        
        # Reshape for CNN
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))
        y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, 
            stratify=np.argmax(y_train, axis=1), random_state=42
        )
        
        # Normalize features
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        
        X_train_normalized = self.eeg_scaler.fit_transform(X_train_reshaped)
        X_val_normalized = self.eeg_scaler.transform(X_val_reshaped)
        X_test_normalized = self.eeg_scaler.transform(X_test_reshaped)
        
        X_train = X_train_normalized.reshape(X_train.shape)
        X_val = X_val_normalized.reshape(X_val.shape)
        X_test = X_test_normalized.reshape(X_test.shape)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Number of classes: {num_classes}")
        
        # Create model
        model = self.create_eeg_model(X_train.shape[1:], num_classes)
        
        # Compile and train
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        ]
        
        print("🚀 Starting EEG model training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"🧠 EEG Model Test Accuracy: {test_accuracy:.4f}")
        
        return model, history, test_accuracy
    
    def train_text_model(self, texts, y):
        """Train text emotion detection model."""
        print("\n📝 === TRAINING TEXT MODEL ===")
        
        # Process text data
        self.text_processor.build_vocabulary(texts)
        X_text = np.array([self.text_processor.text_to_sequence(text) for text in texts])
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))
        y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_text, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, 
            stratify=np.argmax(y_train, axis=1), random_state=42
        )
        
        print(f"Text training data shape: {X_train.shape}")
        print(f"Number of classes: {num_classes}")
        print(f"Vocabulary size: {self.text_processor.vocab_size}")
        
        # Create model
        model = self.create_text_model(
            self.text_processor.vocab_size,
            self.text_processor.max_sequence_length,
            num_classes
        )
        
        # Compile and train
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        ]
        
        print("🚀 Starting text model training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"📝 Text Model Test Accuracy: {test_accuracy:.4f}")
        
        return model, history, test_accuracy
    
    def create_eeg_model(self, input_shape, num_classes):
        """Create CNN-GRU model for EEG."""
        inputs = Input(shape=input_shape)
        
        # CNN layers
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(0.2)(x)
        
        # GRU layers
        x = Bidirectional(GRU(256, return_sequences=True, dropout=0.2))(x)
        x = Bidirectional(GRU(128, return_sequences=False, dropout=0.2))(x)
        
        # Dense layers
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='EEG_Model')
        return model
    
    def create_text_model(self, vocab_size, max_sequence_length, num_classes):
        """Create LSTM model for text."""
        inputs = Input(shape=(max_sequence_length,))
        x = Embedding(vocab_size, 128, input_length=max_sequence_length)(inputs)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Bidirectional(LSTM(64, return_sequences=False))(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='Text_Model')
        return model

class TextProcessor:
    """Simple text processor for emotion detection."""
    
    def __init__(self, max_vocab_size=10000, max_sequence_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0
        
    def preprocess_text(self, text):
        """Clean and preprocess text."""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove punctuation
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def build_vocabulary(self, texts):
        """Build vocabulary from text corpus."""
        print("📚 Building vocabulary from text corpus...")
        word_counts = Counter()
        
        for text in texts:
            processed_text = self.preprocess_text(text)
            words = processed_text.split()
            word_counts.update(words)
        
        # Get most common words
        vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.most_common(self.max_vocab_size - 2)]
        
        self.word_to_index = {word: idx for idx, word in enumerate(vocab)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        self.vocab_size = len(vocab)
        
        print(f"✅ Vocabulary built with {self.vocab_size} words")
        return self.word_to_index
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices."""
        processed_text = self.preprocess_text(text)
        words = processed_text.split()
        
        sequence = []
        for word in words:
            if word in self.word_to_index:
                sequence.append(self.word_to_index[word])
            else:
                sequence.append(self.word_to_index['<UNK>'])
        
        # Pad or truncate to max_sequence_length
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length]
        else:
            sequence += [self.word_to_index['<PAD>']] * (self.max_sequence_length - len(sequence))
        
        return sequence

def main():
    """Main function to run emotion detection in Colab."""
    print("🎯 Colab Emotion Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = ColabEmotionDetector()
    
    # Train EEG model
    try:
        print("\n🧠 TRAINING EEG MODEL")
        X_eeg, y_eeg = detector.load_eeg_data("emotions.csv")
        eeg_model, eeg_history, eeg_accuracy = detector.train_eeg_model(X_eeg, y_eeg)
        print(f"✅ EEG Model trained successfully! Accuracy: {eeg_accuracy:.4f}")
    except Exception as e:
        print(f"❌ Error training EEG model: {e}")
        import traceback
        traceback.print_exc()
        eeg_accuracy = 0
    
    # Train text model
    try:
        print("\n📝 TRAINING TEXT MODEL")
        texts, y_text = detector.load_text_data("emotionstxt.csv")
        text_model, text_history, text_accuracy = detector.train_text_model(texts, y_text)
        print(f"✅ Text Model trained successfully! Accuracy: {text_accuracy:.4f}")
    except Exception as e:
        print(f"❌ Error training text model: {e}")
        import traceback
        traceback.print_exc()
        text_accuracy = 0
    
    # Compare results
    print("\n" + "="*50)
    print("🏆 MODEL COMPARISON")
    print("="*50)
    
    if eeg_accuracy > 0:
        print(f"🧠 EEG Model Accuracy: {eeg_accuracy:.4f}")
    
    if text_accuracy > 0:
        print(f"📝 Text Model Accuracy: {text_accuracy:.4f}")
    
    if eeg_accuracy > 0 and text_accuracy > 0:
        if eeg_accuracy > text_accuracy:
            print(f"🏆 EEG Model performs better!")
        elif text_accuracy > eeg_accuracy:
            print(f"🏆 Text Model performs better!")
        else:
            print(f"🏆 Both models perform equally!")
    
    print("\n✅ Emotion detection system completed!")

if __name__ == "__main__":
    main() 

