import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Use tf.keras for layers and models
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, BatchNormalization, LeakyReLU, Dropout, Input
from keras.optimizers import Adam
import os

# --- Configuration ---
INPUT_CSV = 'Fuzzy_for_gen.csv' # Input file
OUTPUT_CSV = 'synthetic_Fuzzy_dataframe_generated_deep_gan.csv' # Output file
GENERATOR_MODEL_PATH = 'generator_Fuzzy_deep_model.h5'
DISCRIMINATOR_MODEL_PATH = 'discriminator_Fuzzy_deep_model.h5'
PAYLOAD_LENGTH = 8
LATENT_DIM = 100 # Dimension of the random noise vector
EPOCHS = 300 # Deeper models might need more epochs
BATCH_SIZE = 64 # Adjust batch size if needed
LEARNING_RATE = 0.0002
BETA_1 = 0.5 # Parameters commonly used in Adam for GANs
DROPOUT_RATE = 0.3 # Dropout rate for Discriminator regularization

# --- Data Loading and Preprocessing ---

# Convert 'Payload' to a numerical representation
def hex_to_int_array(hex_string, length=PAYLOAD_LENGTH):
    """Converts a space-separated hex string to a numpy array of integers."""
    default_payload = np.zeros(length, dtype=int) # Default to zeros
    if isinstance(hex_string, str):
        try:
            hex_values = hex_string.strip().split(' ')
            hex_values = [h for h in hex_values if h] # Remove empty strings from multiple spaces
            int_array = np.array([int(h, 16) for h in hex_values if len(h)>0])
             # Pad or truncate
            if len(int_array) < length:
                padded_array = np.pad(int_array, (0, length - len(int_array)), 'constant', constant_values=0) # Pad with 0
            else:
                padded_array = int_array[:length] # Truncate
            return padded_array
        except ValueError:
            # print(f"Warning: Could not parse hex string: '{hex_string}'. Using default payload.")
            return default_payload
        except Exception as e:
             # print(f"Warning: Unexpected error parsing hex string '{hex_string}': {e}. Using default payload.")
             return default_payload
    elif pd.isna(hex_string):
        return default_payload # Default for missing values
    else:
        # print(f"Warning: Unexpected data type in Payload: {type(hex_string)}, value: {hex_string}. Using default payload.")
        return default_payload

def safe_hex_to_int(hex_str):
    """Safely converts hex string (potentially read as float/int) to int."""
    if isinstance(hex_str, str):
        try:
            return int(hex_str.replace(' ', ''), 16) # Remove spaces and convert
        except ValueError:
            # print(f"Warning: Invalid hex string in ID/RemoteFrame: '{hex_str}'. Using 0.")
            return 0
    try:
        return int(float(hex_str)) # Try converting from float if it was parsed as one
    except (ValueError, TypeError):
        # print(f"Warning: Invalid numeric value in ID/RemoteFrame: {hex_str}. Using 0.")
        return 0

# Load the dataset
print(f"Loading dataset from {INPUT_CSV}...")
if not os.path.exists(INPUT_CSV):
    print(f"Error: Input file not found at {INPUT_CSV}")
    exit()
try:
    data = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(data)} rows.")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()


# Apply payload conversion
data['Payload_Numeric'] = data['Payload'].apply(lambda x: hex_to_int_array(x, PAYLOAD_LENGTH))

# Prepare the data for the GAN
def preprocess_data(df):
    """Normalizes relevant columns and combines features."""
    print("Preprocessing data...")
    df_processed = df.copy()

    # --- Normalization ---
    # TimeInterval: Normalize by max value
    max_timeinterval = df_processed['TimeInterval'].max()
    df_processed['TimeInterval_Normalized'] = df_processed['TimeInterval'] / max_timeinterval if max_timeinterval > 0 else 0

    # DLC: Normalize by max value (typically 8)
    max_dlc = df_processed['DLC'].max()
    df_processed['DLC_Normalized'] = df_processed['DLC'] / max_dlc if max_dlc > 0 else 0

    # ID: Convert hex to int, then normalize by max ID
    df_processed['ID_Int'] = df_processed['ID'].apply(safe_hex_to_int)
    max_id = df_processed['ID_Int'].max() if not df_processed['ID_Int'].empty else 1
    max_id = max(1, max_id) # Ensure not zero
    df_processed['ID_Normalized'] = df_processed['ID_Int'] / max_id

    # RemoteFrame: Convert hex to int, then normalize by max RF (typically 1)
    df_processed['RemoteFrame_Int'] = df_processed['RemoteFrame'].apply(safe_hex_to_int)
    max_rf = df_processed['RemoteFrame_Int'].max() if not df_processed['RemoteFrame_Int'].empty else 1
    max_rf = max(1, max_rf) # Ensure not zero
    df_processed['RemoteFrame_Normalized'] = df_processed['RemoteFrame_Int'] / max_rf

    # Payload: Stack numeric payloads and normalize byte values to [0, 1]
    payload_matrix = np.stack(df_processed['Payload_Numeric'].values)
    payload_normalized = payload_matrix / 255.0

    # Combine features into a single numpy array
    feature_cols = ['TimeInterval_Normalized', 'ID_Normalized', 'RemoteFrame_Normalized', 'DLC_Normalized']
    processed_features = df_processed[feature_cols].values
    final_data = np.concatenate([processed_features, payload_normalized], axis=1)

    print("Preprocessing complete.")
    return final_data, max_timeinterval, max_dlc, max_id, max_rf

# Preprocess the loaded data
processed_data, max_time_global, max_dlc_global, max_id_global, max_rf_global = preprocess_data(data)
num_features = processed_data.shape[1]
print(f"Number of features per sample: {num_features}")
print(f"Shape of processed data: {processed_data.shape}")

real_data = processed_data # Use all data for training

# --- Build Deeper Models ---

# Define the DEEPER Generator model
def build_deep_generator(latent_dim, output_dim):
    model = Sequential(name="Deep_Generator")
    model.add(Dense(256, input_dim=latent_dim)) # Start wider
    model.add(LeakyReLU(alpha=0.2)) # Use standard LeakyReLU alpha or adjust
    model.add(BatchNormalization(momentum=0.8)) # Default momentum is 0.99
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024)) # Add another deeper layer
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_dim, activation='sigmoid')) # Output layer scaled to [0, 1]
    print("\n--- Deep Generator Summary ---")
    model.summary()
    return model

# Define the DEEPER Discriminator model
def build_deep_discriminator(input_dim):
    model = Sequential(name="Deep_Discriminator")
    model.add(Dense(512, input_dim=input_dim)) # Start wider
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(DROPOUT_RATE)) # Add dropout for regularization
    model.add(Dense(512)) # Deeper layer
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(256)) # Another layer
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(1, activation='sigmoid')) # Output probability (real/fake)
    print("\n--- Deep Discriminator Summary ---")
    model.summary()
    return model

# --- Build and Compile ---
print("Building and compiling models...")
generator = build_deep_generator(LATENT_DIM, num_features)
discriminator = build_deep_discriminator(num_features)

# Compile Discriminator
# Use Adam optimizer with recommended GAN parameters
discriminator_optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=discriminator_optimizer,
                      metrics=['accuracy'])

# Build the combined GAN model (Generator -> Discriminator)
discriminator.trainable = False # Freeze discriminator weights for GAN training

# Define GAN input and output using Keras Functional API
gan_input = Input(shape=(LATENT_DIM,))
generated_sample = generator(gan_input)
gan_output = discriminator(generated_sample)

# Define and compile the combined GAN model
gan = Model(gan_input, gan_output, name="Combined_GAN")
gan_optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
gan.compile(loss='binary_crossentropy', optimizer=gan_optimizer)
print("\n--- Combined GAN Summary ---")
gan.summary()


# --- Training the GAN ---
def train(epochs, batch_size, real_data, latent_dim, generator, discriminator, gan,
          generator_model_path='generator_model.h5',
          discriminator_model_path='discriminator_model.h5'):

    num_samples = real_data.shape[0]
    if num_samples == 0:
        print("Error: No data available for training.")
        return
    if batch_size > num_samples:
        print(f"Warning: Batch size ({batch_size}) > number of samples ({num_samples}). Adjusting batch size.")
        batch_size = num_samples

    half_batch = max(1, int(batch_size / 2))

    # Adversarial ground truths
    valid = np.ones((half_batch, 1))
    fake = np.zeros((half_batch, 1))
    valid_gan = np.ones((batch_size, 1)) # For generator loss (full batch)

    print("\n--- Starting GAN Training ---")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Half Batch: {half_batch}")

    for epoch in range(epochs):
        # --- Train Discriminator ---
        # Select a random half batch of real samples
        idx = np.random.randint(0, num_samples, half_batch)
        real_samples = real_data[idx]

        # Generate a half batch of new fake samples
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_samples = generator.predict(noise, verbose=0) # Use predict, verbose=0 for less output

        # Train the discriminator (real classified as ones and fake as zeros)
        d_loss_real = discriminator.train_on_batch(real_samples, valid)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) # Average loss

        # --- Train Generator ---
        # Sample noise for a full batch
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Train the generator (via the combined GAN model)
        # The generator wants the discriminator to label the generated samples as valid (1)
        g_loss = gan.train_on_batch(noise, valid_gan)

        # --- Log Progress ---
        if (epoch + 1) % 100 == 0 or epoch == 0: # Print every 100 epochs and the first epoch
            print(f"Epoch {epoch+1}/{epochs} -- [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

        # Optional: Save models periodically
        # if (epoch + 1) % 1000 == 0:
        #     generator.save(f"generator_epoch_{epoch+1}.h5")

    # --- Save Final Models ---
    try:
        generator.save(generator_model_path)
        print(f"\nGenerator model saved to {generator_model_path}")
        discriminator.save(discriminator_model_path)
        print(f"Discriminator model saved to {discriminator_model_path}")
    except Exception as e:
        print(f"Error saving models: {e}")


# --- Generate Synthetic Samples ---
def generate_synthetic_can_messages(loaded_generator, num_samples, latent_dim,
                                    max_time, max_dlc, max_id, max_rf,
                                    payload_length=PAYLOAD_LENGTH):
    """Generates synthetic CAN messages using the trained generator."""
    print(f"\nGenerating {num_samples} synthetic samples...")
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_data_normalized = loaded_generator.predict(noise, batch_size=BATCH_SIZE, verbose=0) # Use batch size for prediction

    synthetic_messages = []
    for i, sample in enumerate(generated_data_normalized):
        # De-normalize features
        timeinterval_norm = sample[0]
        id_norm = sample[1]
        rf_norm = sample[2]
        dlc_norm = sample[3]
        payload_norm = sample[4:] # The rest is payload

        # De-normalize and handle potential edge cases/clamping
        timeinterval = max(0.0, timeinterval_norm * max_time)

        can_id_int = max(0, int(round(id_norm * max_id)))
        can_id_hex = hex(can_id_int).lstrip('0x').upper().zfill(3) # Pad ID

        # Clamp RemoteFrame to 0 or 1
        remote_frame_int = max(0, min(1, int(round(rf_norm * max_rf))))
        remote_frame_str = str(remote_frame_int)

        # Clamp DLC between 0 and payload_length (usually 8)
        dlc = max(0, min(payload_length, int(round(dlc_norm * max_dlc))))

        # Clip payload bytes to be valid [0, 255] integers
        payload_int = np.clip(np.round(payload_norm * 255), 0, 255).astype(int)

        # Format payload bytes based on the *actual generated DLC*
        payload_hex_list = [f'{byte:02X}' for byte in payload_int[:dlc]] # Only take 'dlc' bytes
        payload_str = ''.join(payload_hex_list)

        synthetic_messages.append({
            'Index': i,
            'ID': can_id_hex,
            'RemoteFrame': remote_frame_str,
            'DLC': dlc,
            'Payload': payload_str,
            'TimeInterval': timeinterval
        })

    print(f"Generation complete.")
    return pd.DataFrame(synthetic_messages)

# --- Main Execution ---

# Optional: Enable eager execution for easier debugging (can slow down training)
# tf.config.run_functions_eagerly(True)
# print("Eager execution enabled.")

# Train the GAN with the deeper models
train(EPOCHS, BATCH_SIZE, real_data, LATENT_DIM, generator, discriminator, gan,
      GENERATOR_MODEL_PATH, DISCRIMINATOR_MODEL_PATH)

# Load the saved *deep* generator model
print(f"\nLoading generator model from {GENERATOR_MODEL_PATH}")
try:
    # Standard layers should load okay, if custom objects were used, specify them
    loaded_generator = load_model(GENERATOR_MODEL_PATH)
    print("Generator loaded successfully.")
except Exception as e:
    print(f"Error loading generator model: {e}")
    exit()

# Generate synthetic CAN messages using the loaded deep model
num_synthetic_samples = 5000  # You can adjust the number of samples
synthetic_df = generate_synthetic_can_messages(
    loaded_generator, num_synthetic_samples, LATENT_DIM,
    max_time_global, max_dlc_global, max_id_global, max_rf_global,
    PAYLOAD_LENGTH
)

# Saving the generated DataFrame to a CSV file
print(f"Saving generated data to {OUTPUT_CSV}...")
try:
    synthetic_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Generated synthetic data saved successfully.")
except Exception as e:
    print(f"Error saving generated data: {e}")


# Print a sample of the generated synthetic data
print("\nGenerated Synthetic Fuzzy Attack CAN Messages (Sample from Deep GAN):")
print(synthetic_df.head())

print("\n--- Deep GAN Process Finished ---")