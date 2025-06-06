import pandas as pd
import numpy as np
import os # Added for checking file existence
from sklearn.model_selection import train_test_split
# Updated Keras imports (assuming TensorFlow backend)
from keras.models import Sequential, Model, load_model # Use Model for combined GAN
from keras.layers import Dense, BatchNormalization, LeakyReLU, Input, Dropout # Add Input and Dropout
# *** FIX: Import the legacy Adam optimizer to resolve the KeyError ***
from keras.optimizers.legacy import Adam
import tensorflow as tf

# --- Configuration ---
INPUT_CSV_FILE = 'Impersonation_for_gen.csv'
# Output filenames for the deep models
GENERATOR_MODEL_PATH = 'generator_Impersonation_deep_model.h5'
DISCRIMINATOR_MODEL_PATH = 'discriminator_Impersonation_deep_model.h5'
OUTPUT_CSV_PATH = 'synthetic_Impersonation_deep_generated.csv'

# Model/Training Parameters
PAYLOAD_LENGTH = 8  # Fixed length for payload byte array
LATENT_DIM = 100    # Dimension of the random noise vector
EPOCHS = 1000       # Adjust as needed
BATCH_SIZE = 64       # Adjust based on memory/performance
LEARNING_RATE = 0.0002 # Common learning rate for GANs
ADAM_BETA_1 = 0.5      # Common beta_1 value for GANs
NUM_SYNTHETIC_SAMPLES = 5000 # Number of samples to generate after training

# --- Load Dataset ---
print(f"--- Loading Data from {INPUT_CSV_FILE} ---")
try:
    data = pd.read_csv(INPUT_CSV_FILE)
    print(f"Successfully loaded {len(data)} rows.")
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_CSV_FILE}' not found.")
    print("Please ensure the file is in the correct directory.")
    exit() # Exit if the primary data file is missing
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()
print("-------------------------------------------\n")


# --- Helper Functions (Robust Payload/Hex Conversion) ---

def hex_to_int_array(hex_string, target_length=PAYLOAD_LENGTH):
    """Converts space-separated hex string to a padded/truncated int array."""
    default_payload = [0] * target_length
    if isinstance(hex_string, str):
        try:
            hex_values = hex_string.strip().split(' ')
            hex_values = [h for h in hex_values if h] # Remove empty strings from multiple spaces
            if not hex_values: return default_payload
            int_array = [int(h, 16) for h in hex_values]
            if len(int_array) < target_length:
                # Pad with 0
                padded_array = np.pad(int_array, (0, target_length - len(int_array)), 'constant', constant_values=0)
            else:
                padded_array = int_array[:target_length]
            return padded_array
        except ValueError:
            # Optional: Print warning only once or less frequently if too verbose
            # print(f"Warning: Could not parse hex string: '{hex_string}'. Using default.")
            return default_payload
        except Exception as e:
            # Optional: Print warning
            # print(f"Warning: Unexpected error parsing hex string '{hex_string}': {e}. Using default.")
            return default_payload
    elif pd.isna(hex_string):
        return default_payload
    else:
        # Optional: Print warning
        # print(f"Warning: Unexpected data type in Payload: {type(hex_string)}, value: {hex_string}. Using default.")
        return default_payload

def safe_hex_to_int(hex_str):
    """Safely converts hex string (or potential float/int) to integer."""
    if isinstance(hex_str, str):
        try:
            # Remove potential '0x' prefix and ensure it's a valid hex
            hex_str_cleaned = hex_str.lower().strip()
            if hex_str_cleaned.startswith('0x'):
                 hex_str_cleaned = hex_str_cleaned[2:]
            if not hex_str_cleaned: # Handle empty string after removing 0x or if input was just spaces
                return 0
            return int(hex_str_cleaned, 16)
        except ValueError:
            # Optional: Print warning
            # print(f"Warning: Invalid hex value format: '{hex_str}'. Using 0.")
            return 0
        except Exception as e:
             # Optional: Print warning
             # print(f"Warning: Error converting hex '{hex_str}': {e}. Using 0.")
             return 0
    try:
        # Handle cases where it might already be parsed as float or int
        # Ensure conversion to float first to handle scientific notation if present
        return int(float(hex_str))
    except (ValueError, TypeError, OverflowError):
        # Optional: Print warning
        # print(f"Warning: Invalid or non-string value: {hex_str}. Using 0.")
        return 0

# Apply Payload conversion
print("--- Preprocessing Data ---")
print("Processing Payload column...")
data['Payload_Numeric'] = data['Payload'].apply(lambda x: hex_to_int_array(x, PAYLOAD_LENGTH))

# --- Preprocessing Function ---
def preprocess_data(df):
    print("Applying further preprocessing steps...")
    df_proc = df.copy()

    # Convert hex IDs and RemoteFrames AFTER handling Payload
    print("Processing ID and RemoteFrame...")
    df_proc['ID_Int'] = df_proc['ID'].apply(safe_hex_to_int)
    # *** Use safe_hex_to_int for RemoteFrame as well ***
    df_proc['RemoteFrame_Int'] = df_proc['RemoteFrame'].apply(safe_hex_to_int)

    # Convert DLC and TimeInterval to numeric, handling errors robustly
    print("Processing DLC and TimeInterval...")
    df_proc['TimeInterval'] = pd.to_numeric(df_proc['TimeInterval'], errors='coerce')
    median_time = df_proc['TimeInterval'].median() # Use median for filling missing TimeIntervals
    df_proc['TimeInterval'].fillna(median_time if not pd.isna(median_time) else 0.0, inplace=True)

    df_proc['DLC'] = pd.to_numeric(df_proc['DLC'], errors='coerce').fillna(0).astype(int)
    df_proc['DLC'] = df_proc['DLC'].clip(0, PAYLOAD_LENGTH) # Clip DLC to valid range

    # Calculate Max values for normalization (using the current dataset)
    print("Calculating normalization bounds...")
    max_timeinterval = df_proc['TimeInterval'].max()
    max_dlc = df_proc['DLC'].max() # Should be PAYLOAD_LENGTH or less after clipping
    max_id = df_proc['ID_Int'].max()
    max_rf = df_proc['RemoteFrame_Int'].max() # Max value found in RemoteFrame column
    print(f"  Max values: TimeInterval={max_timeinterval:.4f}, DLC={max_dlc}, ID={max_id:X}, RF={max_rf:X}")

    # Handle potential division by zero if max values are 0 (or very close to 0)
    epsilon = 1e-6 # Small value to avoid division by zero
    max_timeinterval = max(max_timeinterval, epsilon)
    max_dlc = max(max_dlc, 1.0) # DLC max should be at least 1 if data exists
    max_id = max(max_id, epsilon)
    max_rf = max(max_rf, 1.0) # RF max should be at least 1 if data exists

    # Normalization to [0, 1] range
    print("Normalizing features...")
    df_proc['TimeInterval_Normalized'] = df_proc['TimeInterval'] / max_timeinterval
    df_proc['ID_Normalized'] = df_proc['ID_Int'] / max_id
    df_proc['RemoteFrame_Normalized'] = df_proc['RemoteFrame_Int'] / max_rf
    df_proc['DLC_Normalized'] = df_proc['DLC'] / max_dlc

    # Normalize Payload byte values to [0, 1]
    payload_matrix = np.array(df_proc['Payload_Numeric'].tolist())
    payload_normalized = payload_matrix / 255.0

    # Combine normalized features into a single NumPy array
    processed_features = df_proc[['TimeInterval_Normalized', 'ID_Normalized', 'RemoteFrame_Normalized', 'DLC_Normalized']].values
    combined_normalized_data = np.concatenate([processed_features, payload_normalized], axis=1)

    print(f"Preprocessing finished. Output data shape: {combined_normalized_data.shape}")
    print("--------------------------\n")
    return combined_normalized_data, max_timeinterval, max_dlc, max_id, max_rf

# Preprocess the data
processed_data, max_timeinterval_global, max_dlc_global, max_id_global, max_rf_global = preprocess_data(data.copy())

# Prepare data for GAN input
real_data = processed_data
num_features = real_data.shape[1]
print(f"--- Data Ready for GAN ---")
print(f"Number of samples: {real_data.shape[0]}")
print(f"Number of features per sample: {num_features}")
print(f"Latent dimension: {LATENT_DIM}")
print("--------------------------\n")


# --- Define Deeper Models ---

def build_deep_generator(latent_dim, output_dim):
    print("--- Building DEEP Generator ---")
    model = Sequential(name="Deep_Generator")
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())

    # Added Layer 1
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())

    # Added Layer 2
    model.add(Dense(512)) # Increase neurons
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())

    # Added Layer 3
    model.add(Dense(256)) # Decrease back
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())

    # Output layer: 'sigmoid' activation for data normalized to [0, 1]
    model.add(Dense(output_dim, activation='sigmoid'))
    model.summary()
    print("-------------------------------\n")
    return model

def build_deep_discriminator(input_dim):
    print("--- Building DEEP Discriminator ---")
    model = Sequential(name="Deep_Discriminator")
    model.add(Dense(512, input_dim=input_dim)) # Start wider
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.3)) # Add dropout for regularization

    # Added Layer 1
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.3)) # Add dropout

    # Added Layer 2
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.3)) # Add dropout

    # Added Layer 3
    model.add(Dense(64)) # Smaller layer before output
    model.add(LeakyReLU(alpha=0.01))
    # No dropout right before the output layer is common practice

    # Output layer: single neuron predicting probability (real/fake)
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    print("-----------------------------------\n")
    return model

# Build the DEEPER models
generator = build_deep_generator(LATENT_DIM, num_features)
discriminator = build_deep_discriminator(num_features)

# --- Compile Models using LEGACY Adam Optimizer ---
# Use legacy Adam optimizer - this is the fix for the KeyError
print("--- Compiling Models (Using Legacy Adam) ---")
optimizer = Adam(learning_rate=LEARNING_RATE, beta_1=ADAM_BETA_1)

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print("Discriminator compiled.")

# --- Define the combined GAN model ---
# Freeze the discriminator's weights when training the generator via the combined model
discriminator.trainable = False
print("Discriminator weights frozen for GAN training.")

# Use Keras Functional API for the combined model
gan_input = Input(shape=(LATENT_DIM,), name="GAN_Input")
generated_output = generator(gan_input) # Output from generator
gan_output = discriminator(generated_output) # Output from frozen discriminator

# Create and compile the combined GAN model
gan = Model(gan_input, gan_output, name="GAN_Combined")
gan.compile(loss='binary_crossentropy', optimizer=optimizer) # Use the same optimizer instance

print("\nCombined GAN model summary:")
gan.summary()
print("------------------------------------------\n")


# --- Training the GAN ---
def train(epochs, batch_size, real_data,
          generator, discriminator, gan, # Pass models explicitly
          generator_model_path, discriminator_model_path):

    num_samples = real_data.shape[0]
    if num_samples == 0:
        print("Error: No data available for training.")
        return

    # Adjust batch size if dataset is smaller
    if num_samples < batch_size:
        print(f"Warning: Dataset size ({num_samples}) is smaller than batch size ({batch_size}).")
        # Option 1: Adjust batch size (ensure it's at least 2 for half_batch calculation)
        batch_size = max(2, num_samples)
        print(f"Adjusted batch size to: {batch_size}")
        # Option 2: Raise error or skip training
        # raise ValueError("Dataset too small for the specified batch size.")

    half_batch = int(batch_size / 2)
    if half_batch <= 0: # Should not happen with max(2,...) adjustment
         print("Error: Calculated half_batch is zero or less. Check batch_size.")
         return # Or raise error

    batches_per_epoch = num_samples // batch_size
    print(f"\n--- Starting Training (Deep Models) ---")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Batches/Epoch: {batches_per_epoch}")
    if batches_per_epoch == 0:
        print("Warning: batches_per_epoch is zero. Training might not proceed effectively.")

    # Adversarial ground truths
    valid = np.ones((half_batch, 1)) # Labels for real samples for discriminator
    fake = np.zeros((half_batch, 1))  # Labels for fake samples for discriminator
    # Ground truth for generator training (wants discriminator to output 'valid')
    gen_valid = np.ones((batch_size, 1))


    for epoch in range(epochs):
        # Shuffle data indices for each epoch (optional but good practice)
        epoch_indices = np.random.permutation(num_samples)

        for batch_num in range(batches_per_epoch):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of real samples
            # Use shuffled indices for potentially better mini-batch diversity
            batch_start = batch_num * batch_size
            real_idx = epoch_indices[batch_start : batch_start + half_batch]
            # Ensure we don't go out of bounds if last batch is smaller (less likely with half_batch)
            if len(real_idx) != half_batch: continue # Skip if not enough samples for half batch
            real_samples = real_data[real_idx]

            # Generate a half batch of new fake samples
            noise = np.random.normal(0, 1, (half_batch, LATENT_DIM))
            # Use verbose=0 to reduce console spam during training
            fake_samples = generator.predict(noise, verbose=0)

            # Train the discriminator on real and fake samples
            # Ensure discriminator is trainable for this step
            discriminator.trainable = True
            d_loss_real = discriminator.train_on_batch(real_samples, valid)
            d_loss_fake = discriminator.train_on_batch(fake_samples, fake)
            discriminator.trainable = False # Freeze again before training generator
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) # Average loss

            # ---------------------
            #  Train Generator
            # ---------------------

            # Generate a full batch of noise
            noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))

            # Train the generator (via the combined GAN model where discriminator is frozen)
            # The generator learns to make the discriminator output 'valid' (1)
            g_loss = gan.train_on_batch(noise, gen_valid)

        # Print progress (e.g., every 100 epochs) - print based on last batch's loss
        #if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

    # --- Save Models ---
    print("\n--- Saving Trained Models ---")
    try:
        generator.save(generator_model_path)
        print(f"Generator model saved successfully to {generator_model_path}")
        # Save discriminator (optional, but can be useful)
        discriminator.trainable = True # Unfreeze layers before saving if needed
        discriminator.save(discriminator_model_path)
        print(f"Discriminator model saved successfully to {discriminator_model_path}")
    except Exception as e:
        print(f"\nError saving models: {e}")
        print("Models may not have been saved correctly.")

    print("--- Training Finished ---")
    print("-------------------------\n")


# --- Generate synthetic samples using the loaded model ---
def generate_synthetic_can_messages(loaded_generator, num_samples, max_timeinterval, max_dlc, max_id, max_rf, latent_dim, payload_length=PAYLOAD_LENGTH):
    """Generates synthetic CAN messages using the trained generator."""
    print(f"\n--- Generating {num_samples} Synthetic Samples ---")
    if loaded_generator is None:
        print("Error: Generator model not loaded.")
        return pd.DataFrame()

    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    # Use a reasonable batch size for prediction to manage memory
    pred_batch_size = min(BATCH_SIZE * 2, num_samples) # Example prediction batch size
    generated_data_normalized = loaded_generator.predict(noise, batch_size=pred_batch_size, verbose=1)

    synthetic_messages = []
    index_counter = 0

    print("Denormalizing and formatting generated samples...")
    for i, sample in enumerate(generated_data_normalized):
        timeinterval_norm = sample[0]
        id_norm = sample[1]
        rf_norm = sample[2]
        dlc_norm = sample[3]
        payload_norm = sample[4:] # The rest are payload bytes

        # Denormalize
        timeinterval = timeinterval_norm * max_timeinterval

        can_id_float = id_norm * max_id
        # Clamp ID to valid range (e.g., 11-bit: 0x7FF, 29-bit: 0x1FFFFFFF). Using 29-bit max.
        can_id_int = max(0, min(int(round(can_id_float)), 0x1FFFFFFF))
        # Format as hex, padding appropriately (e.g., 3 for std, 8 for ext) - Use 3 for now.
        can_id_hex = f"{can_id_int:03X}" # Pad with leading zeros up to 3 hex digits

        # Denormalize and threshold RemoteFrame (expecting 0 or 1)
        rf_float = rf_norm * max_rf
        remote_frame_int = 1 if rf_float >= 0.5 else 0 # Simple thresholding
        remote_frame_hex = f"{remote_frame_int:X}" # Format as single hex digit ('0' or '1')

        # Denormalize and round DLC, clamping to valid range [0, payload_length]
        dlc_float = dlc_norm * max_dlc
        dlc_int = max(0, min(int(round(dlc_float)), payload_length))
        dlc = dlc_int # Keep DLC as integer

        # Denormalize payload bytes, clip to [0, 255], and convert to hex string
        payload_int = np.clip(np.round(payload_norm * 255), 0, 255).astype(int)
        # Use only the first 'dlc' bytes for the final payload string representation
        actual_payload_bytes = payload_int[:dlc]
        payload_hex_list = [f'{byte:02X}' for byte in actual_payload_bytes] # Pad each byte to 2 hex digits
        payload_str = ' '.join(payload_hex_list) # Space-separated hex bytes

        synthetic_messages.append({
            'Index': index_counter,
            'ID': can_id_hex,
            'RemoteFrame': remote_frame_hex,
            'DLC': dlc,
            'Payload': payload_str,
            'TimeInterval': timeinterval
        })
        index_counter += 1

    print("--- Synthetic sample generation finished ---")
    print("--------------------------------------------\n")
    return pd.DataFrame(synthetic_messages)


# --- Main Execution ---
if __name__ == "__main__":

    # Set training parameters from configuration section
    epochs = EPOCHS
    batch_size = BATCH_SIZE
    generator_model_path = GENERATOR_MODEL_PATH
    discriminator_model_path = DISCRIMINATOR_MODEL_PATH
    output_csv_path = OUTPUT_CSV_PATH
    num_synthetic_samples = NUM_SYNTHETIC_SAMPLES

    # Optional: Control TensorFlow logging verbosity
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress INFO logs

    # Optional: Enable eager execution ONLY for debugging if necessary. Graph mode is faster.
    # tf.config.run_functions_eagerly(False) # Use graph mode (default)

    # --- Train the GAN with deep models ---
    train(epochs, batch_size, real_data,
          generator, discriminator, gan, # Pass the built models
          generator_model_path, discriminator_model_path)

    # --- Load Generator and Generate Final Output ---
    print(f"\n--- Loading Generator and Generating Final Output ---")
    loaded_generator = None # Initialize
    try:
        # Load the saved generator model
        loaded_generator = load_model(generator_model_path)
        print(f"Successfully loaded generator model from {generator_model_path}")

    except FileNotFoundError:
        print(f"\nError: Trained generator model file not found at {generator_model_path}.")
        print("Cannot generate synthetic data. Ensure training completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred during model loading: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for full traceback during debugging

    # Proceed only if generator was loaded successfully
    if loaded_generator:
        try:
            # Generate synthetic CAN messages using the loaded deep model
            synthetic_df = generate_synthetic_can_messages(
                loaded_generator,
                num_synthetic_samples,
                max_timeinterval_global,
                max_dlc_global,
                max_id_global,
                max_rf_global,
                LATENT_DIM, # Pass latent_dim
                PAYLOAD_LENGTH
            )

            if not synthetic_df.empty:
                # Save the generated DataFrame
                synthetic_df.to_csv(output_csv_path, index=False)
                print(f"\nGenerated {len(synthetic_df)} synthetic CAN messages.")
                print(f"Saved synthetic data to: {output_csv_path}")

                # Print a sample of the generated synthetic data
                print("\nGenerated Synthetic Impersonation Attack CAN Messages (Sample - Deep Model):")
                print(synthetic_df.head())
            else:
                print("\nNo synthetic data was generated.")

        except Exception as e:
            print(f"\nAn error occurred during data generation: {e}")
            # import traceback
            # traceback.print_exc() # Uncomment for full traceback

    print("\n--- Script Finished ---")