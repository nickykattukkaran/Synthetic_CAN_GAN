import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Ensure TensorFlow is installed and Keras is accessible (usually tf.keras)
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Embedding, Concatenate, Flatten, Reshape
from keras.optimizers import Adam
import os # Added for checking file existence

# --- Constants and Configuration ---
PAYLOAD_LENGTH = 8
LATENT_DIM = 100
EMBEDDING_DIM = 10 # Dimension for embedding the class label
NUM_CLASSES = 2 # 0: attack-free, 1: fuzzy attack
EPOCHS = 10000 # Adjust as needed based on convergence
BATCH_SIZE = 100 # Adjust as needed
GENERATOR_MODEL_PATH = 'cgan_generator_model.h5'
DISCRIMINATOR_MODEL_PATH = 'cgan_discriminator_model.h5'
SYNTHETIC_ATTACK_FREE_PATH = 'synthetic_attackfree_generated_cgan.csv'
SYNTHETIC_FUZZY_PATH = 'synthetic_fuzzy_generated_cgan.csv'
ATTACK_FREE_CSV = 'Attackfree_for_gen.csv'
FUZZY_CSV = 'Fuzzy_for_gen.csv' # Make sure this filename matches your fuzzy attack data file

# --- Data Loading and Preparation ---

def hex_to_int_array(hex_string, length=PAYLOAD_LENGTH):
    """Converts a space-separated hex string to a numpy array of integers."""
    default_payload = np.zeros(length, dtype=int)
    if isinstance(hex_string, str):
        try:
            hex_values = hex_string.strip().split(' ')
            # Handle cases where split might produce empty strings if multiple spaces exist
            hex_values = [h for h in hex_values if h]
            # Ensure hex value is not empty before conversion
            int_array = np.array([int(h, 16) for h in hex_values if len(h)>0])
             # Pad or truncate
            if len(int_array) < length:
                # Pad with 0 for consistency
                padded_array = np.pad(int_array, (0, length - len(int_array)), 'constant', constant_values=0)
            else:
                padded_array = int_array[:length] # Truncate
            return padded_array
        except ValueError:
            print(f"Warning: Could not parse hex string: '{hex_string}'. Using default payload.")
            return default_payload
        except Exception as e:
             print(f"Warning: Unexpected error parsing hex string '{hex_string}': {e}. Using default payload.")
             return default_payload
    elif pd.isna(hex_string):
        # Use default for missing values
        return default_payload
    else:
        print(f"Warning: Unexpected data type in Payload: {type(hex_string)}, value: {hex_string}. Using default payload.")
        return default_payload

def safe_hex_to_int(hex_str):
    """Safely converts hex string (potentially read as float/int) to int."""
    if isinstance(hex_str, str):
        try:
            # Handle potential '0x' prefix and remove spaces before converting
            return int(hex_str.replace(' ', ''), 16)
        except ValueError:
            print(f"Warning: Invalid hex string in ID/RemoteFrame: '{hex_str}'. Using 0.")
            return 0
    try:
        # Try converting from float/int if it was parsed as such
        return int(float(hex_str))
    except (ValueError, TypeError):
        # Handle cases where conversion fails
        print(f"Warning: Invalid numeric value in ID/RemoteFrame: {hex_str}. Using 0.")
        return 0

def load_and_preprocess_data(filepath, label):
    """Loads data, assigns label, applies initial hex conversion."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: Data file not found at {filepath}")
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Initial rows loaded: {len(df)}")

    # Keep only necessary columns if others exist
    required_cols = ['TimeInterval', 'ID', 'RemoteFrame', 'DLC', 'Payload']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
         raise ValueError(f"Missing required columns in {filepath}: {missing_cols}")
    df = df[required_cols]

    # Basic check for data types before processing
    # Convert to numeric, coercing errors to NaN
    df['TimeInterval'] = pd.to_numeric(df['TimeInterval'], errors='coerce')
    df['DLC'] = pd.to_numeric(df['DLC'], errors='coerce')

    # Report NaNs introduced by coercion
    if df['TimeInterval'].isna().any():
        print(f"Warning: Found {df['TimeInterval'].isna().sum()} NaN(s) in TimeInterval after coercion in {filepath}. Rows will be dropped.")
    if df['DLC'].isna().any():
         print(f"Warning: Found {df['DLC'].isna().sum()} NaN(s) in DLC after coercion in {filepath}. Rows will be dropped.")

    # Drop rows with NaN in critical numeric columns after coercion
    initial_rows = len(df)
    df.dropna(subset=['TimeInterval', 'DLC'], inplace=True)
    if len(df) < initial_rows:
         print(f"Dropped {initial_rows - len(df)} rows due to NaNs in TimeInterval/DLC from {filepath}.")

    # Apply payload conversion carefully
    df['Payload_Numeric'] = df['Payload'].apply(lambda x: hex_to_int_array(x, PAYLOAD_LENGTH))

    df['Label'] = label
    print(f"Finished initial processing for {filepath}. Rows remaining: {len(df)}")
    return df

# Load datasets
try:
    df_attack_free = load_and_preprocess_data(ATTACK_FREE_CSV, 0)
    df_fuzzy = load_and_preprocess_data(FUZZY_CSV, 1)
except FileNotFoundError as e:
    print(e)
    exit() # Exit if files are not found
except ValueError as e:
    print(e)
    exit() # Exit if columns are missing


# Combine datasets
data = pd.concat([df_attack_free, df_fuzzy], ignore_index=True)
print(f"\nTotal combined data rows before shuffling: {len(data)}")
data = data.sample(frac=1).reset_index(drop=True) # Shuffle combined data
print(f"Total combined data rows after shuffling: {len(data)}")


# --- Global Normalization and Final Preprocessing ---
def normalize_and_finalize(df):
    """Applies normalization based on the combined dataset."""
    if df.empty:
        raise ValueError("Cannot normalize an empty DataFrame.")

    # Calculate max values *after* combining, cleaning, and shuffling
    max_timeinterval = df['TimeInterval'].max()
    max_dlc = df['DLC'].max()

    df['ID_Int'] = df['ID'].apply(safe_hex_to_int)
    # Ensure max_id is at least 1 to avoid division by zero if all IDs are 0
    max_id = df['ID_Int'].max() if not df['ID_Int'].empty else 1
    max_id = max(1, max_id)


    df['RemoteFrame_Int'] = df['RemoteFrame'].apply(safe_hex_to_int)
     # Ensure max_rf is at least 1, usually it's just 0 or 1
    max_rf = df['RemoteFrame_Int'].max() if not df['RemoteFrame_Int'].empty else 1
    max_rf = max(1, max_rf)


    print(f"\nMax values for normalization:")
    print(f"  Max TimeInterval: {max_timeinterval}")
    print(f"  Max DLC: {max_dlc}")
    print(f"  Max ID (int): {max_id}")
    print(f"  Max RemoteFrame (int): {max_rf}")

    # Apply Normalization (handle potential division by zero using calculated max values)
    df['TimeInterval_Normalized'] = df['TimeInterval'] / max_timeinterval if max_timeinterval > 0 else 0
    df['DLC_Normalized'] = df['DLC'] / max_dlc if max_dlc > 0 else 0
    df['ID_Normalized'] = df['ID_Int'] / max_id # max_id is guaranteed >= 1
    df['RemoteFrame_Normalized'] = df['RemoteFrame_Int'] / max_rf # max_rf is guaranteed >= 1


    # Normalize Payload
    # Stack the arrays, normalize
    payload_matrix = np.stack(df['Payload_Numeric'].values)
    payload_normalized = payload_matrix / 255.0  # Normalize byte values to [0, 1]

    # Combine processed features
    processed_features = df[['TimeInterval_Normalized', 'ID_Normalized', 'RemoteFrame_Normalized', 'DLC_Normalized']].values
    final_data = np.concatenate([processed_features, payload_normalized], axis=1)

    labels = df['Label'].values

    return final_data, labels, max_timeinterval, max_dlc, max_id, max_rf

# Apply final preprocessing and normalization
try:
    real_data, real_labels, max_time_global, max_dlc_global, max_id_global, max_rf_global = normalize_and_finalize(data)
except ValueError as e:
    print(e)
    exit() # Exit if normalization fails (e.g., empty dataframe)


num_features = real_data.shape[1] # Number of features excluding label (Time, ID, RF, DLC, Payload bytes)
print(f"Number of features per sample: {num_features}")
print(f"Shape of processed data: {real_data.shape}")
print(f"Shape of labels: {real_labels.shape}")

# --- CGAN Model Definitions ---

# Generator
def build_generator(latent_dim, num_classes, embedding_dim, output_dim):
    # Latent space input
    noise_input = Input(shape=(latent_dim,), name="Noise_Input")

    # Label input
    label_input = Input(shape=(1,), dtype='int32', name="Label_Input")
    # Embed the label
    label_embedding = Embedding(num_classes, embedding_dim, name="Label_Embedding")(label_input)
    label_embedding = Flatten(name="Flatten_Embedding")(label_embedding) # Flatten embedding output

    # Combine noise and label embedding
    merged_input = Concatenate(name="Concatenate_Noise_Label")([noise_input, label_embedding])

    # Generator network
    model = Dense(128, name="G_Dense_1")(merged_input)
    model = LeakyReLU(alpha=0.01, name="G_LeakyRelu_1")(model)
    model = BatchNormalization(name="G_BatchNorm_1")(model)
    model = Dense(256, name="G_Dense_2")(model)
    model = LeakyReLU(alpha=0.01, name="G_LeakyRelu_2")(model)
    model = BatchNormalization(name="G_BatchNorm_2")(model)
    # Output layer - Sigmoid activation for normalized features [0, 1]
    output = Dense(output_dim, activation='sigmoid', name="G_Output")(model)

    # Define model
    generator_model = Model([noise_input, label_input], output, name="Generator")
    print("\n--- Generator Summary ---")
    generator_model.summary()
    return generator_model

# Discriminator
def build_discriminator(input_dim, num_classes, embedding_dim):
    # Feature input
    feature_input = Input(shape=(input_dim,), name="Feature_Input")

    # Label input
    label_input = Input(shape=(1,), dtype='int32', name="Label_Input")
    # Embed the label
    label_embedding = Embedding(num_classes, embedding_dim, name="Label_Embedding")(label_input)
    label_embedding = Flatten(name="Flatten_Embedding")(label_embedding) # Flatten embedding output

    # Combine features and label embedding
    merged_input = Concatenate(name="Concatenate_Feature_Label")([feature_input, label_embedding])

    # Discriminator network
    model = Dense(256, name="D_Dense_1")(merged_input)
    model = LeakyReLU(alpha=0.01, name="D_LeakyRelu_1")(model)
    model = Dense(128, name="D_Dense_2")(model)
    model = LeakyReLU(alpha=0.01, name="D_LeakyRelu_2")(model)
    # Output layer - Sigmoid for probability (real/fake)
    output = Dense(1, activation='sigmoid', name="D_Output")(model)

    # Define model
    discriminator_model = Model([feature_input, label_input], output, name="Discriminator")
    print("\n--- Discriminator Summary ---")
    discriminator_model.summary()
    return discriminator_model

# --- Build and Compile Models ---
generator = build_generator(LATENT_DIM, NUM_CLASSES, EMBEDDING_DIM, num_features)
discriminator = build_discriminator(num_features, NUM_CLASSES, EMBEDDING_DIM)

# Compile Discriminator
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=0.0002, beta_1=0.5), # Explicit learning rate
                      metrics=['accuracy'])

# Build the combined CGAN model (Generator -> Discriminator)
discriminator.trainable = False # Freeze discriminator weights for GAN training

# CGAN inputs
noise_input_cgan = Input(shape=(LATENT_DIM,), name="Noise_Input_CGAN")
label_input_cgan = Input(shape=(1,), dtype='int32', name="Label_Input_CGAN")

# Generate fake samples using the generator part
generated_samples = generator([noise_input_cgan, label_input_cgan])

# Discriminator determines validity of generated samples given the label
validity = discriminator([generated_samples, label_input_cgan])

# Define combined CGAN model
cgan = Model([noise_input_cgan, label_input_cgan], validity, name="CGAN")
print("\n--- Combined CGAN Summary ---")
cgan.summary()

# Compile CGAN (only trains generator weights due to discriminator.trainable = False)
cgan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# --- CGAN Training ---
def train_cgan(epochs, batch_size, real_data, real_labels, num_classes, latent_dim,
               generator_path='cgan_generator_model.h5',
               discriminator_path='cgan_discriminator_model.h5'):
    num_samples = real_data.shape[0]
    if num_samples == 0:
        print("Error: No data available for training after preprocessing.")
        return
    if batch_size > num_samples:
        print(f"Warning: Batch size ({batch_size}) is larger than the number of samples ({num_samples}). Adjusting batch size to {num_samples}.")
        batch_size = num_samples

    half_batch = max(1, int(batch_size / 2)) # Ensure half_batch is at least 1

    # Adversarial ground truths
    valid = np.ones((half_batch, 1))
    fake = np.zeros((half_batch, 1))
    # Labels for generator training step (full batch)
    valid_gan = np.ones((batch_size, 1))

    print("\n--- Starting CGAN Training ---")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Half Batch: {half_batch}")

    for epoch in range(epochs):
        # --- Train Discriminator ---

        # Select a random half batch of real samples and their labels
        idx = np.random.randint(0, num_samples, half_batch)
        real_samples_batch = real_data[idx]
        real_labels_batch = real_labels[idx].reshape(-1, 1) # Ensure correct shape

        # Sample noise and generate a half batch of fake samples
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        # Generate fake samples with random labels for this batch
        sampled_labels = np.random.randint(0, num_classes, half_batch).reshape(-1, 1)
        # Use predict_on_batch for potentially faster inference if needed, or predict
        fake_samples_batch = generator.predict([noise, sampled_labels], verbose=0)

        # Train the discriminator (real classified as ones and fake as zeros)
        # Ensure inputs are lists as the model expects multiple inputs
        d_loss_real = discriminator.train_on_batch([real_samples_batch, real_labels_batch], valid)
        d_loss_fake = discriminator.train_on_batch([fake_samples_batch, sampled_labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # --- Train Generator ---

        # Sample noise and random labels for a full batch
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_labels_gan = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

        # Train the generator (via the combined CGAN model) by tricking the discriminator
        # Ensure inputs are lists as the model expects multiple inputs
        g_loss = cgan.train_on_batch([noise, sampled_labels_gan], valid_gan)

        # --- Log Progress ---
        if (epoch + 1) % 100 == 0 or epoch == 0: # Print every 100 epochs and the first epoch
             print(f"Epoch {epoch+1}/{epochs} -- [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

        # Optional: Save models periodically (e.g., every 1000 epochs)
        # if (epoch + 1) % 1000 == 0:
        #     try:
        #         generator.save(f"cgan_generator_epoch_{epoch+1}.h5")
        #         # It's often useful to save discriminator too, making sure trainable=True first if needed elsewhere
        #         # discriminator.save(f"cgan_discriminator_epoch_{epoch+1}.h5")
        #     except Exception as e:
        #         print(f"Error saving model at epoch {epoch+1}: {e}")


    # --- Save Final Models ---
    try:
        generator.save(generator_path)
        print(f"\nTraining complete. Generator saved to {generator_path}")
        # Save discriminator state as well (optional but good practice)
        # If you load the discriminator later, compile it again.
        discriminator.save(discriminator_path)
        print(f"Discriminator saved to {discriminator_path}")
    except Exception as e:
        print(f"Error saving final models: {e}")

# --- Synthetic Data Generation (with RemoteFrame fix) ---
def generate_synthetic_can_messages(loaded_generator, num_samples, label, num_classes, latent_dim,
                                     max_time, max_dlc, max_id, max_rf, payload_length=PAYLOAD_LENGTH):
    """Generates synthetic CAN messages for a specific label."""
    print(f"\nGenerating {num_samples} synthetic samples for label {label}...")

    # Prepare input for the generator
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    # Create label input array for the desired class
    label_input = np.full((num_samples, 1), label, dtype='int32') # Ensure dtype is int32

    # Generate data using the loaded generator
    # Use predict for larger batches, predict_on_batch might be slightly faster sometimes
    generated_data_normalized = loaded_generator.predict([noise, label_input], verbose=0, batch_size=BATCH_SIZE) # Use batch_size for prediction too

    synthetic_messages = []
    for i, sample in enumerate(generated_data_normalized):
        # De-normalize features
        timeinterval_norm = sample[0]
        id_norm = sample[1]
        rf_norm = sample[2] # <-- Normalized RemoteFrame value
        dlc_norm = sample[3]
        payload_norm = sample[4:] # The rest is payload

        # Handle potential edge cases during de-normalization
        timeinterval = max(0.0, timeinterval_norm * max_time) # Ensure non-negative time

        can_id_int = max(0, int(round(id_norm * max_id)))

        # --- CORRECTED RemoteFrame Handling ---
        # De-normalize using max_rf (which should be 1 if input was only 0/1)
        remote_frame_float = rf_norm * max_rf
        # Round to nearest integer (should be 0 or 1)
        remote_frame_int = int(round(remote_frame_float))
        # Explicitly clamp to 0 or 1, as these are the only valid standard values
        remote_frame_int = max(0, min(1, remote_frame_int))
        # Directly convert the integer 0 or 1 to string '0' or '1'
        remote_frame_str = str(remote_frame_int)
        # --- End Correction ---

        # Clamp DLC between 0 and PAYLOAD_LENGTH (usually 8)
        dlc = max(0, min(payload_length, int(round(dlc_norm * max_dlc))))

        # Clip payload bytes to be valid [0, 255] integers
        payload_int = np.clip(np.round(payload_norm * 255), 0, 255).astype(int)

        # Format output
        can_id_hex = hex(can_id_int).lstrip('0x').upper().zfill(3) # Pad ID if needed

        # Format payload bytes based on the *actual generated DLC*
        payload_hex_list = [f'{byte:02X}' for byte in payload_int[:dlc]] # Only take 'dlc' bytes
        payload_str = ''.join(payload_hex_list)


        synthetic_messages.append({
            'Index': i,
            'ID': can_id_hex,
            'RemoteFrame': remote_frame_str, # <-- Use the corrected string '0' or '1'
            'DLC': dlc,
            'Payload': payload_str,
            'TimeInterval': timeinterval,
            #'GeneratedLabel': label # Add label to know what was generated
        })

    print(f"Generation for label {label} complete.")
    return pd.DataFrame(synthetic_messages)


# --- Main Execution ---

# Optional: Enable eager execution for easier debugging if needed (can slow down training)
# tf.config.run_functions_eagerly(True)
# print("Eager execution enabled.")

# Train the CGAN
train_cgan(EPOCHS, BATCH_SIZE, real_data, real_labels, NUM_CLASSES, LATENT_DIM,
           GENERATOR_MODEL_PATH, DISCRIMINATOR_MODEL_PATH)

# Load the trained generator model
print(f"\nLoading generator model from {GENERATOR_MODEL_PATH}")
try:
    # Keras functional models with standard layers usually load without custom_objects
    loaded_generator = load_model(GENERATOR_MODEL_PATH)
    print("Generator loaded successfully.")
    loaded_generator.summary() # Print summary to verify loaded model structure
except Exception as e:
    print(f"Error loading generator model: {e}")
    print("Ensure the model file exists and was saved correctly.")
    print("If using custom layers/objects not recognized, add them to custom_objects argument in load_model.")
    exit()


# Generate synthetic CAN messages for each class
num_synthetic_samples = 5000 # Number of samples to generate per class

# --- Generate Attack-Free (label 0) ---
try:
    synthetic_df_attack_free = generate_synthetic_can_messages(
        loaded_generator, num_synthetic_samples, 0, NUM_CLASSES, LATENT_DIM,
        max_time_global, max_dlc_global, max_id_global, max_rf_global, PAYLOAD_LENGTH
    )
    # Save the generated data
    synthetic_df_attack_free.to_csv(SYNTHETIC_ATTACK_FREE_PATH, index=False)
    print(f"\nGenerated Attack-Free data saved to {SYNTHETIC_ATTACK_FREE_PATH}")
    print("\nGenerated Attack-Free CAN Messages (Sample):")
    print(synthetic_df_attack_free.head())
except Exception as e:
    print(f"Error generating or saving synthetic attack-free data: {e}")


# --- Generate Fuzzy Attack (label 1) ---
try:
    synthetic_df_fuzzy = generate_synthetic_can_messages(
        loaded_generator, num_synthetic_samples, 1, NUM_CLASSES, LATENT_DIM,
        max_time_global, max_dlc_global, max_id_global, max_rf_global, PAYLOAD_LENGTH
    )
    # Save the generated data
    synthetic_df_fuzzy.to_csv(SYNTHETIC_FUZZY_PATH, index=False)
    print(f"\nGenerated Fuzzy Attack data saved to {SYNTHETIC_FUZZY_PATH}")
    print("\nGenerated Fuzzy Attack CAN Messages (Sample):")
    print(synthetic_df_fuzzy.head())
except Exception as e:
    print(f"Error generating or saving synthetic fuzzy attack data: {e}")


print("\n--- CGAN Process Finished ---")