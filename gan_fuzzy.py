import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
import tensorflow as tf

# Load the dataset
data = pd.read_csv('Fuzzy_for_gen.csv')

# Convert 'Payload' to a numerical representation
def hex_to_int_array(hex_string):
    if isinstance(hex_string, str):
        hex_values = hex_string.split(' ')
        int_array = [int(h, 16) for h in hex_values]
        return int_array
    elif pd.isna(hex_string):
        return [0] * 8  # Or some other appropriate default for missing values
    else:
        print(f"Warning: Unexpected data type in Payload: {type(hex_string)}, value: {hex_string}. Skipping.")
        return [0] * 8  # Or handle the error as needed

data['Payload_Numeric'] = data['Payload'].apply(hex_to_int_array)

# Define sequence length for Payload
PAYLOAD_LENGTH = 8

# Prepare the data for the GAN
def preprocess_data(df):
    # Normalize Timestamp (crude normalization)
    max_timeinterval = df['TimeInterval'].max()
    df['TimeInterval_Normalized'] = df['TimeInterval'] / max_timeinterval if max_timeinterval > 0 else 0

    # Normalize DLC (assuming it's within a small range)
    max_dlc = df['DLC'].max()
    df['DLC_Normalized'] = df['DLC'] / max_dlc if max_dlc > 0 else 0

    # Scale ID
    def safe_hex_to_int(hex_str):
        if isinstance(hex_str, str):
            return int(hex_str, 16)
        try:
            return int(float(hex_str)) # Try converting from float if it was parsed as one
        except (ValueError, TypeError):
            print(f"Warning: Invalid ID value: {hex_str}. Using 0.")
            return 0

    df['ID_Int'] = df['ID'].apply(safe_hex_to_int)
    max_id = df['ID_Int'].max() if not df['ID_Int'].empty else 1
    df['ID_Normalized'] = df['ID_Int'] / max_id

    # Scale RemoteFrame
    def safe_rf_hex_to_int(hex_str):
        if isinstance(hex_str, str):
            return int(hex_str, 16)
        try:
            return int(float(hex_str)) # Try converting from float if it was parsed as one
        except (ValueError, TypeError):
            print(f"Warning: Invalid RemoteFrame value: {hex_str}. Using 0.")
            return 0

    df['RemoteFrame_Int'] = df['RemoteFrame'].apply(safe_rf_hex_to_int)
    max_rf = df['RemoteFrame_Int'].max() if not df['RemoteFrame_Int'].empty else 1
    df['RemoteFrame_Normalized'] = df['RemoteFrame_Int'] / max_rf

    # Pad or truncate Payload_Numeric to a fixed length
    payload_matrix = np.array([np.pad(p, (0, PAYLOAD_LENGTH - len(p)), 'constant', constant_values=1)[:PAYLOAD_LENGTH] for p in df['Payload_Numeric']])
    payload_normalized = payload_matrix / 255.0  # Normalize byte values to [0, 1]

    processed_features = df[['TimeInterval_Normalized', 'ID_Normalized', 'RemoteFrame_Normalized', 'DLC_Normalized']].values
    return np.concatenate([processed_features, payload_normalized], axis=1), max_timeinterval, max_dlc, max_id, max_rf

processed_data, max_timeinterval_global, max_dlc_global, max_id_global, max_rf_global = preprocess_data(data.copy())

# Split data into training and "real" samples for the discriminator
real_data = processed_data
latent_dim = 100
num_features = real_data.shape[1]

# Define the Generator model
def build_generator(latent_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dense(output_dim, activation='sigmoid')) # Output scaled to [0, 1]
    return model

# Define the Discriminator model
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid')) # Output probability (real/fake)
    return model

# Build and compile the models
generator = build_generator(latent_dim, num_features)
discriminator = build_discriminator(num_features)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Define the combined GAN model
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training the GAN
def train(epochs, batch_size, real_data, generator_model_path='generator_model.h5', discriminator_model_path='discriminator_model.h5'):
    half_batch = int(batch_size / 2)
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, real_data.shape[0], half_batch)
        real_samples = real_data[idx]
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_samples = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_samples, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_samples, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss[0]:.4f}, D Accuracy: {100*d_loss[1]:.2f}%, G Loss: {g_loss:.4f}")

    # Save the trained generator model
    generator.save(generator_model_path)
    print(f"Generator model saved to {generator_model_path}")
    # Save the trained discriminator model (optional)
    discriminator.save(discriminator_model_path)
    print(f"Discriminator model saved to {discriminator_model_path}")

# Generate synthetic samples using the loaded model
def generate_synthetic_can_messages(loaded_generator, num_samples, max_timeinterval, max_dlc, max_id, max_rf, payload_length=8):
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_data_normalized = loaded_generator.predict(noise)

    synthetic_messages = []

    for i, sample in enumerate(generated_data_normalized):
        timeinterval_norm = sample[0]
        id_norm = sample[1]
        rf_norm = sample[2]
        dlc_norm = sample[3]
        payload_norm = sample[4:]

        timeinterval = timeinterval_norm * max_timeinterval
        can_id = hex(int(id_norm * max_id)).lstrip('0x').zfill(3)
        remote_frame = hex(int(rf_norm * max_rf)).lstrip('0x').zfill(1)
        dlc = int(dlc_norm * max_dlc)
        payload_int = np.round(payload_norm * 255).astype(int)
        payload = ''.join([f'{i:02x}' for i in payload_int])

        synthetic_messages.append({
            'Index': i,
            'ID': can_id,
            'RemoteFrame': remote_frame,
            'DLC': dlc,
            'Payload': payload,
            'TimeInterval': timeinterval
        })

    return pd.DataFrame(synthetic_messages)

# Set training parameters
epochs = 6000
batch_size = 100
generator_model_path = 'generator_Fuzzy_model.h5'
discriminator_model_path = 'discriminator_Fuzzy_model.h5'

# Train the GAN and save the models
tf.config.run_functions_eagerly(True) # Enable eager execution for debugging
train(epochs, batch_size, real_data, generator_model_path, discriminator_model_path)

# Load the saved generator model
loaded_generator = load_model(generator_model_path)
print(f"\nLoaded generator model from {generator_model_path}")

# Generate synthetic CAN messages using the loaded model
num_synthetic_samples = 5000  # You can adjust the number of samples
synthetic_df = generate_synthetic_can_messages(loaded_generator, num_synthetic_samples, max_timeinterval_global, max_dlc_global, max_id_global, max_rf_global, PAYLOAD_LENGTH)

# Saving the generated DataFrame to a CSV file
synthetic_df.to_csv('synthetic_Fuzzy_dataframe_generated_6000.csv', index=False)
print(f"\nGenerated synthetic data saved to synthetic_Fuzzy_dataframe_generated.csv")

# Print a sample of the generated synthetic data
print("\nGenerated Synthetic Fuzzy Attack CAN Messages (Sample):")
print(synthetic_df.head())