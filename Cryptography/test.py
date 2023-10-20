import PySimpleGUI as sg
from tinyec import registry
import secrets
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from PIL import Image
from PIL import ImageTk
import math
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Getting the 'brainpoolP256r1' curve from the registry
curve = registry.get_curve('brainpoolP256r1')

# -- Bob and Alice Key Exchange -- 
# Generating Alice's private key
alice_privatekey = secrets.randbelow(curve.field.n)
print("Alice's private key: ", alice_privatekey)

# Generating Bob's private key
bob_privatekey = secrets.randbelow(curve.field.n)
print("Bob's private key: ", bob_privatekey)

# Generate Alice's public key from her private key and Generator point
alice_publickey = alice_privatekey * curve.g
print("Alice's public key: ", alice_publickey)

# Generate Bob's public key from his private key and Generator point
bob_publickey = bob_privatekey * curve.g
print("Bob's public key: ", bob_publickey)

# The shared key with Alice
alice_sharedkey = alice_privatekey * bob_publickey
print("Alice's shared secret key: ", alice_sharedkey)

# The shared key with Bob
bob_sharedkey = bob_privatekey * alice_publickey
print("Bob's shared secret key: ", bob_sharedkey)

# Check if shared secret keys match
if alice_sharedkey == bob_sharedkey:
    print("Shared secret keys match each other")
else:
    print("Shared secret keys don't match each other")

# -- Convert the x & y components to bytes of length 32 -- 
x_component = int.to_bytes(alice_sharedkey.x, 32, 'big')
y_component = int.to_bytes(bob_sharedkey.y, 32, 'big')

# Create a SHA3_256 class
sha3_key = hashlib.sha3_256()

# Update the hash object with x_component
sha3_key.update(x_component)

# Concatenate the y_component with x_component in the hash object
sha3_key.update(y_component)

# Derive the key
secret_key = sha3_key.digest()
print("Derived secret key: ", secret_key.hex())

# Image Encryption
# Initialize AES cipher with the shared secret key
cipher = AES.new(secret_key, AES.MODE_EAX)

# Function to calculate horizontal correlation
def calculate_horizontal_correlation(image):
    horizontal_kernel = np.array([[1, 1, 1],
                                  [0, 0, 0],
                                  [-1, -1, -1]])
    horizontal_correlation = cv2.filter2D(image, -1, horizontal_kernel)
    mean_correlation = np.mean(np.abs(horizontal_correlation))
    return mean_correlation

# Function to calculate vertical correlation
def calculate_vertical_correlation(image):
    vertical_kernel = np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]])
    vertical_correlation = cv2.filter2D(image, -1, vertical_kernel)
    mean_correlation = np.mean(np.abs(vertical_correlation))
    return mean_correlation

# Function to calculate diagonal correlation
def calculate_diagonal_correlation(image, diagonal_kernel):
    diagonal_correlation = cv2.filter2D(image, -1, diagonal_kernel)
    mean_correlation = np.mean(np.abs(diagonal_correlation))
    return mean_correlation

def encrypt(input_image_path, output_image_path):
    # Load the image you want to encrypt
    with open(input_image_path, 'rb') as file:
        image_data = file.read()

    # Encrypt the image
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(image_data)

    # Save the encrypted image
    with open(output_image_path, 'wb') as file:
        file.write(nonce)
        file.write(tag)
        file.write(ciphertext)

    print(f"Image encrypted and saved as {output_image_path}")

def decrypt(input_image_path, output_image_path):
    with open(input_image_path, 'rb') as file:
        nonce = file.read(16)
        tag = file.read(16)
        ciphertext = file.read()

    # Initialize the AES cipher for decryption
    cipher = AES.new(secret_key, AES.MODE_EAX, nonce=nonce)

    # Decrypt the image
    decrypted_image = cipher.decrypt(ciphertext)

    # Verify the tag to check for integrity
    try:
        cipher.verify(tag)
        print("Tag verification successful. Image integrity preserved.")
    except ValueError:
        print("Tag verification failed. Image may be corrupted.")

    # Save the decrypted image
    with open(output_image_path, 'wb') as output_file:
        output_file.write(decrypted_image)

    print(f"Image decrypted and saved as {output_image_path}")

# Function to save the encrypted image as a visible BMP image
def save_encrypted_image_as_bmp(input_image_path, encrypted_image_path):
    with open(input_image_path, 'rb') as file:
        image_data = file.read()

    # Reinitialize the AES cipher with the shared secret key
    cipher = AES.new(secret_key, AES.MODE_EAX)

    # Encrypt the image data
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(image_data)

    # Create a visible image
    num_bytes = len(ciphertext)
    num_pixels = int((num_bytes + 2) / 3)  # 3 bytes per pixel

    # Calculate the image dimensions
    W = H = int(math.ceil(num_pixels ** 0.5))  # W=H, such that everything fits in
    total_pixels = W * H * 3  # 3 bytes per pixel

    # Ensure the image data is padded to the required length
    imagedata = ciphertext + b'\0' * (total_pixels - num_bytes)

    # Create a visible image
    visible_image = Image.frombytes('RGB', (W, H), imagedata)

    # Save the visible image as a BMP file
    visible_image.save(encrypted_image_path, "BMP")

    print(f"Image encrypted and saved as {encrypted_image_path}")


# Function to display a histogram using PySimpleGUI
def display_histogram(image_path, title, window, key):
    img = cv2.imread(image_path, 0)
    histr = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Create a histogram figure
    fig, ax = plt.subplots()
    ax.plot(histr)
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    # Create a layout for the histogram window
    layout = [
        [sg.Canvas(key=key)],
        [sg.Text(title, size=(20, 1))]
    ]

    # Create the histogram window
    window_histogram = sg.Window(title, layout, finalize=True)
    canvas_elem = window_histogram[key]
    canvas = FigureCanvasTkAgg(fig, master=canvas_elem.Widget)
    canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
    canvas.draw()

    event, values = window_histogram.read()

    # Close the histogram window when done
    window_histogram.close()

"""
# Function to create grayscale scatterplots
def create_gray_scatterplot(image_path, title, key):
    # Load the image
    image = Image.open(image_path)

    # Convert the image to grayscale
    gray_image = image.convert('L')

    # Get the pixel values as a NumPy array
    gray_values = np.array(gray_image)

    # Get the dimensions of the image
    height, width = gray_values.shape

    # Create a scatterplot
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.repeat(np.arange(width), height)
    y = np.tile(np.arange(height), width)
    scatter = ax.scatter(x, y, c=gray_values.ravel(), cmap='gray', s=1)
    ax.invert_yaxis()  # Invert y-axis to match the image orientation

    # Set labels and title
    ax.set_xlabel('X (Pixel Column)')
    ax.set_ylabel('Y (Pixel Row)')
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label='Gray Value')

    # Embed the scatterplot in the PySimpleGUI window
    layout = [
        [sg.Canvas(key=key)],
        [sg.Text(title, size=(20, 1))]
    ]

    window = sg.Window(title, layout, finalize=True)
    canvas_elem = window[key]
    canvas = FigureCanvasTkAgg(fig, master=canvas_elem.Widget)
    canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
    canvas.draw()

    event, values = window.read()

    # Close the scatterplot window when done
    window.close()
"""

# Function to calculate image entropy
def image_entropy(image_path):
    img = Image.open(image_path).convert('L')  # Open the image and convert to grayscale
    img_array = np.array(img)
    
    # Calculate the histogram of pixel values
    hist = np.histogram(img_array, bins=256, range=(0, 256))[0]
    hist = hist / float(np.sum(hist))  # Normalize the histogram
    
    # Calculate entropy
    entropy = -np.sum([p * math.log2(p + 1e-10) for p in hist if p != 0])
    
    return entropy

# Function to plot the entropy comparison
def plot_entropy_comparison(original_entropy, encrypted_entropy):
    labels = ['Original Image', 'Encrypted Image']
    entropies = [original_entropy, encrypted_entropy]

    plt.bar(labels, entropies)
    plt.ylabel('Entropy')
    plt.title('Entropy Comparison')
    plt.show()


# GUI AREA
layout = [
    [sg.Text('Image Encryptor and Decryptor')],
    [sg.Text('Select an image to process:'), sg.InputText(key='input_file', do_not_clear=False), sg.FileBrowse()],
    [sg.Button('Encrypt'), sg.Button('Decrypt')],
    [sg.Text('', key='status_text')],  # Add a text element to display the status
    [sg.Image(size=(150, 150), key='-ORIGINAL-'), sg.Image(size=(150, 150), key='-ENCRYPTED-'), sg.Image(size=(150, 150), key='-DECRYPTED-')],
    [sg.Text('Original Image', size=(20, 1)), sg.Text('Encrypted Image', size=(20, 1)), sg.Text('Decrypted Image', size=(20, 1))],
    [sg.Text('Horizontal Correlation: ', key='-HORIZONTAL-')],
    [sg.Text('Vertical Correlation: ', key='-VERTICAL-')],
    [sg.Text('Diagonal Correlation: ', key='-DIAGONAL-')],
    # [sg.Canvas(size=(150, 150), key='-HIST_ORIGINAL-'), sg.Canvas(size=(150, 150), key='-HIST_ENCRYPTED-'), sg.Canvas(size=(150, 150), key='-HIST_DECRYPTED-')],
    # [sg.Text('Original Histogram', size=(20, 1)), sg.Text('Encrypted Histogram', size=(20, 1)), sg.Text('Decrypted Histogram', size=(20, 1))],
    # [sg.Canvas(size=(150, 150), key='-SCATTER_ORIGINAL-'), sg.Canvas(size=(150, 150), key='-SCATTER_ENCRYPTED-'), sg.Canvas(size=(150, 150), key='-SCATTER_DECRYPTED-')],
    # [sg.Text('Original Scatterplot', size=(20, 1)), sg.Text('Encrypted Scatterplot', size=(20, 1)), sg.Text('Decrypted Scatterplot', size=(20, 1))],
]

window = sg.Window('Image Encryptor and Decryptor', layout, finalize=True)

def update_images(original_path, encrypted_path, decrypted_path):
    try:
        # Update original image
        original_im = Image.open(original_path)
        original_im = original_im.resize((150, 150), resample=Image.LANCZOS)
        original_image = ImageTk.PhotoImage(image=original_im)
        window['-ORIGINAL-'].update(data=original_image)
    except Exception as e:
        print(f"Error updating original image: {e}")
        window['-ORIGINAL-'].update(data=None)  # Handle the case when the image cannot be opened

    try:
        # Update encrypted image as BMP
        save_encrypted_image_as_bmp(encrypted_path, 'encrypted_image.bmp')
        encrypted_im = Image.open('encrypted_image.bmp')
        encrypted_im = encrypted_im.resize((150, 150), resample=Image.LANCZOS)
        encrypted_image = ImageTk.PhotoImage(image=encrypted_im)
        window['-ENCRYPTED-'].update(data=encrypted_image)
    except Exception as e:
        print(f"Error updating encrypted image as BMP: {e}")
        window['-ENCRYPTED-'].update(data=None)  # Handle the case when the image cannot be opened

    try:
        # Update decrypted image
        decrypted_im = Image.open(decrypted_path)
        decrypted_im = decrypted_im.resize((150, 150), resample=Image.LANCZOS)
        decrypted_image = ImageTk.PhotoImage(image=decrypted_im)
        window['-DECRYPTED-'].update(data=decrypted_image)
    except Exception as e:
        print(f"Error updating decrypted image: {e}")
        window['-DECRYPTED-'].update(data=None)
        
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break
    elif event == 'Encrypt':
        input_file = values['input_file']
        if input_file:
            output_file = "encrypted_image.jpg"
            encrypt(input_file, output_file)

            # Update the status text
            window['status_text'].update("Image encrypted and saved as encrypted_image.jpg, please select an image to decrypt")

            # Update the images (original and encrypted)
            update_images(input_file, output_file, 'decrypted_image.jpg')
            
            # Update the images and display histograms
            # display_histogram('image.jpg', "Original Image Histogram", window, '-HIST_ORIGINAL-')
            # display_histogram('encrypted_image.bmp', "Encrypted Image Histogram", window, '-HIST_ENCRYPTED-')
            # display_histogram('decrypted_image.jpg', "Decrypted Image Histogram", window, '-HIST_DECRYPTED-')

    elif event == 'Decrypt':
        input_file = values['input_file']
        if input_file:
            output_file = 'decrypted_image.jpg'
            decrypt(input_file, output_file)

            # Update the status text
            window['status_text'].update("Image decrypted and saved as decrypted_image.jpg")

            # Update the images (original, encrypted, and decrypted)
            update_images('image.jpg', 'encrypted_image.bmp', output_file)
            display_histogram('image.jpg', "Original Image Histogram", window, '-HIST_ORIGINAL-')
            display_histogram('encrypted_image.bmp', "Encrypted Image Histogram", window, '-HIST_ENCRYPTED-')
            display_histogram('decrypted_image.jpg', "Decrypted Image Histogram", window, '-HIST_DECRYPTED-')
            
            """
            create_gray_scatterplot('image.jpg', "Original Scatterplot", '-SCATTER_ORIGINAL-')
            create_gray_scatterplot('encrypted_image.bmp', "Encrypted Scatterplot", '-SCATTER_ENCRYPTED-')
            create_gray_scatterplot('decrypted_image.jpg', "Decrypted Scatterplot", '-SCATTER_DECRYPTED-')
            """
             # Calculate and update the correlations
            horizontal_correlation_result = calculate_horizontal_correlation(np.array(Image.open('image.jpg').convert('L')))
            vertical_correlation_result = calculate_vertical_correlation(np.array(Image.open('encrypted_image.bmp').convert('L')))
            diagonal_kernel = np.array([[2, -1, -1],
                                       [-1, 2, -1],
                                       [-1, -1, 2]])
            diagonal_correlation_result = calculate_diagonal_correlation(np.array(Image.open('decrypted_image.jpg').convert('L')), diagonal_kernel)
            
             # Calculate the entropies
            entropy_original = image_entropy('image.jpg')
            entropy_encrypted = image_entropy('encrypted_image.bmp')
            
            window['-ENTROPY-'].update(f'Entropy Comparison: Original - {entropy_original:.2f}, Encrypted - {entropy_encrypted:.2f}')

            
            window['-HORIZONTAL-'].update(f'Horizontal Correlation: {horizontal_correlation_result:.2f}')
            window['-VERTICAL-'].update(f'Vertical Correlation: {vertical_correlation_result:.2f}')
            window['-DIAGONAL-'].update(f'Diagonal Correlation: {diagonal_correlation_result:.2f}')

window.close()