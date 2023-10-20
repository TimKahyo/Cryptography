import PySimpleGUI as sg
from tinyec import registry
import secrets
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from PIL import Image
import io

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
    with open('decrypted_image.jpg', 'wb') as output_file:
        output_file.write(decrypted_image)

    print(f"Image decrypted and saved as {output_image_path}")

# GUI AREA
layout = [
    [sg.Text('Image Encryptor and Decryptor')],
    [sg.Text('Select an image to process:'), sg.InputText(key='input_file'), sg.FileBrowse()],
    [sg.Button('Encrypt'), sg.Button('Decrypt')],
    [sg.Text('', key='status_text')]  # Add a text element to display the status
]

window = sg.Window('Image Encryptor and Decryptor', layout, finalize=True)

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
            window['status_text'].update("Image encrypted and saved as encrypted_image.jpg, please select image to decrypt")

    elif event == 'Decrypt':
        input_file = values['input_file']
        if input_file:
            output_file = 'decrypted_image.jpg'
            decrypt(input_file, output_file)

            # Update the status text
            window['status_text'].update("Image decrypted and saved as decrypted_image.jpg")
            
window.close()