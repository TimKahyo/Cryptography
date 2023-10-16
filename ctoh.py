import random


# Function to generate a random homophonic substitution mapping with units
def generate_homophonic_mapping():
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    homophonic_mapping = {}

    for letter in alphabet:
        num_variants = random.randint(2, 5)  # Generate 2 to 5 variants for each letter
        variants = []

        for _ in range(num_variants):
            variant = random.choice(alphabet)
            while variant in variants:
                variant = random.choice(alphabet)

            unit = random.randint(1, 9)  # Generate a random unit value (1-9)
            variants.append((variant, unit))

        homophonic_mapping[letter] = variants

    return homophonic_mapping


# Function to encrypt a message using the homophonic Caesar cipher with units
def encrypt_caesar_homophonic(plaintext, shift, homophonic_mapping):
    encrypted_text = ""

    for char in plaintext:
        if char.isalpha():
            is_upper = char.isupper()
            char = char.upper()

            if char in homophonic_mapping:
                variants = homophonic_mapping[char]
                encrypted_char, unit = random.choice(variants)

                if is_upper:
                    encrypted_text += encrypted_char + str(unit)
                else:
                    encrypted_text += encrypted_char.lower() + str(unit)
            else:
                encrypted_text += char
        else:
            encrypted_text += char

    return encrypted_text


# Function to decrypt a message encrypted with the homophonic Caesar cipher with units
def decrypt_caesar_homophonic(ciphertext, shift, homophonic_mapping):
    decrypted_text = ""

    i = 0
    while i < len(ciphertext):
        char = ciphertext[i]
        if char.isalpha():
            is_upper = char.isupper()
            char = char.upper()

            for key, variants in homophonic_mapping.items():
                if char in [variant[0] for variant in variants]:
                    for variant in variants:
                        if variant[0] == char:
                            encrypted_char = variant[0]
                            unit = variant[1]
                            if is_upper:
                                decrypted_text += encrypted_char
                            else:
                                decrypted_text += encrypted_char.lower()
                            break
                    i += len(str(unit))
                    break
            else:
                decrypted_text += char
                i += 1
        else:
            decrypted_text += char
            i += 1

    return decrypted_text


# Example usage
plaintext = "HELLO WORLD"
shift = 3
homophonic_mapping = generate_homophonic_mapping()

encrypted_text = encrypt_caesar_homophonic(plaintext, shift, homophonic_mapping)
print("Encrypted:", encrypted_text)

decrypted_text = decrypt_caesar_homophonic(encrypted_text, shift, homophonic_mapping)
print("Decrypted:", decrypted_text)
