import numpy as np


def salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.copy()

    num_salt = int(salt_prob * image.size)
    salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    num_pepper = int(pepper_prob * image.size)
    pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image


def gaussian_noise(image, mean=0, stddev=0):
    gaussian = np.random.normal(mean, stddev, image.shape)
    noisy_image = np.array(image, dtype=float) + gaussian
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


def sparkle_noise(image, mean=0, stddev=0):
    gaussian = np.random.normal(mean, stddev, image.shape)
    noisy_image = np.array(image, dtype=float) + (np.array(image,dtype=float)*gaussian/255.0)
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)
