import os
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine


def filter_images_by_tile(directories, tile_code):
    """
    Collects a list of filenames from multiple directories that contain 
    a specific MGRS tile code.

    Parameters:
    - directories: A list of directories containing the image files.
    - tile_code: The MGRS tile code to filter by (e.g., 'T32UPB').

    Returns:
    - A list of filenames that contain the specified MGRS tile code.
    """
    # List to hold the filenames that match the tile_code
    filtered_filenames = []

    # Iterate over each directory
    for directory in directories:
        # Iterate over all files in the given directory
        for filename in os.listdir(directory):
            # Check if the tile_code is part of the filename
            if tile_code in filename:
                filtered_filenames.append(filename)

    return filtered_filenames


def extract_date_from_filename(filename):
    """
    Extracts the date from a filename based on a specific format.

    Assumes the filename format is 'S2A_MSIL1C_YYYYMMDD...tif', where the date 
    is embedded as the third part of the filename separated by underscores.

    Parameters:
    - filename: The filename from which to extract the date.

    Returns:
    - A datetime object representing the extracted date.
    """
    # Split the filename by underscore and extract the third element,
    # which is expected to be the date string in 'YYYYMMDDTHHMMSS' format.
    date_str = filename.split('_')[2]

    # Convert the date string into a datetime object and return it.
    return datetime.strptime(date_str, '%Y%m%dT%H%M%S')


def load_and_resize_images(directories, filenames, new_size=(1024, 1024)):
    """
    Loads and resizes images from multiple directories.

    Parameters:
    - directories: A list of directories to search for images.
    - filenames: A list of filenames to load and resize.
    - new_size: A tuple (width, height) representing the new size for the images.

    Returns:
    - A tuple containing two lists:
      - The first list contains the resized images as NumPy arrays.
      - The second list contains the corresponding dates extracted from the filenames.
    """
    # Initialize lists to store the resized images and their corresponding dates.
    resized_images = []
    dates = []

    # Iterate over each directory in the provided list.
    for directory in directories:
        # Iterate over each filename in the provided list.
        for filename in filenames:
            # Construct the full path to the image file.
            file_path = os.path.join(directory, filename)

            # Load the image from the file path.
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            # Check if the image was loaded successfully.
            if image is not None:
                # Resize the image to the new size and add it to the list.
                resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
                resized_images.append(resized_image)

                # Extract the date from the filename and add it to the dates list.
                date = extract_date_from_filename(filename)
                dates.append(date)
            else:
                # Print an error message if the image failed to load.
                print(f"Failed to load image: {file_path}")

    # Return the list of resized images and their corresponding dates.
    return resized_images, dates


def show_images(images, dates):
    """
    Displays a series of images with their corresponding dates.

    Parameters:
    - images: A list of images, where each image is a NumPy array.
    - dates: A list of datetime objects corresponding to the dates of the images.

    Each image is displayed in a separate plot with the acquisition date in the title.
    """
    # Iterate over both images and dates using enumerate for indexing.
    for i, (image, date) in enumerate(zip(images, dates)):
        # Create a new figure with a specified size.
        plt.figure(figsize=(10, 10))

        # Convert the image from BGR (OpenCV default) to RGB for correct color display.
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Set the title of the plot to include the date and the image index.
        # The date is formatted as 'YYYY-MM-DD'.
        plt.title(f'Image Date: {date} - Image {i}')

        # Turn off the axis for a cleaner display.
        plt.axis('off')

        # Display the image.
        plt.show()
        

def load_and_crop_images(directories, filenames, img_height, img_width):
    """
    Loads and crops images from multiple directories.

    Parameters:
    - directories: A list of directories to search for images.
    - filenames: A list of filenames to load and crop.
    - img_height: The height of the top-left crop of the image.
    - img_width: The width of the top-left crop of the image.

    Returns:
    - A tuple containing two lists:
      - The first list contains the cropped images as NumPy arrays.
      - The second list contains the corresponding dates extracted from the filenames.
    """
    cropped_images = []
    dates = []

    for directory in directories:
        for filename in filenames:
            file_path = os.path.join(directory, filename)
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

            if image is not None:
                # Crop the image to the specified top-left area
                cropped_image = image[:img_height, :img_width]
                cropped_images.append(cropped_image)

                # Extract the date from the filename
                date = extract_date_from_filename(filename)
                dates.append(date)
            else:
                print(f"Failed to load image: {file_path}")

    return cropped_images, dates


def calculate_similarity_matrix(cropped_images):
    """
    Calculates a similarity matrix for a list of cropped images.

    Parameters:
    - cropped_images: A list of cropped images as NumPy arrays.

    Returns:
    - A similarity matrix where each element represents the Euclidean distance
      between a pair of images.
    """
    # Flatten and normalize each image for comparison.
    processed_images = [img.flatten() / np.linalg.norm(img.flatten()) for img in cropped_images]
    similarity_matrix = np.zeros((len(processed_images), len(processed_images)))

    for i in range(len(processed_images)):
        for j in range(len(processed_images)):
            # Calculate the Euclidean distance between each pair of images.
            similarity_matrix[i][j] = euclidean(processed_images[i], processed_images[j])
    
    return similarity_matrix



def calculate_cosine_similarity_matrix(cropped_images):
    processed_images = [img.flatten() / np.linalg.norm(img.flatten()) for img in cropped_images]
    similarity_matrix = np.zeros((len(processed_images), len(processed_images)))

    for i in range(len(processed_images)):
        for j in range(len(processed_images)):
            similarity_matrix[i][j] = 1 - cosine(processed_images[i], processed_images[j])
    
    return similarity_matrix



