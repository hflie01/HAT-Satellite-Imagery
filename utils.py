import os
import cv2
from PIL import Image
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_images_to_numpy_arrays_cv2(directory):
    image_arrays = []
    for filename in os.listdir(directory):
        if filename.endswith('.tif'):  # Check for TIFF files
            file_path = os.path.join(directory, filename)
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  # Read the image with unchanged color depth
            if img is not None:
                print(img.shape)
                #image_arrays.append(img)
            else:
                print(f"Could not read the image: {file_path}")
    return image_arrays



def convert_datetime(date_str):
    # Assuming your date format is something like 'YYYYMMDDTHHMMSS'
    return datetime.strptime(date_str, '%Y%m%dT%H%M%S')



def process_satellite_data(directory):
    files = os.listdir(directory)
    columns = ['Satellite', 'Processing_Level', 'Acquisition_DateTime', 'Baseline_Number', 
               'Relative_Orbit', 'Tile_Number', 'Product_Discriminator']

    # Initialize an empty array for data with zero rows and seven columns
    data = np.empty((0, 7)) 

    for file in files:
        parts = file.split('_')
        if len(parts) == 7:
            var = np.array([parts])  # Create a 2D array from parts
            data = np.vstack((data, var))  # Stack the new array under the existing data

    # Create a DataFrame from this numpy array
    df = pd.DataFrame(data, columns=columns)

    # Convert Acquisition_DateTime to datetime object and add a Month column
    df['Month'] = df['Acquisition_DateTime'].apply(convert_datetime).dt.to_period('M')

    return df



def show_images(images):
    """
    Displays images in a Jupyter notebook.

    Parameters:
    - images: A list of NumPy arrays representing the images to display.
    """
    for i, image in enumerate(images):
        plt.figure(figsize=(10, 10))  # Set the figure size to 10x10 inches
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        plt.title(f'Image {i}')
        plt.axis('off')  # Hide the axis
        plt.show()
        
        
def load_and_resize_images(directory, filenames, new_size=(1024, 1024)):
    """
    Loads images from a list of filenames, resizes them to the given size, 
    and stores them as NumPy arrays.

    Parameters:
    - directory: The directory containing the image files.
    - filenames: A list of filenames to load.
    - new_size: A tuple defining the new image size (width, height).

    Returns:
    - A list of NumPy arrays representing the resized images.
    """
    resized_images = []

    # Iterate over the filenames
    for filename in filenames:
        # Construct the full path to the image file
        file_path = os.path.join(directory, filename)
        
        # Load the image using OpenCV
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        # Check if the image was loaded successfully
        if image is not None:
            # Resize the image to the new size
            resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            resized_images.append(resized_image)
        else:
            print(f"Failed to load image: {file_path}")

    return resized_images


def filter_images_by_tile(directory, tile_code):
    """
    Collects a list of filenames that contain a specific MGRS tile code.

    Parameters:
    - directory: The directory containing the image files.
    - tile_code: The MGRS tile code to filter by (e.g., 'T32UPB').

    Returns:
    - A list of filenames that contain the specified MGRS tile code.
    """
    # List to hold the filenames that match the tile_code
    filtered_filenames = []

    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        # Check if the tile_code is part of the filename
        if tile_code in filename:
            filtered_filenames.append(filename)

    return filtered_filenames



def process_satellite_data_dir(directories):
    # Define the column names for the DataFrame.
    columns = ['Satellite', 'Processing_Level', 'Acquisition_DateTime', 
               'Baseline_Number', 'Relative_Orbit', 'Tile_Number', 
               'Product_Discriminator']

    # Initialize an empty DataFrame to hold all the data.
    all_data = pd.DataFrame(columns=columns)

    # Iterate over each directory in the list.
    for directory in directories:
        # List all files in the current directory.
        files = os.listdir(directory)
        
        # Initialize an empty array to store data for this directory.
        data = np.empty((0, 7))

        # Iterate over each file in the directory.
        for file in files:
            # Remove the file extension (.tif) from the filename.
            file_without_extension = file.replace('.tif', '')

            # Split the filename into parts.
            parts = file_without_extension.split('_')
            
            # Check if the filename has the expected 7 parts.
            if len(parts) == 7:
                # Convert the parts into a 2D numpy array.
                var = np.array([parts])
                
                # Stack this array with the existing data.
                data = np.vstack((data, var))

        # Convert the numpy array to a DataFrame and append it to the all_data DataFrame.
        df = pd.DataFrame(data, columns=columns)
        all_data = pd.concat([all_data, df], ignore_index=True)

    # Convert the 'Acquisition_DateTime' column to datetime objects and extract the month.
    # This adds a new column 'Month' to the DataFrame.
    all_data['Month'] = all_data['Acquisition_DateTime'].apply(convert_datetime).dt.to_period('M')

    # Return the combined DataFrame.
    return all_data

