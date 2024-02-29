import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os
import cv2

def load_image_s2(red_path, green_path, blue_path):
    
    # Rotkanal lesen
    with rasterio.open(red_path) as src:
        red = src.read(1)  # Annahme: das relevante Band ist das erste Band
        red = scale_to_8bit(red)

    # Grünkanal lesen
    with rasterio.open(green_path) as src:
        green = src.read(1)  # Annahme: das relevante Band ist das erste Band
        green = scale_to_8bit(green)

    # Blaukanal lesen
    with rasterio.open(blue_path) as src:
        blue = src.read(1)  # Annahme: das relevante Band ist das erste Band
        blue = scale_to_8bit(blue)
        

    # RGB-Bild zusammensetzen
    rgb = np.dstack((red, green, blue))
    
    return rgb


def load_image_wv(red_path, green_path, blue_path):
    
    # Rotkanal lesen
    with rasterio.open(red_path) as src:
        red = src.read(1)  # Annahme: das relevante Band ist das erste Band
        
    # Grünkanal lesen
    with rasterio.open(green_path) as src:
        green = src.read(1)  # Annahme: das relevante Band ist das erste Band
        
    # Blaukanal lesen
    with rasterio.open(blue_path) as src:
        blue = src.read(1)  # Annahme: das relevante Band ist das erste Band
        
    # RGB-Bild zusammensetzen
    rgb = np.dstack((red, green, blue))
    
    return rgb


def scale_to_8bit(data_array, max_value=3000):
    """ Skaliert die Bilddaten auf den 8-Bit-Bereich (0-255). """
    data_array = np.clip(data_array, 0, max_value)  # Werte über max_value abschneiden
    normalized = data_array / max_value  # Normalisiere auf den Bereich von 0 bis 1
    return (normalized * 255).astype(np.uint8)  # Skaliere auf den Bereich von 0 bis 255


def calculate_brightness(rgb_image):
    """
    Berechnet die durchschnittliche Helligkeit eines RGB-Bildes.
    :param rgb_image: Ein RGB-Bild als Numpy-Array.
    :return: Durchschnittliche Helligkeit des Bildes.
    """
    # Konvertiere das Bild zu Graustufen, indem du den Durchschnitt der drei Farbkanäle nimmst
    grayscale_image = np.mean(rgb_image, axis=2)
    # Berechne die durchschnittliche Helligkeit
    return np.mean(grayscale_image)


def adjust_brightness(rgb_image, factor):
    """
    Passt die Helligkeit eines RGB-Bildes an.
    :param rgb_image: Ein RGB-Bild als Numpy-Array.
    :param factor: Faktor, um die Helligkeit anzupassen (>1 heller, <1 dunkler).
    :return: Ein angepasstes RGB-Bild als Numpy-Array.
    """
    # Stelle sicher, dass die Operation innerhalb des gültigen Bereichs bleibt
    adjusted_image = np.clip(rgb_image * factor, 0, 255)
    return adjusted_image.astype(np.uint8)


def split_image_into_tiles_save(path, image_np, num_tiles_row, num_tiles_col):
    """
    Zerlegt ein Bild (Numpy-Array) in eine vorgegebene Anzahl von Bildausschnitten und speichert sie im RGB-Format.
    
    :param path: Pfad, in den die Bildausschnitte gespeichert werden sollen.
    :param image_np: Numpy-Array des Bildes im BGR-Format.
    :param num_tiles_row: Anzahl der Kacheln pro Reihe.
    :param num_tiles_col: Anzahl der Kacheln pro Spalte.
    """
    tile_height = image_np.shape[0] // num_tiles_row
    tile_width = image_np.shape[1] // num_tiles_col
    index = 0
    
    for row in range(num_tiles_row):
        for col in range(num_tiles_col):
            start_row = row * tile_height
            start_col = col * tile_width
            end_row = start_row + tile_height
            end_col = start_col + tile_width
            # Schneide den Bildausschnitt aus
            tile = image_np[start_row:end_row, start_col:end_col]
            print(tile.shape)
            # Konvertiere BGR zu RGB
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            # Speichere den Bildausschnitt im RGB-Format
            cropped_filename = f"image_crop_{index}.png"
            save_path = os.path.join(path, cropped_filename)
            if not cv2.imwrite(save_path, tile_rgb):
                print(f"Fehler beim Speichern von {cropped_filename}")
            
            index += 1

    

def plot_image(image):
    
    plt.imshow(image)
    plt.title('RGB Image')
    plt.xlabel('Pixel')
    plt.ylabel('Pixel')
    plt.show()
    
    import matplotlib.pyplot as plt

def plot_two_images(image1, image2):
    """
    Plots two images side by side.

    :param image1: The first image to be plotted.
    :param image2: The second image to be plotted.
    """
    plt.figure(figsize=(12, 6))  # Adjust the figure size as needed

    # Plotting the first image
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first position
    plt.imshow(image1)
    plt.title('WorldView2')
    plt.xlabel('Pixel')
    plt.ylabel('Pixel')

    # Plotting the second image
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second position
    plt.imshow(image2)
    plt.title('Sentinel2')
    plt.xlabel('Pixel')
    plt.ylabel('Pixel')

    plt.show()

