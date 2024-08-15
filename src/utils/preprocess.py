from typing import List, Tuple

import cv2
import numpy as np

def get_background_color(
    image: np.ndarray,
    threshold: int = 230
    ) -> List[int]:

    """
    Gets the average background color of the image.

    Parameters
    ----------
    image: np.array
        The whole slide image in the form of a numpy array.

    threshold: int
        The threshold for binary conversion. (default: 230)

    Returns
    -------
    background_color: List[int]
        The average background color of the whole slide image.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    background_intensity = np.mean(image[mask == 255]).astype(np.uint8)
    background_color = np.array([background_intensity for _ in range(3)])

    return background_color


def get_contours(
    image: np.ndarray,
    threshold: int = 230,
    kernel_size: Tuple[int] = (3, 3)
    ) -> List[np.ndarray]:

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.GaussianBlur(binary, kernel_size, 0)

    kernel = np.ones(kernel_size, np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 200]

    return contours


def get_coordinates(
    contours: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int]]:
    
    coordinates = [cv2.boundingRect(c) for c in contours]

    return coordinates

def squeeze_coordinates(
    coordinates: List[Tuple[int]]
    ) -> List[Tuple[int]]:

    new_cordinates = []
    coordinates = sorted(coordinates, key=lambda x: (x[1], x[0]))
    max_width = max(coordinates, key=lambda x: x[2])[2]
    max_width = max_width if max_width > 10_000 else 10_000

    pointer_x, pointer_y = 0, 0

    for c in coordinates:
        tissue_width, tissue_height = c[2], c[3]

        if (pointer_x + tissue_width) > max_width:
            pointer_y += (max(new_cordinates, key=lambda x: (x[0], x[3]))[3] + 10)
            pointer_x = 0
        
        new_cordinates.append((pointer_x, pointer_y, tissue_width, tissue_height))
        pointer_x += new_cordinates[-1][2] + 10

    return new_cordinates


def organize_tissues(
    image: np.ndarray,
    scaling_factor: int,
    contours: List[Tuple[int]]
    ) -> np.ndarray:

    """
    This script processes the downsampled whole slide image along with the tissue contours,
    removing whitespace between tissues to preserve the essential information within the image.
    """

    contours = [c * scaling_factor for c in contours]

    coordinates = get_contours(contours)

     
    



