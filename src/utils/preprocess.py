import itertools
from typing import List, Tuple, Dict, Union

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


def downsample(
    image: np.ndarray, 
    scaling_factor: int
    ) -> np.ndarray:

    height, width = image.shape[:2]
    downsampled = cv2.resize(image, (width // scaling_factor, height // scaling_factor))

    return downsampled


def get_valid_contours(contours: List[np.ndarray]) -> List[np.ndarray]:
    valid_contours = []

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        if w > 30 and h > 30:
            valid_contours.append(contour)

    return valid_contours


def get_contours(
    image: np.ndarray,
    threshold: int = 230,
    kernel_size: Tuple[int] = (5, 5)
    ) -> List[np.ndarray]:

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.GaussianBlur(binary, kernel_size, 0)

    kernel = np.ones(kernel_size, np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = get_valid_contours(contours)

    return contours


def get_coordinates(
    contours: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int]]:
    
    coordinates = [{"coordinates": cv2.boundingRect(c), "contours": c} for c in contours]

    return coordinates


def squeeze_coordinates(
    coordinates: List[Tuple[int]],
    spacing: int
    ) -> List[Tuple[int]]:

    squeezed_coordinates = []
    sorted_coordinates = sorted(coordinates, key=lambda x: (x["coordinates"][1], x["coordinates"][0]))
    max_width = max(sorted_coordinates, key=lambda x: x["coordinates"][2])
    max_width = max_width["coordinates"][2]
    max_width = max_width if max_width > 10_000 else 10_000

    pointer_x, pointer_y = 0, 0

    for c in sorted_coordinates:
        c = c["coordinates"]
        tissue_width, tissue_height = c[2], c[3]

        if (pointer_x + tissue_width) > max_width:
            pointer_y += (max(squeezed_coordinates, key=lambda x: (x[1], x[3]))[3] + spacing)
            pointer_x = 0
        
        squeezed_coordinates.append((pointer_x, pointer_y, tissue_width, tissue_height))
        pointer_x += squeezed_coordinates[-1][2] + spacing

    coordinate_dict = {
        "sorted_coordinates": sorted_coordinates,
        "squeezed_coordinates": squeezed_coordinates
    }

    return coordinate_dict


def organize_tissues(
    image: np.ndarray,
    spacing: int,
    coordinate_dict: List[Dict[str, Union[np.ndarray, List[int]]]]
    ) -> np.ndarray:

    """
    This script processes the whole slide image along with the tissue contours,
    removing whitespace between tissues to preserve the essential information within the image.
    """

    sorted_coordinates = coordinate_dict["sorted_coordinates"]
    squeezed_coordinates = coordinate_dict["squeezed_coordinates"]

    max_width = max(sorted_coordinates, key=lambda x: x["coordinates"][2])
    max_width = max_width["coordinates"][2]
    max_width = max_width if max_width > 10_000 else 10_000

    rows_height = [max(group, key=lambda x: x[3])[3] + spacing for _, group in itertools.groupby(squeezed_coordinates, key=lambda x: x[1])]
    max_height = sum(rows_height)

    processed_image = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    for sorted, squeezed in zip(sorted_coordinates, squeezed_coordinates):
        coordinates = sorted["coordinates"]
        contours = [sorted["contours"]]

        contour_mask = np.zeros_like(image, dtype=np.float32)
        contour_mask = cv2.drawContours(contour_mask, contours, -1, (1, 1, 1), cv2.FILLED)

        snapshot = image.copy() * contour_mask
        
        x0, y0, w0, h0 = coordinates
        x1, y1, w1, h1 = squeezed

        detected_tissue = snapshot[y0:y0+h0, x0:x0+w0]
        processed_image[y1:y1+h1, x1:x1+w1] = detected_tissue
    
    return processed_image, contour_mask
