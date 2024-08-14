from typing import List, Tuple

import cv2
import numpy as np

def get_background_color(
    image: np.ndarray,
    threshold: int = 230
    ) -> List[np.unit8, np.unit8, np.unit8]:

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
    background_color: np.uint8
        The average background color of the whole slide image.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    background_intensity = np.mean(image[mask == 255], dtype=np.unit8)
    background_color = [background_intensity for _ in range(3)]

    return background_color


def get_contours(
    image: np.ndarray,
    threshold: int = 230,
    kernel_size: Tuple[int, int] = (3, 3)
    ) -> List[Tuple[int, int, int, int]]:

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.GaussianBlur(binary, kernel_size)

    kernel = np.ones(kernel_size, np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours
