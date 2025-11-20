import numpy as np
from typing import Tuple
from PIL import Image

class ConvolutionFilter:
    @staticmethod
    def apply_kernel(
        image_array: np.ndarray, 
        kernel: np.ndarray
    ) -> np.ndarray:
        height, width = image_array.shape[:2]
        kernel_height, kernel_width = kernel.shape
        
        pad_h = kernel_height // 2
        pad_w = kernel_width // 2
        
        if len(image_array.shape) == 3:
            padded = np.pad(
                image_array,
                ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                mode='edge'
            )
            output = np.zeros_like(image_array)
            
            for i in range(height):
                for j in range(width):
                    for c in range(image_array.shape[2]):
                        region = padded[i:i+kernel_height, j:j+kernel_width, c]
                        output[i, j, c] = np.sum(region * kernel)
        else:
            padded = np.pad(
                image_array,
                ((pad_h, pad_h), (pad_w, pad_w)),
                mode='edge'
            )
            output = np.zeros_like(image_array)
            
            for i in range(height):
                for j in range(width):
                    region = padded[i:i+kernel_height, j:j+kernel_width]
                    output[i, j] = np.sum(region * kernel)
        
        return np.clip(output, 0, 255).astype(np.uint8)
    
    @staticmethod
    def blur(image: Image.Image) -> Image.Image:
        kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=np.float32) / 16.0
        
        image_array = np.array(image)
        filtered = ConvolutionFilter.apply_kernel(image_array, kernel)
        
        return Image.fromarray(filtered)
    
    @staticmethod
    def edge_detection(image: Image.Image) -> Image.Image:
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)
        
        image_array = np.array(image)
        filtered = ConvolutionFilter.apply_kernel(image_array, kernel)
        
        return Image.fromarray(filtered)
    
    @staticmethod
    def sharpen(image: Image.Image) -> Image.Image:
        kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)
        
        image_array = np.array(image)
        filtered = ConvolutionFilter.apply_kernel(image_array, kernel)
        
        return Image.fromarray(filtered)
    
    @staticmethod
    def get_available_filters() -> list:
        return ["blur", "edge_detection", "sharpen", "none"]