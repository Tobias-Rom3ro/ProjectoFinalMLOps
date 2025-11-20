# file: cnn_image/tests/test_filters.py
import pytest
import numpy as np
from PIL import Image
from filters.convolution_filters import ConvolutionFilter


@pytest.fixture
def test_image():
    img_array = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    return Image.fromarray(img_array, mode='L')


@pytest.fixture
def test_color_image():
    img_array = np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode='RGB')


def test_blur_filter(test_image):
    filtered = ConvolutionFilter.blur(test_image)
    
    assert filtered.size == test_image.size
    assert filtered.mode == test_image.mode


def test_edge_detection_filter(test_image):
    filtered = ConvolutionFilter.edge_detection(test_image)
    
    assert filtered.size == test_image.size
    assert filtered.mode == test_image.mode


def test_sharpen_filter(test_image):
    filtered = ConvolutionFilter.sharpen(test_image)
    
    assert filtered.size == test_image.size
    assert filtered.mode == test_image.mode


def test_blur_color_image(test_color_image):
    filtered = ConvolutionFilter.blur(test_color_image)
    
    assert filtered.size == test_color_image.size
    assert filtered.mode == test_color_image.mode


def test_get_available_filters():
    filters = ConvolutionFilter.get_available_filters()
    
    assert isinstance(filters, list)
    assert "blur" in filters
    assert "edge_detection" in filters
    assert "sharpen" in filters
    assert "none" in filters


def test_apply_kernel_grayscale():
    img_array = np.ones((10, 10), dtype=np.uint8) * 100
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
    
    result = ConvolutionFilter.apply_kernel(img_array, kernel)
    
    assert result.shape == img_array.shape
    assert result.dtype == np.uint8


def test_apply_kernel_color():
    img_array = np.ones((10, 10, 3), dtype=np.uint8) * 100
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
    
    result = ConvolutionFilter.apply_kernel(img_array, kernel)
    
    assert result.shape == img_array.shape
    assert result.dtype == np.uint8