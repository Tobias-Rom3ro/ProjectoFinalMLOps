import logging
from PIL import Image
from filters.convolution_filters import ConvolutionFilter

logger = logging.getLogger(__name__)

class FilterService:
    def __init__(self):
        self.filter_engine = ConvolutionFilter()
        logger.info("Filter service initialized")
    
    def apply_filter(self, image: Image.Image, filter_name: str) -> Image.Image:
        logger.info(f"Applying filter: {filter_name}")
        
        if filter_name == "blur":
            return self.filter_engine.blur(image)
        elif filter_name == "edge_detection":
            return self.filter_engine.edge_detection(image)
        elif filter_name == "sharpen":
            return self.filter_engine.sharpen(image)
        elif filter_name == "none":
            return image
        else:
            logger.warning(f"Unknown filter: {filter_name}, returning original")
            return image
    
    def get_available_filters(self) -> list:
        return self.filter_engine.get_available_filters()