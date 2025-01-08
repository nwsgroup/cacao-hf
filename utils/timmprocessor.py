from transformers import PretrainedConfig
from transformers import BaseImageProcessor
import json
import os
import timm

class TimmImageProcessor(BaseImageProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @classmethod
    def from_timm_name(cls, model_name, save_directory=None):
        """Create an ImageProcessor from a timm model name."""
        # Get model default config from timm
        model = timm.create_model(model_name, pretrained=False)
        data_config = timm.data.resolve_data_config(model.pretrained_cfg)
        
        # Create processor config
        config = {
            "do_normalize": True,
            "do_resize": True,
            "do_center_crop": True,
            "image_mean": data_config['mean'],
            "image_std": data_config['std'],
            "size": {
                "height": data_config['input_size'][1],
                "width": data_config['input_size'][2]
            },
            "crop_size": {
                "height": data_config['input_size'][1],
                "width": data_config['input_size'][2]
            },
            "do_rescale": True,
            "rescale_factor": 1/255,
            "resample": 3,  # BICUBIC
        }
        
        processor = cls(**config)
        
        if save_directory:
            # Save the config
            os.makedirs(save_directory, exist_ok=True)
            config_file = os.path.join(save_directory, "preprocessor_config.json")
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
        
        return processor

def get_image_processor(model_name_or_path, trust_remote_code=False):
    """
    Universal function to get image processor for both HF and timm models.
    """
    try:
        # First try loading as regular HF model
        from transformers import AutoImageProcessor
        return AutoImageProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code
        )
    except Exception as e:
        # If that fails, assume it's a timm model
        if "timm/" in model_name_or_path:
            timm_name = model_name_or_path.replace("timm/", "")
            return TimmImageProcessor.from_timm_name(
                timm_name, 
                save_directory=model_name_or_path if os.path.isdir(model_name_or_path) else None
            )
        raise e  # If it's not a timm model, raise the original error
