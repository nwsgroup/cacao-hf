import json
import os
from pathlib import Path

def create_preprocessor_config(config_path):
    """Create preprocessor config from model config"""
    try:
        # Read the model config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract preprocessing info - first try pretrained_cfg, then direct parameters
        if 'pretrained_cfg' in config:
            cfg = config['pretrained_cfg']
            mean = cfg.get('mean', [0.485, 0.456, 0.406])
            std = cfg.get('std', [0.229, 0.224, 0.225])
            size = cfg.get('input_size', [3, 224, 224])
            crop_pct = cfg.get('crop_pct', 0.875)
        else:
            # Default ImageNet values if not found
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            size = [3, 224, 224]
            crop_pct = 0.875

        # Create preprocessor config
        preprocessor_config = {
            "crop_pct": crop_pct,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": mean,
            "image_processor_type": "ConvNextImageProcessor",
            "image_std": std,
            "resample": 3,
            "rescale_factor": 0.00392156862745098,  # 1/255
            "size": {
                "shortest_edge": size[1] if len(size) > 1 else 224
            }
        }

        # Save path - same directory as config.json
        save_path = os.path.join(os.path.dirname(config_path), 'preprocessor_config.json')
        
        # Save with pretty printing
        with open(save_path, 'w') as f:
            json.dump(preprocessor_config, f, indent=2)
            
        print(f"Created preprocessor_config.json for {os.path.dirname(config_path)}")
        
    except Exception as e:
        print(f"Error processing {config_path}: {str(e)}")

def main():
    # Base models directory
    models_dir = "models"
    
    # Find all config.json files
    for model_dir in os.listdir(models_dir):
        config_path = os.path.join(models_dir, model_dir, "config.json")
        if os.path.exists(config_path):
            create_preprocessor_config(config_path)
        else:
            print(f"No config.json found in {model_dir}")

if __name__ == "__main__":
    main()