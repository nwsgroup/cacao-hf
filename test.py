from transformers import AutoImageProcessor

proc = AutoImageProcessor.from_pretrained("/home/agrosavia/Documents/IA4CACAO/cacao-hf/outputs/outputs_convnext_xlarge", trust_remote_code= True)

print("data_config", proc.data_config.keys())
print("mean", proc.data_config["mean"])
print("std", proc.data_config["std"])