from transformers import AutoImageProcessor, AutoConfig, AutoModelForImageClassification
m = "CristianR8/vit_large2-model"
proc = AutoImageProcessor.from_pretrained(m, revision="main", trust_remote_code=True)
print("Processor:", type(proc).__name__)
cfg = AutoConfig.from_pretrained(m, trust_remote_code=True)
print("Config model_type:", cfg.model_type)
model = AutoModelForImageClassification.from_pretrained(m, trust_remote_code=True)
print("Model:", type(model).__name__)
