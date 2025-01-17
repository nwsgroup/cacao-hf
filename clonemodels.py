from huggingface_hub import snapshot_download
import os

def download_models():
    # Lista de modelos a descargar
    model_ids = [
        "CristianR8/vit_large-model",
        #"CristianR8/vit_base-model", 
        #"CristianR8/vgg19-model",
        #"CristianR8/mobilenet_large-model"
    ]
    
    # Directorio base donde se guardarán los modelos
    base_dir = "models"
    os.makedirs(base_dir, exist_ok=True)
    
    # Descargar cada modelo
    for model_id in model_ids:
        try:
            print(f"Descargando modelo: {model_id}")
            model_dir = os.path.join(base_dir, model_id.split('/')[-1])
            
            snapshot_download(
                repo_id=model_id,
                local_dir=model_dir,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.pdf"],
                local_dir_use_symlinks=False
            )
            print(f"✓ Modelo {model_id} descargado exitosamente en: {model_dir}")
            
        except Exception as e:
            print(f"❌ Error al descargar {model_id}: {str(e)}")

if __name__ == "__main__":
    print("Iniciando descarga de modelos...")
    download_models()
    print("Proceso completado!")