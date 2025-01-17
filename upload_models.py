from huggingface_hub import HfApi, create_repo, upload_folder
import os
from pathlib import Path

def upload_model_to_hf(local_model_path, repo_name):
    """
    Sube un modelo local a Hugging Face Hub
    """

    # Inicializar el API
    api = HfApi()
    # get token from env
    token = os.getenv("HF_TOKEN")
    
    # Nombre completo del repositorio (username/repo_name)
    full_repo_name = f"{api.whoami(token)['name']}/{repo_name}"
    
    print(f"Subiendo modelo desde {local_model_path} a {full_repo_name}...")
    
    # Crear el repositorio (ignorar error si ya existe)
    try:
        create_repo(
            repo_id=full_repo_name,
            token=token,
            repo_type="model",
            exist_ok=True
        )
    except Exception as e:
        print(f"Nota: {str(e)}")
    
    # Subir todos los archivos del modelo
    api.upload_folder(
        folder_path=local_model_path,
        repo_id=full_repo_name,
        repo_type="model",
        token=token
    )
    
    print(f"✓ Modelo subido exitosamente a: https://huggingface.co/{full_repo_name}")

def main():
    # Tu token de Hugging Face (también puedes usar environment variable)
    #token = input("Ingresa tu token de Hugging Face: ")
    
    # Directorio base donde están los modelos
    models_dir = "models"
    
    # Subir cada modelo
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            print(f"\nProcesando {model_name}...")
            upload_model_to_hf(model_path, model_name)

if __name__ == "__main__":
    main()