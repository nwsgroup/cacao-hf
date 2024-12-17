# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning any 🤗 Transformers model for image classification leveraging 🤗 Accelerate."""

import argparse
import json
import logging
import math
import wandb
import os
from pathlib import Path

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
    ColorJitter,
    RandomRotation,
    RandomAffine,
    RandomVerticalFlip,
    RandomGrayscale,
    RandomPerspective
)
from tqdm.auto import tqdm

import transformers
from sklearn.metrics import confusion_matrix
from torchvision.transforms.functional import to_pil_image
import numpy as np
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, SchedulerType, get_scheduler
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.47.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="cifar10",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset)."
        ),
    )
    parser.add_argument("--train_dir", type=str, default=None, help="A folder containing the training data.")
    parser.add_argument("--validation_dir", type=str, default=None, help="A folder containing the validation data.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=0.15,
        help="Percent to split off of train for validation",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="google/vit-base-patch16-224-in21k",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--image_column_name",
        type=str,
        default="image",
        help="The name of the dataset column containing the image data. Defaults to 'image'.",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default="label",
        help="The name of the dataset column containing the labels. Defaults to 'label'.",
    )

    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Si se especifica, ejecuta el entrenamiento del modelo."
    )

    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Si se especifica, ejecuta la evaluación del modelo."
    )

    parser.add_argument(
        "--remove_unused_columns",
        type=str2bool,
        default=True,
        help="Determina si se deben eliminar columnas no utilizadas del conjunto de datos."
    )

    parser.add_argument(
        "--push_to_hub_model_id",
        type=str,
        default=None,
        help="El nombre del repositorio para sincronizar con el directorio local `output_dir`."
    )

    parser.add_argument(
        "--logging_strategy",
        type=str,
        default="steps",
        choices=["steps", "epoch"],
        help="La estrategia para el registro de logs. Puede ser 'steps' o 'epoch'."
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Número de pasos entre cada registro cuando `logging_strategy` es 'steps'."
    )

    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["steps", "epoch"],
        help="La estrategia para la evaluación. Puede ser 'steps' o 'epoch'."
    )

    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["steps", "epoch"],
        help="La estrategia para guardar el modelo. Puede ser 'steps' o 'epoch'."
    )

    parser.add_argument(
        "--load_best_model_at_end",
        type=str2bool,
        default=False,
        help="Determina si se debe cargar el mejor modelo al final del entrenamiento."
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Número máximo de modelos guardados. Los modelos más antiguos se eliminarán cuando se supere este límite."
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Nombre de la ejecución en Weights & Biases (W&B)."
    )
    
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_dir is None and args.validation_dir is None:
        raise ValueError("Need either a dataset name or a training/validation folder.")

    if args.push_to_hub or args.with_tracking:
        if args.output_dir is None:
            raise ValueError(
                "Need an `output_dir` to create a repo when `--push_to_hub` or `with_tracking` is specified."
            )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # Enviar telemetría
    send_example_telemetry("run_image_classification_no_trainer", args)

    # Inicializar el acelerador
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    logger.info(accelerator.state)
    # Configuración de logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Configurar la semilla
    if args.seed is not None:
        set_seed(args.seed)

    # Manejar la creación del repositorio
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Inferir repo_name
            repo_name = args.push_to_hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Crear repo y obtener repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Cargar el dataset
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, trust_remote_code=args.trust_remote_code)
    else:
        data_files = {}
        if args.train_dir is not None:
            data_files["train"] = os.path.join(args.train_dir, "**")
        if args.validation_dir is not None:
            data_files["validation"] = os.path.join(args.validation_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
        )
        # Más detalles: https://huggingface.co/docs/datasets/v2.0.0/en/image_process#imagefolder.

    # Comprobar las columnas del dataset
    dataset_column_names = dataset["train"].column_names if "train" in dataset else dataset["validation"].column_names
    if args.image_column_name not in dataset_column_names:
        raise ValueError(
            f"--image_column_name {args.image_column_name} no encontrado en el dataset '{args.dataset_name}'. "
            "Asegúrate de establecer `--image_column_name` al nombre correcto de la columna de imagen."
        )
    if args.label_column_name not in dataset_column_names:
        raise ValueError(
            f"--label_column_name {args.label_column_name} no encontrado en el dataset '{args.dataset_name}'. "
            "Asegúrate de establecer `--label_column_name` al nombre correcto de la columna de etiquetas."
        )

    # Dividir el dataset si no hay una validación
    if "validation" not in dataset.keys() and args.train_val_split is not None:
        if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
            split = dataset["train"].train_test_split(args.train_val_split)
            dataset["train"] = split["train"]
            dataset["validation"] = split["test"]

    # Preparar las etiquetas
    labels = dataset["train"].features[args.label_column_name].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}

    # Cargar el modelo y el procesador de imágenes preentrenados
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        finetuning_task="image-classification",
        trust_remote_code=args.trust_remote_code,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
    )

    # Preprocesamiento de los datasets
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = (
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
        else Lambda(lambda x: x)
    )
    train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(p=0.5),  
            RandomVerticalFlip(p=0.2),
            RandomRotation(degrees=15),  
            RandomAffine(degrees=0, translate=(0.1, 0.1)),  
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            RandomGrayscale(p=0.1),
            RandomPerspective(distortion_scale=0.2, p=0.5),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_train(example_batch):
        """Aplicar transformaciones de entrenamiento en un batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch[args.image_column_name]
        ]
        return example_batch

    def preprocess_val(example_batch):
        """Aplicar transformaciones de validación en un batch."""
        example_batch["pixel_values"] = [
            val_transforms(image.convert("RGB")) for image in example_batch[args.image_column_name]
        ]
        return example_batch

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Establecer las transformaciones de entrenamiento
        train_dataset = dataset["train"].with_transform(preprocess_train)
        if args.max_eval_samples is not None:
            dataset["validation"] = dataset["validation"].shuffle(seed=args.seed).select(range(args.max_eval_samples))
        # Establecer las transformaciones de validación
        eval_dataset = dataset["validation"].with_transform(preprocess_val)

    # Crear DataLoaders
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[args.label_column_name] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler y cálculo de pasos de entrenamiento
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Preparar todo con el acelerador
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Recalcular los pasos de entrenamiento si es necesario
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Recalcular el número de épocas
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Determinar cuándo guardar los estados del acelerador
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Inicializar rastreadores (trackers) si es necesario
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard no puede registrar Enums, necesitamos el valor crudo
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("image_classification_no_trainer", experiment_config)

    # Cargar funciones de métricas
    metric_accuracy = evaluate.load("accuracy")
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_f1 = evaluate.load("f1")
    # No se carga 'specificity' directamente

    # Calcular el tamaño total del batch
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Iniciando entrenamiento *****")
    logger.info(f"  Número de ejemplos = {len(train_dataset)}")
    logger.info(f"  Número de épocas = {args.num_train_epochs}")
    logger.info(f"  Tamaño de batch instantáneo por dispositivo = {args.per_device_train_batch_size}")
    logger.info(f"  Tamaño total de batch (con paralelo, distribuido & acumulación) = {total_batch_size}")
    logger.info(f"  Pasos de acumulación de gradiente = {args.gradient_accumulation_steps}")
    logger.info(f"  Pasos totales de optimización = {args.max_train_steps}")
    
    # Mostrar la barra de progreso solo una vez en cada máquina
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    resume_step = None  # Inicializar para evitar errores

    # Cargar desde un checkpoint si es necesario
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None and args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Obtener el checkpoint más reciente
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Ordena carpetas por fecha de modificación, el checkpoint más reciente es el último
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Reanudando desde el checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extraer `epoch_{i}` o `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # Necesitamos multiplicar `gradient_accumulation_steps` para reflejar los pasos reales
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # Actualizar la barra de progreso si se carga desde un checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # Saltar los primeros `n` batches en el dataloader al reanudar desde un checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # Seguimiento de la pérdida en cada época
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Verificar si el acelerador ha realizado un paso de optimización detrás de escena
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                    if args.push_to_hub and epoch < args.num_train_epochs - 1:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            args.output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                        if accelerator.is_main_process:
                            image_processor.save_pretrained(args.output_dir)
                            api.upload_folder(
                                commit_message=f"Training in progress epoch {epoch}",
                                folder_path=args.output_dir,
                                repo_id=repo_id,
                                repo_type="model",
                                token=args.hub_token,
                            )

            if completed_steps >= args.max_train_steps:
                break

        # Evaluación después de cada época si está habilitado
        if args.do_eval:
            model.eval()
            all_predictions = []
            all_references = []
            all_images = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1)
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                metric_accuracy.add_batch(
                    predictions=predictions,
                    references=references,
                )
                metric_precision.add_batch(
                    predictions=predictions,
                    references=references,
                )
                metric_recall.add_batch(
                    predictions=predictions,
                    references=references,
                )
                metric_f1.add_batch(
                    predictions=predictions,
                    references=references,
                )

                # Convertir a listas de Python y extender las listas acumuladoras
                all_predictions.extend(predictions.cpu().numpy())
                all_references.extend(references.cpu().numpy())
                all_images.extend(batch["pixel_values"].cpu().numpy())

            # Calcular métricas
            accuracy = metric_accuracy.compute()
            precision = metric_precision.compute(average='weighted')
            recall = metric_recall.compute(average='weighted')
            f1 = metric_f1.compute(average='weighted')

            # Calcular especificidad manualmente
            cm = confusion_matrix(all_references, all_predictions)
            specificity_per_class = []
            for i in range(len(cm)):
                TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
                FP = cm[:, i].sum() - cm[i, i]
                specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
                specificity_per_class.append(specificity)
            specificity = {
                'specificity_per_class': specificity_per_class,
                'mean_specificity': sum(specificity_per_class) / len(specificity_per_class)
            }

            eval_metric = accuracy  # Puedes combinar todas las métricas si lo deseas
            logger.info(f"Época {epoch}: {eval_metric}")

            if args.with_tracking:
                accelerator.log(
                    {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "specificity": specificity,
                        "train_loss": total_loss.item() / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

            # Seleccionar una muestra de imágenes para loguear
            num_images_to_log = 10  # Puedes ajustar este número según tus necesidades
            num_images_to_log = min(num_images_to_log, len(all_images))
            sample_indices = np.random.choice(len(all_images), num_images_to_log, replace=False)
            sample_images = [all_images[i] for i in sample_indices]
            sample_predictions = [all_predictions[i] for i in sample_indices]
            sample_references = [all_references[i] for i in sample_indices]

            pil_images = []
            captions = []
            for idx in range(num_images_to_log):
                img = sample_images[idx]
                pred = sample_predictions[idx]
                ref = sample_references[idx]
                
                # Si las imágenes fueron normalizadas, desnormalizarlas para una visualización correcta
                if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std"):
                    img = img * np.array(image_processor.image_std).reshape(-1, 1, 1) + np.array(image_processor.image_mean).reshape(-1, 1, 1)
                    img = np.clip(img, 0, 1)

                # Convertir a tensor de PyTorch
                img_tensor = torch.tensor(img)

                # Convertir a PIL Image
                pil_img = to_pil_image(img_tensor)
                pil_images.append(pil_img)

                # Preparar el caption con la etiqueta real y predicha
                caption = f"Real: {labels[ref]}, Predicción: {labels[pred]}"
                captions.append(caption)

            # Logear la matriz de confusión y las imágenes en W&B
            if args.with_tracking and 'wandb' in args.report_to:
                wandb_api_key = os.getenv('WANDB_API_KEY')
                if not wandb_api_key:
                    raise ValueError("W&B API key not found in the environment. Please set the WANDB_API_KEY environment variable.")
                wandb.login(key=wandb_api_key)
                conf_matrix = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_references,
                    preds=all_predictions,
                    class_names=labels
                )
                wandb.log({"conf_mat": conf_matrix}, step=completed_steps)

                # Crear una lista de objetos wandb.Image con sus captions
                wandb_images = [wandb.Image(img, caption=cap) for img, cap in zip(pil_images, captions)]

                # Registrar las imágenes en W&B
                wandb.log({"Predicciones_vs_Real": wandb_images}, step=completed_steps)

            # Reiniciar las métricas
            metric_accuracy = evaluate.load("accuracy")
            metric_precision = evaluate.load("precision")
            metric_recall = evaluate.load("recall")
            metric_f1 = evaluate.load("f1")
            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    image_processor.save_pretrained(args.output_dir)
                    api.upload_folder(
                        commit_message=f"Training in progress epoch {epoch}",
                        folder_path=args.output_dir,
                        repo_id=repo_id,
                        repo_type="model",
                        token=args.hub_token,
                    )

            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

    # Finalizar el entrenamiento
    if args.with_tracking and accelerator.is_main_process:
        wandb.finish()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            image_processor.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            all_results = {
                "eval_accuracy": accuracy["accuracy"],
                "eval_precision": precision["precision"],
                "eval_recall": recall["recall"],
                "eval_f1": f1["f1"],
                "eval_specificity_mean": specificity["mean_specificity"],
                # Añadir otras métricas si lo deseas
            }
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)

if __name__ == "__main__":
    main()
