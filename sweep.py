import wandb
api = wandb.Api()

# Replace with your sweep ID
sweep_id = "cristianrey/cocoa-image-classification/your-sweep-id"
sweep = api.sweep(sweep_id)

# Get the best run
best_run = sweep.best_run()

# Get the best hyperparameters
best_config = best_run.config
print("Best hyperparameters found:")
print(f"Learning rate: {best_config['learning_rate']}")
print(f"Number of epochs: {best_config['num_train_epochs']}")
print(f"Batch size: {best_config['per_device_train_batch_size']}")
print(f"LR scheduler: {best_config['lr_scheduler_type']}")
print(f"Weight decay: {best_config['weight_decay']}")
print(f"Best accuracy achieved: {best_run.summary.get('eval_accuracy')}")