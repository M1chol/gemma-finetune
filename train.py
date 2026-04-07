from trl import SFTConfig, SFTTrainer
from test_dataset import *

input_model_path = "./models/functiongemma-270m-it/"
output_model_path = "./models/functiongemma-270m-it-example-tune"
learning_rate = 5e-5

torch_dtype = model.dtype

args = SFTConfig(
    output_dir=output_model_path,           # directory to save and repository id
    max_length=512,                         # max sequence length for model and packing of the dataset
    packing=False,                          # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=8,                     # number of training epochs
    per_device_train_batch_size=4,          # batch size per device during training
    gradient_checkpointing=False,           # Caching is incompatible with gradient checkpointing
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=1,                        # log every step
    #save_strategy="epoch",                 # save checkpoint every epoch
    eval_strategy="epoch",                  # evaluate checkpoint every epoch
    learning_rate=learning_rate,            # learning rate
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,  # use bfloat16 precision
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                      # Do not push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer
)

trainer.train()
trainer.save_model()

if input("Plot learnign result? y/N: ").lower() == "y":
    import matplotlib.pyplot as plt

    # Access the log history
    log_history = trainer.state.log_history

    # Extract training / validation loss
    train_losses = [log["loss"] for log in log_history if "loss" in log]
    epoch_train = [log["epoch"] for log in log_history if "loss" in log]
    eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    epoch_eval = [log["epoch"] for log in log_history if "eval_loss" in log]

    # Plot the training loss
    plt.plot(epoch_train, train_losses, label="Training Loss")
    plt.plot(epoch_eval, eval_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

if input("Test model against dataset? y/N: ").lower() == 'y':
    result = check_success_rate(trainer.model)
    print("\n\nFinetuned model score: ", result)
