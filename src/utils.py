import matplotlib.pyplot as plt

def plot_training_logs(train_logs):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    # Loss
    ax[0].plot(train_logs['train_loss'], label="train")
    ax[0].plot(train_logs['val_loss'], label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].grid(True)

    # Validation metric
    ax[1].plot(train_logs['val_metric'], label="val metric", color="tab:orange")
    ax[1].set_title("Validation Metric")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Metric")
    ax[1].grid(True)

    # Learning rate
    ax[2].plot(train_logs['lr'], label="lr", color="tab:green")
    ax[2].set_title("Learning Rate")
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("LR")
    ax[2].grid(True)

    plt.tight_layout();