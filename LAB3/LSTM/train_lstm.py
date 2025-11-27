import torch
from uit_vsfc import collate_fn, Vocab, UIT_VSFC
from lstm import LSTMClassifier
from tqdm import tqdm
from torch import nn, optim
import numpy as np
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "/content/drive/MyDrive/Kì_1_Năm_3/DS200/DL-LAB3/1_LSTM/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Configure logging with flush
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(CHECKPOINT_DIR, "training.log"), mode='a'),
        logging.StreamHandler()
    ],
    force=True  # Override any existing logging config
)

# Force flush after each log
for handler in logging.root.handlers:
    handler.setLevel(logging.INFO)
    if isinstance(handler, logging.FileHandler):
        handler.flush()

def train(model: nn.Module, dataloader: DataLoader, epoch: int, loss_fn, optimizer):
    model.train()
    running_loss = []
    all_preds = []
    all_labels = []
    
    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            labels = item["label"].to(device)

            # forward pass
            logits = model(input_ids)  # Shape: (batch_size, n_labels)
            
            # No reshaping needed - already correct shape
            loss = loss_fn(logits, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({"loss": np.array(running_loss).mean()})

    avg_loss = np.array(running_loss).mean()
    train_f1 = f1_score(all_labels, all_preds, average="macro")
    
    logging.info(f"Epoch {epoch} - Training Loss: {avg_loss:.4f} | F1: {train_f1:.4f}")
    
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
    return avg_loss, train_f1


def evaluate(model: nn.Module, dataloader: DataLoader, epoch: int, loss_fn) -> float:
    model.eval()
    running_loss = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for item in tqdm(dataloader, desc=f"Epoch {epoch} - Evaluating"):
            input_ids = item["input_ids"].to(device)
            labels = item["label"].to(device)

            logits = model(input_ids)  # Shape: (batch_size, n_labels)
            
            loss = loss_fn(logits, labels)
            running_loss.append(loss.item())
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = np.array(running_loss).mean()
    f1 = f1_score(all_labels, all_preds, average="macro")
    logging.info(f"Epoch {epoch} - Validation Loss: {avg_loss:.4f} | F1: {f1:.4f}")

    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()

    return avg_loss, f1

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load checkpoint if exists"""
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_f1 = checkpoint['best_f1']
        logging.info(f"Resumed from epoch {start_epoch} with best F1={best_f1:.4f}")
        return start_epoch, best_f1, True
    return 0, 0.0, False

def visualize_metrics(train_losses, val_losses, train_f1s, val_f1s):
    epochs = range(len(train_losses))

    plt.figure(figsize=(15, 6))

    # --- Biểu đồ 1: Loss History ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', color='tab:blue')
    plt.plot(epochs, val_losses, label='Val Loss', marker='s', color='tab:red') 
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # --- Biểu đồ 2: F1 Score History ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_f1s, label='Train F1', marker='o', color='tab:blue')
    plt.plot(epochs, val_f1s, label='Val F1', marker='s', color='tab:red')
    plt.title("F1 Score History")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    logging.info("="*50)
    logging.info("Starting training script...")
    logging.info(f"Device: {device}")

    try:
        logging.info("Loading vocab...")
        vocab = Vocab(path="/content/drive/MyDrive/Kì_1_Năm_3/DS200/DL-LAB3/UIT-VSFC")

        logging.info("Creating dataset...")
        train_dataset = UIT_VSFC(
            data_dir="/content/drive/MyDrive/Kì_1_Năm_3/DS200/DL-LAB3/UIT-VSFC/UIT-VSFC-train.json",
            vocab=vocab
        )

        test_dataset = UIT_VSFC(
            data_dir="/content/drive/MyDrive/Kì_1_Năm_3/DS200/DL-LAB3/UIT-VSFC/UIT-VSFC-test.json",
            vocab=vocab
        )

        dev_dataset = UIT_VSFC(
            data_dir="/content/drive/MyDrive/Kì_1_Năm_3/DS200/DL-LAB3/UIT-VSFC/UIT-VSFC-dev.json",
            vocab=vocab
        )

        logging.info("Creating dataloaders...")
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn
        )

        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=32,
            shuffle=False,  # No need to shuffle test set
            collate_fn=collate_fn
        )

        dev_dataloader = DataLoader(
            dataset=dev_dataset,
            batch_size=32,
            shuffle=False,  # No need to shuffle validation set
            collate_fn=collate_fn
        )

        logging.info("Initializing model...")
        model = LSTMClassifier(
            vocab_size=vocab.len,
            hidden_size=256,
            n_labels=vocab.n_labels
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss().to(device)

        # Check for existing checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_lstm.pt")
        start_epoch, best_f1, checkpoint_loaded = load_checkpoint(model, optimizer, checkpoint_path)

        if checkpoint_loaded:
            logging.info("Checkpoint found! Evaluating on test set...")
            test_f1 = evaluate(model, test_dataloader, start_epoch)
            logging.info(f"F1-score on test set: {test_f1:.4f}")
            logging.info("Skipping training since checkpoint exists.")
        else:
            logging.info("No checkpoint found. Starting training from scratch...")
            epoch = start_epoch
            patience = 0
            train_losses_history = []
            val_losses_history = []
            train_f1s_history = []
            val_f1s_history = []
            while True:
                epoch += 1
                logging.info(f"\n{'='*50}")
                logging.info(f"Starting Epoch {epoch}")
                logging.info(f"{'='*50}")

                train_loss, train_f1 = train(model, train_dataloader, epoch, loss_fn, optimizer)
  
                val_loss, val_f1 = evaluate(model, dev_dataloader, epoch, loss_fn)
                train_losses_history.append(train_loss)
                val_losses_history.append(val_loss)
                train_f1s_history.append(train_f1)
                val_f1s_history.append(val_f1)

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    patience = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_f1': best_f1
                    }, checkpoint_path)
                    logging.info(f"✓ Saved new best checkpoint with F1={best_f1:.4f}")
                else:
                    patience += 1
                    logging.info(f"No improvement. Patience: {patience}/10")

                if patience == 10:
                    logging.info("\n" + "="*50)
                    logging.info("Early stopping triggered. Training complete.")
                    logging.info("="*50)
                    break

            # Evaluate on test set
            logging.info("\nLoading best model for final evaluation...")
            logging.info("Visualizing training curves...")
            visualize_metrics(train_losses_history, val_losses_history, train_f1s_history, val_f1s_history)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            # Hứng 2 biến: test_loss và test_f1. Nhớ truyền thêm loss_fn
            test_loss, test_f1 = evaluate(model, test_dataloader, start_epoch, loss_fn) 
            logging.info(f"F1-score on test set: {test_f1:.4f}")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        raise
    finally:
        # Final flush
        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        logging.info("Script finished.")
