import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset # Thêm import cần thiết
import numpy as np
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from phoMT import collate_fn, phoMTDataset
from vocab import Vocab
from LSTM_Bahdanau_attn import Seq2seqLSTM_Bahdanau_attn
# Import Metrics
try:
    from torchmetrics.text.rouge import ROUGEScore
except ImportError:
    print("Vui lòng cài đặt torchmetrics: pip install torchmetrics")
    exit()

# --- Config ---
device = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "/content/drive/MyDrive/Kì_1_Năm_3/DS200/DL-LAB4/2_LSTM_Bahdanau_attn/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(CHECKPOINT_DIR, "training.log"), mode='a'),
        logging.StreamHandler()
    ],
    force=True
)

def indices_to_text(indices, vocab, is_target=True):
    tokens = []
    # Chọn từ điển phù hợp
    i2s = vocab.tgt_i2s if is_target else vocab.src_i2s

    for idx in indices:
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        # Dừng nếu gặp <eos> (chỉ có trong Target/Translation)
        if is_target and idx == vocab.eos_idx:
            break

        if idx != vocab.pad_idx:
            # Bỏ qua <bos> chỉ cho Target/Translation
            if is_target and idx == vocab.bos_idx:
                continue

            # Dùng .get để an toàn, nếu không tìm thấy thì trả về <unk>
            token = i2s.get(idx, vocab.unk_token)
            tokens.append(token)

        # Dừng nếu gặp <eos> (chỉ có trong Source)
        if not is_target and idx == vocab.eos_idx:
             break

    return " ".join(tokens)

def train(model: nn.Module, dataloader: DataLoader, epoch: int, loss_fn, optimizer):
    model.train()
    running_loss = []

    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as pbar:
        for item in pbar:
            src = item['src'].to(device)
            tgt = item['tgt'].to(device)

            optimizer.zero_grad()

            # Teacher forcing
            decoder_input = tgt[:, :-1]
            targets = tgt[:, 1:]

            logits = model(src, decoder_input)

            # Flatten để tính loss
            # logits: (bs*len, vocab_size), targets: (bs*len)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss.append(loss.item())
            pbar.set_postfix({"loss": np.mean(running_loss)})

    avg_loss = np.mean(running_loss)
    logging.info(f"Epoch {epoch} - Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model: nn.Module, dataloader: DataLoader, epoch: int, loss_fn, vocab):
    model.eval()
    running_loss = []
    rouge_metric = ROUGEScore().to(device)

    all_preds_text = []
    all_targets_text = []

    # Flag để chỉ in ra một ví dụ duy nhất mỗi epoch
    example_printed = False

    with torch.no_grad():
        for item in tqdm(dataloader, desc=f"Epoch {epoch} - Evaluating"):
            src = item['src'].to(device)
            tgt = item['tgt'].to(device)

            # 1. Validation Loss
            decoder_input = tgt[:, :-1]
            targets = tgt[:, 1:]
            logits = model(src, decoder_input)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            running_loss.append(loss.item())

            # 2. ROUGE-L Prediction
            # Sinh văn bản (Inference)
            generated_tokens = model.predict(src, max_len=tgt.shape[1] + 10)

            # Xử lý trường hợp generated_tokens rỗng
            if generated_tokens.shape[1] == 0:
                generated_tokens = torch.zeros(tgt.shape[0], 1, dtype=torch.long, device=device)

            for i in range(len(tgt)):
                # Decode output của model (Translation)
                pred_seq = generated_tokens[i].tolist()
                pred_text = indices_to_text(pred_seq, vocab, is_target=True)

                # Decode nhãn thật (Reference)
                tgt_seq = tgt[i].tolist()
                tgt_text = indices_to_text(tgt_seq, vocab, is_target=True)

                # LOGIC IN VÍ DỤ
                if not example_printed:
                    # Decode Source (English)
                    src_seq = src[i].tolist()
                    src_text = indices_to_text(src_seq, vocab, is_target=False)

                    logging.info(f"\n--- Example Translation (Epoch {epoch}) ---")
                    logging.info(f"Source:      {src_text}")
                    logging.info(f"Reference:   {tgt_text}")
                    logging.info(f"Translation: {pred_text}")
                    logging.info("-" * 45)
                    example_printed = True # Đảm bảo chỉ in một lần

                all_preds_text.append(pred_text)
                all_targets_text.append(tgt_text)

    # Tính ROUGE trên tập Validation
    if len(all_preds_text) > 0:
        rouge_scores = rouge_metric(all_preds_text, all_targets_text)
        rouge_l = rouge_scores['rougeL_fmeasure'].item()
    else:
        rouge_l = 0.0

    avg_loss = np.mean(running_loss)
    logging.info(f"Epoch {epoch} - Val Loss: {avg_loss:.4f} | ROUGE-L: {rouge_l:.4f}")

    return avg_loss, rouge_l

def visualize_metrics(train_losses, val_losses, rouge_scores):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='s')
    plt.title("Loss History")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, rouge_scores, label='Val ROUGE-L', marker='^', color='green')
    plt.title("ROUGE-L Score History")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    logging.info("="*50)
    logging.info(f"Starting training on Device: {device}")
    vocab = Vocab(
        path="/content/drive/MyDrive/Kì_1_Năm_3/DS200/DL-LAB4/dataset",
        src_language="english",
        tgt_language="vietnamese"
    )

    # Lấy toàn bộ dataset
    train_dataset_full = phoMTDataset("/content/drive/MyDrive/Kì_1_Năm_3/DS200/DL-LAB4/dataset/train.json", vocab)
    dev_dataset_full = phoMTDataset("/content/drive/MyDrive/Kì_1_Năm_3/DS200/DL-LAB4/dataset/dev.json", vocab)
    test_dataset_full = phoMTDataset("/content/drive/MyDrive/Kì_1_Năm_3/DS200/DL-LAB4/dataset/test.json", vocab)

    train_size = 20000
    dev_test_size = 2000

    # Lấy ngẫu nhiên 20k train
    train_indices = torch.randperm(len(train_dataset_full))[:train_size].tolist()
    train_dataset = Subset(train_dataset_full, train_indices)

    # Lấy ngẫu nhiên 2k dev
    dev_indices = torch.randperm(len(dev_dataset_full))[:dev_test_size].tolist()
    dev_dataset = Subset(dev_dataset_full, dev_indices)

    # Lấy ngẫu nhiên 2k test
    test_indices = torch.randperm(len(test_dataset_full))[:dev_test_size].tolist()
    test_dataset = Subset(test_dataset_full, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Khởi tạo model (sử dụng Bahdanau Attention đã sửa ở bước trước)
    model =  Seq2seqLSTM_Bahdanau_attn(
        d_model=256,
        n_encoder=3,
        n_decoder=3,
        dropout=0.3,
        vocab=vocab
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_lstm_mt.pt")
    best_rouge = 0.0
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        best_rouge = ckpt.get('best_rouge', 0.0)
        logging.info(f"Resumed from epoch {start_epoch}, Best ROUGE: {best_rouge:.4f}")

    train_losses, val_losses, val_rouges = [], [], []
    patience = 0

    for epoch in range(start_epoch + 1, 20):
        logging.info(f"\n--- Epoch {epoch} ---")

        t_loss = train(model, train_loader, epoch, loss_fn, optimizer)
        # Trong evaluate sẽ in ra 1 ví dụ
        v_loss, v_rouge = evaluate(model, dev_loader, epoch, loss_fn, vocab)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        val_rouges.append(v_rouge)

        if v_rouge > best_rouge:
            best_rouge = v_rouge
            patience = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_rouge': best_rouge
            }, checkpoint_path)
            logging.info(f"✓ Saved Best Model (ROUGE-L: {best_rouge:.4f})")
        else:
            patience += 1
            logging.info(f"No improvement. Patience: {patience}/10")
            if patience >= 10:
                logging.info("Early stopping!")
                break

    logging.info("\nEvaluating on Test Set with Best Model...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_rouge = evaluate(model, test_loader, 0, loss_fn, vocab)
    logging.info(f"Test Loss: {test_loss:.4f} | Test ROUGE-L: {test_rouge:.4f}")

    visualize_metrics(train_losses, val_losses, val_rouges)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
