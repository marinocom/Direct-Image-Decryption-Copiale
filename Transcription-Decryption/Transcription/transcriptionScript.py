# --------------------------
# TRANSCRIPTION SCRIPT using Baro CRNN 

# Trains a Transcription CRNN model on a given dataset
# Input: dataset.json, image_dir, max_width, target_height
# Output: model.pth
# Output: vocabulary.json (transcripted tokens)

# To run python transcriptionScript.py
# Paths are hardcoded
# --------------------------


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import editdistance
import wandb
from jiwer import wer, cer
import os
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def normalize_tokens(tokens):
    """Map BigA-Z to a-z, leave others unchanged."""
    normalized = []
    for t in tokens:
        if t.startswith("Big") and len(t) == 4 and t[3].isalpha():
            normalized.append(t[3].lower())
        else:
            normalized.append(t)
    return normalized

def create_text_overlay_image(image_tensor, ground_truth, prediction, edit_distance):
    """Create an image with text overlay showing ground truth and prediction"""
    # convert tensor to numpy and denormalize
    img_np = image_tensor.cpu().numpy().squeeze()
    
    # convert to RGB for colored text
    img_rgb = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # create a larger canvas to fit text below image
    height, width = img_rgb.shape[:2]
    text_height = 120  # Space for text
    canvas = np.ones((height + text_height, width, 3), dtype=np.uint8) * 255
    
    # place image on canvas
    canvas[:height, :] = img_rgb
    
    # add text with wrapping for long sequences
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    
    # ground truth (green)
    gt_text = f"GT: {ground_truth[:150]}"
    cv2.putText(canvas, gt_text, (5, height + 20), font, font_scale, (0, 150, 0), thickness)
    
    # prediction (blue)
    pred_text = f"Pred: {prediction[:150]}"
    cv2.putText(canvas, pred_text, (5, height + 50), font, font_scale, (0, 0, 200), thickness)
    
    # edit distance (red if high, green if low)
    ed_color = (0, 200, 0) if edit_distance < 0.3 else (200, 100, 0) if edit_distance < 0.6 else (200, 0, 0)
    ed_text = f"Edit Dist: {edit_distance:.3f}"
    cv2.putText(canvas, ed_text, (5, height + 80), font, font_scale, ed_color, thickness)
    
    return canvas

class CopialeDataset(Dataset):
    """Dataset class for Copiale cipher images and transcriptions"""
    
    def __init__(self, data_file: str, image_dir: str, max_width=800, target_height=64, vocab=None):
        """
        Args:
            data_file: JSON file containing image filenames and transcriptions
            image_dir: Directory containing the image files
            max_width: Maximum width to resize images to
            target_height: Fixed height for all images
            vocab: Optional vocabulary dict to use (for val/test sets)
        """
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.image_dir = Path(image_dir)
        self.max_width = max_width
        self.target_height = target_height
        
        # build or use provided vocabulary
        if vocab is None:
            self.build_vocabulary()
        else:
            self.token_to_idx = vocab['token_to_idx']
            self.idx_to_token = vocab['idx_to_token']
            self.vocab_size = vocab['vocab_size']
        
    def build_vocabulary(self):
        """build token vocabulary from all transcriptions"""
        tokens = set()
        for filename, item in self.data.items():
            transcription = item['transcription']
            # split by whitespace to get individual tokens
            token_list = transcription.split()
            tokens.update(token_list)
        
        # add special tokens
        tokens.add('<PAD>')  # Padding token
        tokens.add('<BLANK>')  # CTC blank token
        tokens.add('<UNK>')  # Unknown token
        
        # create token to index mapping
        # Important: <BLANK> must be at index 0 for CTC
        sorted_tokens = ['<BLANK>'] + sorted([t for t in tokens if t != '<BLANK>'])
        self.token_to_idx = {token: idx for idx, token in enumerate(sorted_tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.vocab_size = len(self.token_to_idx)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample tokens: {list(sorted_tokens)[:20]}")  # show first 20 tokens
    
    def get_vocabulary(self):
        """Return vocabulary dictionary for sharing with other datasets"""
        return {
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'vocab_size': self.vocab_size
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename = list(self.data.keys())[idx]
        item = self.data[filename]
        
        # load and preprocess image
        image_path = self.image_dir / filename
        image = self.load_and_preprocess_image(image_path)
        
        # convert transcription to indices (no SOS/EOS for CTC)
        transcription = item['transcription']
        target_indices = self.text_to_indices(transcription)
        
        return {
            'image': image,
            'target': torch.tensor(target_indices, dtype=torch.long),
            'target_length': len(target_indices),
            'transcription': transcription,
            'filename': filename
        }
    
    def load_and_preprocess_image(self, image_path):
        """load and preprocess image with fixed dimensions"""
        # load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # resize to fixed height while maintaining aspect ratio
        original_height, original_width = image.shape
        
        # calculate new width based on target height
        aspect_ratio = original_width / original_height
        new_width = int(self.target_height * aspect_ratio)
        
        # resize image
        image = cv2.resize(image, (new_width, self.target_height))
        
        # pad or crop to max_width
        if new_width > self.max_width:
            image = image[:, :self.max_width]
        elif new_width < self.max_width:
            # pad with white (255) if too narrow
            pad_width = self.max_width - new_width
            image = np.pad(image, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
        
        # NOW all images are exactly (target_height, max_width)
        
        # normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # convert to tensor and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        
        return image
    
    def text_to_indices(self, text):
        """convert text to list of token indices (no SOS/EOS for CTC)"""
        indices = []
        tokens = text.split()  # split by whitespace
        for token in tokens:
            indices.append(self.token_to_idx.get(token, self.token_to_idx['<UNK>']))
        return indices
    
    def indices_to_text(self, indices):
        """convert list of indices back to text"""
        tokens = []
        for idx in indices:
            token = self.idx_to_token.get(idx, '<UNK>')
            if token in ['<BLANK>', '<PAD>']:
                continue
            tokens.append(token)
        return ' '.join(tokens)

class CNNFeatureExtractor(nn.Module):
    """cnn feature extractor for CRNN - optimized for fixed-size images"""
    
    def __init__(self, input_channels=1, output_channels=256):
        super(CNNFeatureExtractor, self).__init__()
        
        self.cnn = nn.Sequential(
            # first conv block
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # h/2, w/2
            
            # second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # h/4, w/4
            
            # third conv block - pool height more aggressively
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # h/8, w/4
            
            # fourth conv block
            nn.Conv2d(256, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        # x shape: (batch_size, 1, height, width)
        features = self.cnn(x)  # (batch_size, 256, height/8, width/4)
        
        # reshape for RNN: (batch_size, width/4, 256 * height/8)
        batch_size, channels, height, width = features.shape
        features = features.permute(0, 3, 1, 2)  # (batch_size, width, channels, height)
        features = features.contiguous().view(batch_size, width, channels * height)
        
        return features

class CRNNModel(nn.Module):
    """crnn model with CTC loss for sequence recognition"""
    
    def __init__(self, vocab_size, hidden_size=256, num_layers=2, dropout=0.5):
        super(CRNNModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # cnn feature extractor
        self.feature_extractor = CNNFeatureExtractor(output_channels=hidden_size)
        
        # calculate input size for LSTM based on CNN output
        # after CNN: height/8, so if target_height is 64, final height is 8
        self.lstm_input_size = hidden_size * 8  # hidden_size * (target_height/8)
        
        # bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # output projection for CTC
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, images):
        batch_size = images.size(0)
        
        # extract cnn features
        features = self.feature_extractor(images)  # (batch_size, seq_len, feature_dim)
        
        # lstm processing
        lstm_out, _ = self.lstm(features)  # (batch_size, seq_len, hidden_size*2)
        
        # classification
        output = self.classifier(self.dropout(lstm_out))  # (batch_size, seq_len, vocab_size)
        
        # for CTC, need log_softmax
        output = torch.nn.functional.log_softmax(output, dim=2)
        
        return output

def collate_fn(batch):
    """collate function for DataLoader"""
    images = [item['image'] for item in batch]
    targets = [item['target'] for item in batch]
    target_lengths = [item['target_length'] for item in batch]
    transcriptions = [item['transcription'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    # Since all images are now the same size, we can simply stack them
    images_tensor = torch.stack(images)
    
    # Pad targets to same length
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return {
        'images': images_tensor,
        'targets': targets_padded,
        'target_lengths': torch.tensor(target_lengths),
        'transcriptions': transcriptions,
        'filenames': filenames
    }

def ctc_decode(predictions, blank_idx=0):
    """Simple CTC greedy decoder"""
    # predictions: (batch_size, seq_len, vocab_size)
    pred_indices = torch.argmax(predictions, dim=-1)  # (batch_size, seq_len)
    
    decoded = []
    for batch_idx in range(pred_indices.size(0)):
        pred = pred_indices[batch_idx].cpu().numpy()
        
        # CTC collapse: remove repeated tokens and blanks
        prev_token = None
        decoded_seq = []
        for token in pred:
            if token != blank_idx and token != prev_token:
                decoded_seq.append(token)
            prev_token = token
        
        decoded.append(decoded_seq)
    
    return decoded

class HTRTrainer:
    """Training class for the CRNN model with CTC loss"""
    
    def __init__(self, model, train_loader, val_loader, test_loader=None, device='cuda', project_name='copiale_crnn_ctc'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        # Get model attributes (handle DataParallel wrapper)
        if isinstance(model, nn.DataParallel):
            model_unwrapped = model.module
        else:
            model_unwrapped = model
        
        # Initialize Weights & Biases
        self.use_wandb = True
        try:
            wandb.init(
                project=project_name,
                config={
                    'model_type': 'CRNN-CTC',
                    'vocab_size': model.vocab_size,
                    'hidden_size': model.hidden_size,
                    'device': str(device),
                    'train_size': len(train_loader.dataset),
                    'val_size': len(val_loader.dataset),
                    'test_size': len(test_loader.dataset) if test_loader else 0
                }
            )
            # Watch model for gradient tracking
            wandb.watch(model, log='all', log_freq=100)
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            self.use_wandb = False
        
        # CTC Loss
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.1
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['images'].to(self.device)
            targets = batch['targets'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            log_probs = self.model(images)  # (batch_size, seq_len, vocab_size)
            
            # Transpose for CTC: (seq_len, batch_size, vocab_size)
            log_probs = log_probs.permute(1, 0, 2)
            
            # Input lengths (all same since images are fixed width)
            input_lengths = torch.full(
                size=(log_probs.size(1),),
                fill_value=log_probs.size(0),
                dtype=torch.long,
                device=self.device
            )
            
            # CTC Loss
            loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log to wandb every 10 batches
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, data_loader, dataset_name="validation", log_images=True):
        """Validate the model on given data loader"""
        self.model.eval()
        total_loss = 0
        correct_tokens = 0
        total_tokens = 0
        num_batches = 0
        edit_distances = []
        predictions_list = []
        references_list = []
        image_previews = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images = batch['images'].to(self.device)
                targets = batch['targets'].to(self.device)
                target_lengths = batch['target_lengths'].to(self.device)
                transcriptions = batch['transcriptions']
                
                # Forward pass
                log_probs = self.model(images)  # (batch_size, seq_len, vocab_size)
                
                # Transpose for CTC loss
                log_probs_t = log_probs.permute(1, 0, 2)  # (seq_len, batch_size, vocab_size)
                
                # Input lengths
                input_lengths = torch.full(
                    size=(log_probs_t.size(1),),
                    fill_value=log_probs_t.size(0),
                    dtype=torch.long,
                    device=self.device
                )
                
                # Calculate CTC loss
                loss = self.criterion(log_probs_t, targets, input_lengths, target_lengths)
                total_loss += loss.item()
                
                # Decode predictions
                decoded_preds = ctc_decode(log_probs, blank_idx=0)
                
                # Process all predictions for metrics
                batch_size = images.size(0)
                for i in range(batch_size):
                    pred_indices = decoded_preds[i]
                    
                    # Get the dataset from the data_loader to access indices_to_text
                    if hasattr(data_loader.dataset, 'indices_to_text'):
                        pred_text = data_loader.dataset.indices_to_text(pred_indices)
                    else:
                        # For subset datasets, get the original dataset
                        pred_text = data_loader.dataset.dataset.indices_to_text(pred_indices)
                    
                    true_text = transcriptions[i]
                    
                    # Clean texts (remove extra spaces, normalize)
                    pred_text = ' '.join(pred_text.split())
                    true_text = ' '.join(true_text.split())
                    
                    # Normalize tokens for metrics
                    pred_tokens = normalize_tokens(pred_text.split())
                    true_tokens = normalize_tokens(true_text.split())
                    
                    predictions_list.append(' '.join(pred_tokens))
                    references_list.append(' '.join(true_tokens))
                    
                    # Calculate token accuracy
                    target_indices = targets[i][:target_lengths[i]].cpu().numpy()
                    
                    # Count correct tokens
                    min_len = min(len(pred_indices), len(target_indices))
                    for j in range(min_len):
                        if pred_indices[j] == target_indices[j]:
                            correct_tokens += 1
                    total_tokens += len(target_indices)
                    
                    # Calculate edit distance on token level
                    edit_dist = editdistance.eval(pred_tokens, true_tokens)
                    normalized_edit_dist = edit_dist / max(len(true_tokens), 1)
                    edit_distances.append(normalized_edit_dist)
                    
                    # Collect image previews (up to 10 examples)
                    if log_images and len(image_previews) < 10:
                        overlay_img = create_text_overlay_image(
                            images[i], 
                            ' '.join(true_tokens),
                            ' '.join(pred_tokens),
                            normalized_edit_dist
                        )
                        image_previews.append({
                            'image': wandb.Image(overlay_img, caption=f"ED: {normalized_edit_dist:.3f}"),
                            'ground_truth': ' '.join(true_tokens),
                            'prediction': ' '.join(pred_tokens),
                            'edit_distance': normalized_edit_dist
                        })
                
                num_batches += 1
        
        # calculate metrics
        avg_loss = total_loss / num_batches
        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        avg_edit_distance = np.mean(edit_distances) if edit_distances else 1.0
        
        # calculate WER and CER using jiwer
        try:
            word_error_rate = wer(references_list, predictions_list)
            char_error_rate = cer(references_list, predictions_list) 
        except:
            # fallback if jiwer fails
            word_error_rate = avg_edit_distance  # Approximate
            char_error_rate = avg_edit_distance   # Approximate
            print(f"Warning: Could not calculate WER/CER with jiwer, using edit distance approximation")
        
        # store validation metrics
        if dataset_name == "validation":
            self.val_losses.append(avg_loss)
            self.val_accuracies.append(token_accuracy)
        
        # log image previews to wandb
        if self.use_wandb and log_images and len(image_previews) > 0:
            wandb.log({
                f"{dataset_name}_image_previews": [ex['image'] for ex in image_previews],
                f"{dataset_name}_predictions_table": wandb.Table(
                    columns=["Ground Truth", "Prediction", "Edit Distance"],
                    data=[[ex['ground_truth'][:100], ex['prediction'][:100], f"{ex['edit_distance']:.3f}"] 
                          for ex in image_previews]
                )
            })
        
        return avg_loss, token_accuracy, avg_edit_distance, word_error_rate, char_error_rate
    
    def test(self):
        """Test the model on test set"""
        if self.test_loader is None:
            print("No test loader provided")
            return None
        
        print("Running final test evaluation...")
        test_loss, test_accuracy, test_edit_distance, test_wer, test_cer = self.validate(
            self.test_loader, "test", log_images=True
        )
        
        print(f"Test Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Token Accuracy: {test_accuracy:.4f}")
        print(f"  Test Edit Distance: {test_edit_distance:.4f}")
        print(f"  Test WER (Word Error Rate): {test_wer:.4f}")
        print(f"  Test CER (Character Error Rate): {test_cer:.4f}")
        
        if self.use_wandb:
            wandb.log({
                'test_loss': test_loss,
                'test_token_accuracy': test_accuracy,
                'test_edit_distance': test_edit_distance,
                'test_wer': test_wer,
                'test_cer': test_cer
            })
        
        return {
            'test_loss': test_loss,
            'test_token_accuracy': test_accuracy,
            'test_edit_distance': test_edit_distance,
            'test_wer': test_wer,
            'test_cer': test_cer
        }
    
    def train(self, num_epochs, save_path='copiale_crnn_ctc_modelOct25Nachsommer.pth'):
        """complete training loop"""
        best_edit_distance = float('inf')
        best_metrics = {}
        
        print("Starting training...")
        for epoch in range(num_epochs):
            # train
            train_loss = self.train_epoch()
            
            # validate (log images every epoch)
            val_loss, val_accuracy, avg_edit_distance, val_wer, val_cer = self.validate(
                self.val_loader, "validation", log_images=True
            )
            
            # update scheduler
            self.scheduler.step(val_loss)
            
            # log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_token_accuracy': val_accuracy,
                'val_edit_distance': avg_edit_distance,
                'val_wer': val_wer,
                'val_cer': val_cer,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            if self.use_wandb:
                wandb.log(metrics)
            
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Token Accuracy: {val_accuracy:.4f}")
            print(f"  Val Edit Distance: {avg_edit_distance:.4f}")
            print(f"  Val WER: {val_wer:.4f}")
            print(f"  Val CER: {val_cer:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # track best model metrics by lowest edit distance
            if avg_edit_distance < best_edit_distance:
                best_edit_distance = avg_edit_distance
                best_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_token_accuracy': val_accuracy,
                    'val_edit_distance': avg_edit_distance,
                    'val_wer': val_wer,
                    'val_cer': val_cer,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }
                print(f"  New best model found at epoch {epoch + 1} (edit distance={avg_edit_distance:.4f})!")
            
            print("-" * 50)
        
            # save the best model after training is complete
        if best_metrics:
            checkpoint = {
                'model_state_dict': best_metrics['model_state_dict'],
                'optimizer_state_dict': best_metrics['optimizer_state_dict'],
                'epoch': best_metrics['epoch'],
                'val_loss': best_metrics['val_loss'],
                'val_token_accuracy': best_metrics['val_token_accuracy'],
                'val_edit_distance': best_metrics['val_edit_distance'],
                'val_wer': best_metrics['val_wer'],
                'val_cer': best_metrics['val_cer']
            }
            torch.save(checkpoint, save_path)
            
            # save to wandb
            if self.use_wandb:
                wandb.save(save_path)
            
            print(f"Training complete! Best model saved from epoch {best_metrics['epoch']} with val_edit_distance {best_metrics['val_edit_distance']:.4f}")
        
        # run final test evaluation
        test_results = self.test()
        
        if self.use_wandb:
            wandb.finish()
            
        return test_results
    
    def plot_training_history(self):
        """plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # accuracy plot
        ax2.plot(self.val_accuracies, label='Val Token Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Token Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_crnn.png', dpi=300, bbox_inches='tight')
        
        # log to wandb
        if self.use_wandb:
            wandb.log({"training_history": wandb.Image('training_history_crnn.png')})
        
        plt.show()

def main():
    """main training function"""

    #wandb.login()
    torch.cuda.empty_cache()
    gc.collect()

    wandb.init(project="", name="")
    

    # Configuration
    config = {
        'train_data_file': '',
        'val_data_file': '',
        'test_data_file': '',
        'train_image_dir': '',
        'val_image_dir': '',
        'test_image_dir': '',
        'batch_size': 8,
        'num_epochs': 8,
        'hidden_size': 256,
        'num_layers': 4,
        'dropout': 0.4,
        'max_width': 800,
        'target_height': 64,
        'project_name': ''
    }
    
    # create datasets
    print("Loading datasets...")
    
    # load training dataset first to build vocabulary
    train_dataset = CopialeDataset(
        data_file=config['train_data_file'],
        image_dir=config['train_image_dir'],
        max_width=config['max_width'],
        target_height=config['target_height']
    )
    
    # get vocabulary from training dataset
    vocab = train_dataset.get_vocabulary()
    
    # export vocabulary for inference
    with open('.json', 'w') as f:
        json.dump(vocab, f, indent=2)
    print("Vocabulary exported to .json")
    
    # load validation and test datasets with shared vocabulary
    val_dataset = CopialeDataset(
        data_file=config['val_data_file'],
        image_dir=config['val_image_dir'],
        max_width=config['max_width'],
        target_height=config['target_height'],
        vocab=vocab
    )
    
    test_dataset = CopialeDataset(
        data_file=config['test_data_file'],
        image_dir=config['test_image_dir'],
        max_width=config['max_width'],
        target_height=config['target_height'],
        vocab=vocab
    )
    
    # create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CRNNModel(
        vocab_size=train_dataset.vocab_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # create trainer and train
    trainer = HTRTrainer(
        model, 
        train_loader, 
        val_loader,
        test_loader,
        device, 
        project_name=config['project_name']
    )
    
    # train model and get test results
    test_results = trainer.train(config['num_epochs'], save_path='.pth')
    
    # plot results
    trainer.plot_training_history()
    
    print("Training completed!")
    if test_results:
        print("Final test results:")
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")



if __name__ == "__main__":
    main()