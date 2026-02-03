"""
DIRECT IMAGE DECRYPTION model creation and training (8-head attention).

Purpose
-------
Defines and trains a direct image-to-plaintext DIRECT IMAGE DECRYPTION model: cipher images
are mapped to plaintext character sequences in one end-to-end pass. No separate OCR or transcription step is required.

Architecture
------------
- Encoder (CRNN-style): A CRNN feature extractor turns each image into a
  sequence of vectors. It consists of:
  - CNN: 5 conv blocks (3x3, BatchNorm, ReLU) with progressive pooling
    (e.g. H/16, W/4), producing a feature map that is reshaped to (batch,
    seq_len, feature_dim) along the width dimension.
  - Bidirectional LSTM over that sequence, yielding encoder output
    (batch, seq_len, hidden_size*2).
- Decoder: An LSTM decoder consumes previous plaintext character embeddings
  (with optional teacher forcing during training) and attends over the encoder
  output via multi-head scaled dot-product attention (8 heads). A linear layer
  projects the decoder hidden state to the plaintext vocabulary size for
  character-level prediction.
- Special tokens: <PAD>, <SOS>, <EOS>, <UNK> for sequence boundaries and
  unknown characters.

Data and vocabulary
-------------------
- Dataset: JSON mapping image filenames to {"plaintext": "..."}. Images are
  loaded from a configurable directory, resized to a fixed height and
  max width, normalized, and fed as (batch, 1, H, W).
- Vocabulary: Built from the training set only (all unique characters in
  plaintexts plus special tokens). The same vocabulary is shared with
  validation and test via get_vocabulary() / vocab_plaintext to ensure
  consistent indices. Long sequences beyond max_seq_len are filtered out.

Training
--------
- ImageDecryptionTrainer handles training loop, validation, and test evaluation
  (e.g. character/sequence accuracy, edit distance, BLEU, WER/CER). Checkpoints
  and optional Weights & Biases logging are supported.
- Paths, batch size, epochs, embed/hidden sizes, dropout, and project name are
  set in the config dict in main(). The plaintext vocabulary is exported (e.g.
  to JSON) for use by inference scripts.

Config paths in main 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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
import random
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class CopialeImageDecryptionDataset(Dataset):
    """Dataset class for direct image-to-decryption (plaintext) conversion"""
    
    def __init__(self, data_file: str, image_dir: str, max_width=800, target_height=64, 
                 vocab_plaintext=None, max_seq_len=200):
        """
        Args:
            data_file: JSON file containing image filenames and plaintexts
            image_dir: Directory containing the image files
            max_width: Maximum width to resize images to
            target_height: Fixed height for all images
            vocab_plaintext: Optional vocabulary dict for plaintext characters
            max_seq_len: Maximum sequence length for truncation
        """
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.image_dir = Path(image_dir)
        self.max_width = max_width
        self.target_height = target_height
        self.max_seq_len = max_seq_len
        
        # build vocabulary if not provided
        if vocab_plaintext is None:
            self.build_vocabulary()
        else:
            self.plaintext_vocab = vocab_plaintext
        
        # filter out sequences that are too long
        self.filter_long_sequences()
        
    def build_vocabulary(self):
        """Build vocabulary for plaintext characters"""
        plaintext_chars = set()
        
        for filename, item in self.data.items():
            plaintext = item['plaintext']
            # add all characters from plaintext (including spaces)
            plaintext_chars.update(list(plaintext))
        
        # add special tokens
        special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        plaintext_chars.update(special_tokens)
        
        # build plaintext vocabulary
        sorted_chars = special_tokens + sorted([c for c in plaintext_chars if c not in special_tokens])
        
        self.plaintext_vocab = {
            'char_to_idx': {char: idx for idx, char in enumerate(sorted_chars)},
            'idx_to_char': {idx: char for idx, char in enumerate(sorted_chars)},
            'vocab_size': len(sorted_chars)
        }
        
        print(f"Plaintext vocabulary size: {self.plaintext_vocab['vocab_size']}")
        print(f"Sample plaintext chars: {sorted_chars[:20]}")
    
    def filter_long_sequences(self):
        """Filter out sequences that exceed max_seq_len"""
        original_count = len(self.data)
        filtered_data = {}
        
        for filename, item in self.data.items():
            plaintext_chars = list(item['plaintext'])
            
            # keep sequences within length limits (accounting for SOS/EOS tokens)
            if len(plaintext_chars) <= self.max_seq_len - 2:
                filtered_data[filename] = item
        
        self.data = filtered_data
        print(f"Filtered dataset: {original_count} -> {len(self.data)} samples "
              f"(removed {original_count - len(self.data)} long sequences)")
    
    def get_vocabulary(self):
        """Return vocabulary for sharing with other datasets"""
        return self.plaintext_vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # get the filename and item data
        filename = list(self.data.keys())[idx]
        item = self.data[filename]
        
        # load and preprocess image
        image_path = self.image_dir / filename
        image = self.load_and_preprocess_image(image_path)
        
        # convert plaintext to indices
        plaintext = item['plaintext']
        target_indices = self.plaintext_to_indices(plaintext)
        
        return {
            'image': image,
            'target_seq': torch.tensor(target_indices, dtype=torch.long),
            'target_length': len(target_indices),
            'plaintext': plaintext,
            'filename': filename
        }
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image with fixed dimensions"""
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
            # crop if too wide
            image = image[:, :self.max_width]
        elif new_width < self.max_width:
            # pad with white (255) if too narrow
            pad_width = self.max_width - new_width
            image = np.pad(image, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
        
        # normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        
        return image
    
    def plaintext_to_indices(self, plaintext):
        """Convert plaintext characters to indices"""
        indices = [self.plaintext_vocab['char_to_idx']['<SOS>']]
        for char in plaintext:
            indices.append(self.plaintext_vocab['char_to_idx'].get(char, self.plaintext_vocab['char_to_idx']['<UNK>']))
        indices.append(self.plaintext_vocab['char_to_idx']['<EOS>'])
        return indices
    
    def indices_to_plaintext(self, indices):
        """Convert indices back to plaintext"""
        chars = []
        for idx in indices:
            char = self.plaintext_vocab['idx_to_char'].get(idx, '<UNK>')
            if char in ['<SOS>', '<EOS>', '<PAD>']:
                continue
            chars.append(char)
        return ''.join(chars)


class CRNNFeatureExtractor(nn.Module):
    """CRNN-based feature extractor for cipher symbols (inspired by Baró et al.)"""
    
    def __init__(self, input_channels=1, hidden_size=256, num_layers=2, dropout=0.3):
        super(CRNNFeatureExtractor, self).__init__()
        
        self.hidden_size = hidden_size
        
        # convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # H/2, W/2
            
            # block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # H/4, W/4
            
            # block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # H/8, W/4 - pool height more aggressively
            
            # block 4 - deeper features for symbol understanding
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # H/16, W/4
        )
        
        # bidirectional LSTM for sequential modeling (like CRNN)
        # input: (batch, seq_len, feature_dim)
        # after conv: height/16, so if target_height=64, final_height=4
        self.feature_dim = 512 * 4  # channels * height after pooling
        
        self.rnn = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, 1, height, width)
        Returns:
            features: (batch_size, seq_len, hidden_size*2)
        """
        # extract CNN features
        conv_features = self.conv_layers(x)  # (batch, 512, H/16, W/4)
        
        # reshape for RNN: (batch, width, channels*height)
        batch_size, channels, height, width = conv_features.shape
        conv_features = conv_features.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        conv_features = conv_features.contiguous().view(batch_size, width, channels * height)
        
        # Apply RNN for sequential modeling
        rnn_output, _ = self.rnn(conv_features)  # (batch, seq_len, hidden_size*2)
        
        return self.dropout(rnn_output)


class CopialeImageDecryptionModelWithCRNN(nn.Module):
    """Direct image-to-decryption model using CRNN + Seq2Seq architecture"""
    
    def __init__(self, plaintext_vocab_size, embed_size=128, hidden_size=256, 
                 num_layers=2, dropout=0.3, plaintext_vocab=None, 
                 pretrained_crnn_path=None):
        super(CopialeImageDecryptionModelWithCRNN, self).__init__()
        
        self.plaintext_vocab_size = plaintext_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # store vocabulary for getting special token indices
        self.plaintext_vocab = plaintext_vocab
        
        # get special token indices
        if plaintext_vocab:
            self.sos_idx = plaintext_vocab.get('char_to_idx', {}).get('<SOS>', 1)
            self.eos_idx = plaintext_vocab.get('char_to_idx', {}).get('<EOS>', 2)
            self.pad_idx = plaintext_vocab.get('char_to_idx', {}).get('<PAD>', 0)
        else:
            # fallback values
            self.sos_idx = 1
            self.eos_idx = 2
            self.pad_idx = 0
        
        # CRNN feature extractor (inspired by transcription model)
        self.feature_extractor = CRNNFeatureExtractor(
            input_channels=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # load pretrained CRNN weights if provided
        if pretrained_crnn_path and os.path.exists(pretrained_crnn_path):
            print(f"Loading pretrained CRNN weights from {pretrained_crnn_path}")
            self._load_pretrained_crnn(pretrained_crnn_path)
        
        # the CRNN output is already bidirectional LSTM features: (batch, seq_len, hidden_size*2)
        # so we can use these directly as encoder output
        
        # plaintext embedding for decoder
        self.plaintext_embedding = nn.Embedding(plaintext_vocab_size, embed_size)
        
        # decoder LSTM (generates plaintext characters)
        self.decoder = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size * 2,  # match encoder output size (bidirectional)
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # output projection
        self.output_projection = nn.Linear(hidden_size * 2, plaintext_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def _load_pretrained_crnn(self, pretrained_path):
        """Load pretrained CRNN weights (if available from transcription model)"""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # extract only the feature extractor weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # filter to only load compatible layers
            crnn_state_dict = {}
            for key, value in state_dict.items():
                # map the keys appropriately
                if key.startswith('cnn.') or key.startswith('conv_layers.'):
                    new_key = key.replace('cnn.', 'conv_layers.')
                    crnn_state_dict[new_key] = value
                elif key.startswith('rnn.'):
                    crnn_state_dict[key] = value
            
            # load with strict=False to allow partial loading
            self.feature_extractor.load_state_dict(crnn_state_dict, strict=False)
            print("Successfully loaded pretrained CRNN weights (partial)")
            
        except Exception as e:
            print(f"Warning: Could not load pretrained CRNN weights: {e}")
            print("Training CRNN from scratch")
    
    def forward(self, images, target_seq=None, max_length=None):
        """
        Forward pass
        Args:
            images: Input images (batch_size, 1, height, width)
            target_seq: Target plaintext character indices for training (batch_size, target_seq_len)
            max_length: Maximum length for inference
        """
        batch_size = images.size(0)
        
        # extract CRNN features - this gives us encoded representations
        encoder_output = self.feature_extractor(images)  # (batch_size, seq_len, hidden_size*2)
        
        # create encoder hidden and cell states from the last time step
        # for compatibility with the decoder initialization
        final_features = encoder_output[:, -1, :]  # (batch_size, hidden_size*2)
        
        # split bidirectional features and create initial states
        encoder_hidden = final_features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        encoder_cell = torch.zeros_like(encoder_hidden)
        
        if self.training and target_seq is not None:
            # Training mode: use teacher forcing
            return self._forward_train(encoder_output, target_seq, encoder_hidden, encoder_cell)
        else:
            # inference mode: generate sequence
            max_len = max_length or 200
            return self._forward_inference(encoder_output, encoder_hidden, encoder_cell, batch_size, max_len)
    
    def _forward_train(self, encoder_output, target_seq, encoder_hidden, encoder_cell):
        """Training forward pass with teacher forcing"""
        # prepare decoder input (shift target by one position)
        decoder_input = target_seq[:, :-1]  # remove last token
        target_output = target_seq[:, 1:]   # remove first token (SOS)
        
        # embed decoder input
        decoder_embedded = self.plaintext_embedding(decoder_input)
        
        # decode
        decoder_output, _ = self.decoder(decoder_embedded, (encoder_hidden, encoder_cell))
        
        # apply attention
        attended_output, _ = self.attention(decoder_output, encoder_output, encoder_output)
        
        # generate output probabilities
        output = self.output_projection(self.dropout(attended_output))
        
        return output, target_output
    
    def _forward_inference(self, encoder_output, encoder_hidden, encoder_cell, batch_size, max_length):
        """Inference forward pass with greedy decoding"""
        # start with SOS token
        decoder_input = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=encoder_output.device)
        
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        
        outputs = []
        
        for step in range(max_length):
            # embed current input
            decoder_embedded = self.plaintext_embedding(decoder_input)
            
            # decode one step
            decoder_output, (decoder_hidden, decoder_cell) = self.decoder(
                decoder_embedded, (decoder_hidden, decoder_cell)
            )
            
            # Apply attention
            attended_output, _ = self.attention(decoder_output, encoder_output, encoder_output)
            
            # generate output probabilities
            output = self.output_projection(self.dropout(attended_output))
            
            # get predicted token
            predicted_token = torch.argmax(output, dim=-1)
            outputs.append(predicted_token)
            
            # use predicted token as next input
            decoder_input = predicted_token
            
            # check if all sequences have generated EOS
            if torch.all(predicted_token == self.eos_idx):
                break
        
        if len(outputs) == 0:
            # return empty sequence if no outputs generated
            return torch.zeros((batch_size, 1), dtype=torch.long, device=encoder_output.device)
        
        return torch.cat(outputs, dim=1)


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    images = [item['image'] for item in batch]
    target_seqs = [item['target_seq'] for item in batch]
    target_lengths = [item['target_length'] for item in batch]
    plaintexts = [item['plaintext'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    # stack images (all same size)
    images_tensor = torch.stack(images)
    
    # pad target sequences
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)
    
    return {
        'images': images_tensor,
        'target_seqs': target_seqs_padded,
        'target_lengths': torch.tensor(target_lengths),
        'plaintexts': plaintexts,
        'filenames': filenames
    }


class ImageDecryptionTrainer:
    """Training class for the image decryption model with CRNN"""
    
    def __init__(self, model, train_loader, val_loader, test_loader=None, device='cuda', 
                 project_name='copiale_image_decryption_crnn'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Initialize Weights & Biases
        self.use_wandb = True
        try:
            wandb.init(
                project=project_name,
                config={
                    'model_type': 'CRNN-Seq2Seq-Attention-DirectDecryption',
                    'plaintext_vocab_size': model.plaintext_vocab_size,
                    'hidden_size': model.hidden_size,
                    'device': str(device),
                    'train_size': len(train_loader.dataset),
                    'val_size': len(val_loader.dataset),
                    'test_size': len(test_loader.dataset) if test_loader else 0
                }
            )
            wandb.watch(model, log='all', log_freq=100)
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            self.use_wandb = False
        
        # loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # training history
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
            target_seqs = batch['target_seqs'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # forward pass
            outputs, targets = self.model(images, target_seqs)
            
            # calculate loss
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.contiguous().view(-1, vocab_size)
            targets = targets.contiguous().view(-1)
            
            loss = self.criterion(outputs, targets)
            
            # backward pass
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
    
    def validate(self, data_loader, dataset_name="validation"):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct_chars = 0
        total_chars = 0
        num_batches = 0
        edit_distances = []
        predictions_list = []
        references_list = []
        inference_examples = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images = batch['images'].to(self.device)
                target_seqs = batch['target_seqs'].to(self.device)
                plaintexts = batch['plaintexts']
                
                # for loss calculation
                self.model.train()
                outputs, targets = self.model(images, target_seqs)
                self.model.eval()
                
                # calculate loss
                batch_size, seq_len, vocab_size = outputs.shape
                outputs_flat = outputs.contiguous().view(-1, vocab_size)
                targets_flat = targets.contiguous().view(-1)
                
                loss = self.criterion(outputs_flat, targets_flat)
                total_loss += loss.item()
                
                # calculate character accuracy
                predictions = torch.argmax(outputs, dim=-1)
                mask = targets != 0  # ignore padding
                correct_chars += ((predictions == targets) & mask).sum().item()
                total_chars += mask.sum().item()
                
                # inference forward pass for text generation
                generated_seqs = self.model(images, max_length=200)
                
                # process predictions for metrics
                for i in range(batch_size):
                    pred_indices = generated_seqs[i].cpu().numpy()
                    # access the dataset correctly
                    if hasattr(data_loader.dataset, 'indices_to_plaintext'):
                        pred_text = data_loader.dataset.indices_to_plaintext(pred_indices)
                    else:
                        # For subset datasets, get the original dataset
                        pred_text = data_loader.dataset.dataset.indices_to_plaintext(pred_indices)
                    
                    true_text = plaintexts[i]
                    
                    # clean texts
                    pred_text = pred_text.strip()
                    true_text = true_text.strip()
                    
                    predictions_list.append(pred_text)
                    references_list.append(true_text)
                    
                    # calculate edit distance
                    edit_dist = editdistance.eval(pred_text, true_text)
                    normalized_edit_dist = edit_dist / max(len(true_text), 1)
                    edit_distances.append(normalized_edit_dist)
                    
                    # collect inference examples
                    if len(inference_examples) < 5:
                        # Convert image tensor for logging
                        img_np = images[i].cpu().numpy().squeeze()
                        inference_examples.append({
                            'image': wandb.Image(img_np, caption=f"True: {true_text[:100]}..."),
                            'prediction': pred_text[:100],
                            'ground_truth': true_text[:100],
                            'edit_distance': normalized_edit_dist
                        })
                
                num_batches += 1
        
        # calculate metrics
        avg_loss = total_loss / num_batches
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        avg_edit_distance = np.mean(edit_distances) if edit_distances else 1.0
        
        # calculate WER and CER
        try:
            word_error_rate = wer(references_list, predictions_list)
            char_error_rate = cer(references_list, predictions_list)
        except:
            word_error_rate = avg_edit_distance
            char_error_rate = avg_edit_distance
            print(f"Warning: Could not calculate WER/CER with jiwer")
        
        # store validation metrics
        if dataset_name == "validation":
            self.val_losses.append(avg_loss)
            self.val_accuracies.append(char_accuracy)
        
        # log examples to wandb
        if self.use_wandb and len(inference_examples) > 0:
            wandb.log({
                f"{dataset_name}_inference_examples": [ex['image'] for ex in inference_examples],
                f"{dataset_name}_predictions_table": wandb.Table(
                    columns=["Ground Truth", "Prediction", "Edit Distance"],
                    data=[[ex['ground_truth'], ex['prediction'], f"{ex['edit_distance']:.3f}"] 
                          for ex in inference_examples]
                )
            })
        
        return avg_loss, char_accuracy, avg_edit_distance, word_error_rate, char_error_rate
    
    def train(self, num_epochs, save_path='decipher_model.pth'):
        """Complete training loop"""
        best_edit_distance = float('inf')
        best_metrics = {}
        
        print("Starting training...")
        for epoch in range(num_epochs):
            # train
            train_loss = self.train_epoch()
            
            # validate
            val_loss, val_accuracy, avg_edit_distance, val_wer, val_cer = self.validate(self.val_loader, "validation")
            
            # update scheduler
            self.scheduler.step(val_loss)
            
            # log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_char_accuracy': val_accuracy,
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
            print(f"  Val Char Accuracy: {val_accuracy:.4f}")
            print(f"  Val Edit Distance: {avg_edit_distance:.4f}")
            print(f"  Val WER: {val_wer:.4f}")
            print(f"  Val CER: {val_cer:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Track best model
            if avg_edit_distance < best_edit_distance:
                best_edit_distance = avg_edit_distance
                best_metrics = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_char_accuracy': val_accuracy,
                    'val_edit_distance': avg_edit_distance,
                    'val_wer': val_wer,
                    'val_cer': val_cer,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }
                print(f"  New best model found at epoch {epoch + 1} (edit distance={avg_edit_distance:.4f})!")
            
            print("-" * 50)
        
        # save best model
        if best_metrics:
            checkpoint = {
                'model_state_dict': best_metrics['model_state_dict'],
                'optimizer_state_dict': best_metrics['optimizer_state_dict'],
                'epoch': best_metrics['epoch'],
                'val_loss': best_metrics['val_loss'],
                'val_char_accuracy': best_metrics['val_char_accuracy'],
                'val_edit_distance': best_metrics['val_edit_distance'],
                'val_wer': best_metrics['val_wer'],
                'val_cer': best_metrics['val_cer']
            }
            torch.save(checkpoint, save_path)
            
            if self.use_wandb:
                wandb.save(save_path)
            
            print(f"Training complete! Best model saved from epoch {best_metrics['epoch']} "
                  f"with val_edit_distance {best_metrics['val_edit_distance']:.4f}")
        
        # test evaluation
        if self.test_loader:
            print("Running final test evaluation...")
            test_loss, test_accuracy, test_edit_distance, test_wer, test_cer = self.validate(self.test_loader, "test")
            
            print(f"Test Results:")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Test Char Accuracy: {test_accuracy:.4f}")
            print(f"  Test Edit Distance: {test_edit_distance:.4f}")
            print(f"  Test WER: {test_wer:.4f}")
            print(f"  Test CER: {test_cer:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'test_loss': test_loss,
                    'test_char_accuracy': test_accuracy,
                    'test_edit_distance': test_edit_distance,
                    'test_wer': test_wer,
                    'test_cer': test_cer
                })
        
        if self.use_wandb:
            wandb.finish()


def main():
    """main training function; set paths and project name in config below."""
    torch.cuda.empty_cache()
    gc.collect()
    
    # configuration: set paths and project name for your environment
    config = {
        'train_data_file': 'path/to/train.json',
        'val_data_file': 'path/to/val.json',
        'test_data_file': 'path/to/test.json',
        'train_image_dir': 'path/to/train_images',
        'val_image_dir': 'path/to/val_images',
        'test_image_dir': 'path/to/test_images',
        'pretrained_crnn_path': None,
        'vocab_output': 'plaintext_vocabulary.json',
        'model_save_path': 'decipher_model.pth',
        'batch_size': 16,
        'num_epochs': 100,
        'embed_size': 128,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'max_width': 800,
        'target_height': 64,
        'max_seq_len': 200,
        'project_name': 'image_decryption_training',
    }
    
    # create datasets
    print("Loading datasets...")
    
    # load training dataset first to build vocabulary
    train_dataset = CopialeImageDecryptionDataset(
        data_file=config['train_data_file'],
        image_dir=config['train_image_dir'],
        max_width=config['max_width'],
        target_height=config['target_height'],
        max_seq_len=config['max_seq_len']
    )
    
    # get vocabulary from training dataset
    vocab = train_dataset.get_vocabulary()
    
    # Export vocabulary
    with open('quijoteVocabulary.json', 'w') as f:
        json.dump(vocab, f, indent=2)
    print("Vocabulary exported to quijoteVocabulary.json")
    
    # load validation and test datasets with shared vocabulary
    val_dataset = CopialeImageDecryptionDataset(
        data_file=config['val_data_file'],
        image_dir=config['val_image_dir'],
        max_width=config['max_width'],
        target_height=config['target_height'],
        vocab_plaintext=vocab,
        max_seq_len=config['max_seq_len']
    )
    
    test_dataset = CopialeImageDecryptionDataset(
        data_file=config['test_data_file'],
        image_dir=config['test_image_dir'],
        max_width=config['max_width'],
        target_height=config['target_height'],
        vocab_plaintext=vocab,
        max_seq_len=config['max_seq_len']
    )
    
    # create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # create model with CRNN feature extractor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CopialeImageDecryptionModelWithCRNN(
        plaintext_vocab_size=train_dataset.plaintext_vocab['vocab_size'],
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        plaintext_vocab=train_dataset.plaintext_vocab,
        pretrained_crnn_path=config['pretrained_crnn_path']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # create trainer and train
    trainer = ImageDecryptionTrainer(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        project_name=config['project_name']
    )
    
    # train model
    trainer.train(config['num_epochs'], save_path=config['model_save_path'])
    
    print("Training completed!")

    # clean up memory
    del model, trainer, train_dataset, val_dataset, test_dataset
    del train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()


def decrypt_image(model, dataset, image_path, device='cuda', max_length=200):
    """
    Decrypt a single image using the trained model
    Args:
        model: Trained CopialeImageDecryptionModelWithCRNN
        dataset: Dataset instance (for vocabulary access)
        image_path: Path to input image
        device: Device to run inference on
        max_length: Maximum output length
    """
    model.eval()
    
    # load and preprocess image
    image = dataset.load_and_preprocess_image(Path(image_path))
    image_tensor = image.unsqueeze(0).to(device)  # add batch dimension
    
    with torch.no_grad():
        # generate output sequence
        output_indices = model(image_tensor, max_length=max_length)
        
        # convert back to text
        output_text = dataset.indices_to_plaintext(output_indices[0].cpu().numpy())
    
    return output_text


# Example usage for inference
def inference_example():
    """Example of how to use the trained model for inference"""
    
    # load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabulary
    with open('copialeDirectImageDecryptionVocabulary.json', 'r') as f:
        vocab = json.load(f)
    
    # create a dummy dataset instance for vocabulary access
    class InferenceDataset:
        def __init__(self, vocab, max_width=800, target_height=64):
            self.plaintext_vocab = vocab
            self.max_width = max_width
            self.target_height = target_height
        
        def load_and_preprocess_image(self, image_path):
            """Load and preprocess image with fixed dimensions"""
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
                # crop if too wide
                image = image[:, :self.max_width]
            elif new_width < self.max_width:
                # pad with white (255) if too narrow
                pad_width = self.max_width - new_width
                image = np.pad(image, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
            
            # normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # convert to tensor and add channel dimension
            image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
            
            return image
        
        def indices_to_plaintext(self, indices):
            chars = []
            for idx in indices:
                char = self.plaintext_vocab['idx_to_char'].get(str(idx), '<UNK>')
                if char in ['<SOS>', '<EOS>', '<PAD>']:
                    continue
                chars.append(char)
            return ''.join(chars)
    
    # create model
    model = CopialeImageDecryptionModelWithCRNN(
        plaintext_vocab_size=vocab['vocab_size'],
        embed_size=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        plaintext_vocab=vocab
    )
    
    # load trained weights
    checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # create inference dataset
    inference_dataset = InferenceDataset(vocab)
    
    # decipher
    try:
        decrypted_text = decrypt_image(model, inference_dataset, example_image_path, device)
        
        print(f"Input image: {example_image_path}")
        print(f"Decrypted text: {decrypted_text}")
    except Exception as e:
        print(f"Error during decryption: {e}")
        print("Make sure you have a trained model and the image path is correct")

    # clean up memory
    del model, inference_dataset
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
    
    # uncomment to run inference example
    # inference_example()