"""
Decryption model creation and training.

This script defines and trains a sequence-to-sequence (encoder–decoder with additive
attention) model for cipher decipherment: 2 STEP DECRYPTION MODEL

it maps transcription token sequences
(input from a CRNN transcription model) to plaintext character sequences. It uses
the transcription model’s vocabulary for input and builds or reuses a plaintext
character vocabulary.

Components:
- CopialeDecipherDataset: loads JSON (filename → {transcription, plaintext}), converts
  CRNN transcription vocab to seq2seq (SOS/EOS), builds or reuses plaintext char vocab,
  filters long sequences.
- CopialeDecipherModel: bidirectional LSTM encoder, unidirectional LSTM decoder with
  additive (Bahdanau-style) attention, character-level output.
- DecipherTrainer: training loop with teacher forcing, validation (loss, accuracy, edit
  distance, WER/CER), best-model checkpointing, optional wandb logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import numpy as np
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


class CopialeDecipherDataset(Dataset):
    """Dataset class for Copiale cipher transcription to plaintext decipherment - Character Level
    Uses the vocabulary from the transcription model for input"""
    
    def __init__(self, data_file: str, vocab_transcription=None, vocab_plaintext=None, max_seq_len=200):
        """
        Args:
            data_file: JSON file containing transcriptions and plaintexts
            vocab_transcription: Vocabulary dict from transcription model (REQUIRED)
            vocab_plaintext: Optional vocabulary dict for plaintext characters
            max_seq_len: Maximum sequence length for truncation
        """
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.max_seq_len = max_seq_len
        
        # use provided transcription vocabulary (from CRNN model)
        if vocab_transcription is None:
            raise ValueError("vocab_transcription must be provided from the transcription model!")
        
        # convert transcription vocab from CRNN format to seq2seq format
        self.transcription_vocab = self.convert_crnn_vocab_to_seq2seq(vocab_transcription)
        
        # build plaintext vocabulary if not provided
        if vocab_plaintext is None:
            self.build_plaintext_vocabulary()
        else:
            self.plaintext_vocab = vocab_plaintext
        
        # filter out sequences that are too long
        self.filter_long_sequences()
    
    def convert_crnn_vocab_to_seq2seq(self, crnn_vocab):
        """Convert CRNN vocabulary format to seq2seq format with SOS/EOS tokens"""
        # crnn vocab has: token_to_idx, idx_to_token, vocab_size
        # we need to add SOS and EOS tokens for seq2seq
        
        token_to_idx = crnn_vocab['token_to_idx'].copy()
        
        # add SOS and EOS tokens if not present
        if '< SOS >' not in token_to_idx:
            token_to_idx['< SOS >'] = len(token_to_idx)
        if '<EOS>' not in token_to_idx:
            token_to_idx['<EOS>'] = len(token_to_idx)
        
        # create reverse mapping
        idx_to_token = {idx: token for token, idx in token_to_idx.items()}
        
        return {
            'token_to_idx': token_to_idx,
            'idx_to_token': idx_to_token,
            'vocab_size': len(token_to_idx)
        }
        
    def build_plaintext_vocabulary(self):
        """Build plaintext vocabulary - character level (matching pixelHacking structure)"""
        plaintext_chars = set()
        
        for filename, item in self.data.items():
            plaintext = item['plaintext']
            # add all characters from plaintext (including spaces)
            plaintext_chars.update(list(plaintext))
        
        # build plaintext vocabulary - matching pixelHacking structure exactly
        # special tokens first, then space, then uppercase, then lowercase, then special characters
        special_tokens = ['<PAD>', '< SOS >', '<EOS>', '<UNK>']
        space_char = [' ']
        uppercase = [chr(i) for i in range(ord('A'), ord('Z')+1) if chr(i) != 'Q']  # no Q in vocab
        lowercase = [chr(i) for i in range(ord('a'), ord('z')+1) if chr(i) != 'q']  # no q in vocab
        
        # add other characters found in the data that aren't covered above
        other_chars = []
        for char in plaintext_chars:
            if (char not in special_tokens and 
                char not in space_char and 
                char not in uppercase and 
                char not in lowercase):
                other_chars.append(char)
        
        # sort other characters for consistency
        other_chars = sorted(other_chars)
        
        # combine all in the order: special_tokens, space, uppercase, lowercase, others
        sorted_plaintext_chars = special_tokens + space_char + uppercase + lowercase + other_chars
        
        self.plaintext_vocab = {
            'char_to_idx': {char: idx for idx, char in enumerate(sorted_plaintext_chars)},
            'idx_to_char': {idx: char for idx, char in enumerate(sorted_plaintext_chars)},
            'vocab_size': len(sorted_plaintext_chars)
        }
        
        print(f"Transcription vocabulary size: {self.transcription_vocab['vocab_size']}")
        print(f"Plaintext vocabulary size: {self.plaintext_vocab['vocab_size']}")
        print(f"Sample transcription tokens: {list(self.transcription_vocab['token_to_idx'].keys())[:20]}")
        print(f"Sample plaintext chars: {sorted_plaintext_chars[:20]}")
    
    def filter_long_sequences(self):
        """Filter out sequences that exceed max_seq_len"""
        original_count = len(self.data)
        filtered_data = {}
        
        for filename, item in self.data.items():
            transcription_tokens = item['transcription'].split()
            plaintext_chars = list(item['plaintext'])
            
            # keep sequences within length limits (accounting for SOS/EOS tokens)
            if len(transcription_tokens) <= self.max_seq_len - 2 and len(plaintext_chars) <= self.max_seq_len - 2:
                filtered_data[filename] = item
        
        self.data = filtered_data
        print(f"Filtered dataset: {original_count} -> {len(self.data)} samples (removed {original_count - len(self.data)} long sequences)")
    
    def get_vocabularies(self):
        """Return vocabularies for sharing with other datasets"""
        return {
            'transcription': self.transcription_vocab,
            'plaintext': self.plaintext_vocab
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # get the filename and item data
        filename = list(self.data.keys())[idx]
        item = self.data[filename]
        
        transcription = item['transcription']
        plaintext = item['plaintext']
        
        # convert transcription to indices (input sequence)
        input_indices = self.transcription_to_indices(transcription)
        
        # convert plaintext to indices (target sequence)
        target_indices = self.plaintext_to_indices(plaintext)
        
        return {
            'input_seq': torch.tensor(input_indices, dtype=torch.long),
            'target_seq': torch.tensor(target_indices, dtype=torch.long),
            'input_length': len(input_indices),
            'target_length': len(target_indices),
            'transcription': transcription,
            'plaintext': plaintext,
            'filename': filename
        }
    
    def transcription_to_indices(self, transcription):
        """Convert transcription tokens to indices"""
        indices = [self.transcription_vocab['token_to_idx']['< SOS >']]
        tokens = transcription.split()
        for token in tokens:
            indices.append(self.transcription_vocab['token_to_idx'].get(token, self.transcription_vocab['token_to_idx']['<UNK>']))
        indices.append(self.transcription_vocab['token_to_idx']['<EOS>'])
        return indices
    
    def plaintext_to_indices(self, plaintext):
        """Convert plaintext characters to indices - character level"""
        indices = [self.plaintext_vocab['char_to_idx']['< SOS >']]
        for char in plaintext:
            indices.append(self.plaintext_vocab['char_to_idx'].get(char, self.plaintext_vocab['char_to_idx']['<UNK>']))
        indices.append(self.plaintext_vocab['char_to_idx']['<EOS>'])
        return indices
    
    def indices_to_transcription(self, indices):
        """Convert indices back to transcription tokens"""
        tokens = []
        for idx in indices:
            token = self.transcription_vocab['idx_to_token'].get(idx, '<UNK>')
            if token in ['< SOS >', '<EOS>', '<PAD>']:
                continue
            tokens.append(token)
        return ' '.join(tokens)
    
    def indices_to_plaintext(self, indices):
        """Convert indices back to plaintext - character level"""
        chars = []
        for idx in indices:
            char = self.plaintext_vocab['idx_to_char'].get(idx, '<UNK>')
            if char in ['< SOS >', '<EOS>', '<PAD>']:
                continue
            chars.append(char)
        return ''.join(chars)


class CopialeDecipherModel(nn.Module):
    """Sequence-to-sequence model for Copiale cipher decipherment"""
    
    def __init__(self, transcription_vocab_size, plaintext_vocab_size, embed_size=128, 
                 hidden_size=256, num_layers=2, dropout=0.3, plaintext_vocab=None):
        super(CopialeDecipherModel, self).__init__()
        
        self.transcription_vocab_size = transcription_vocab_size
        self.plaintext_vocab_size = plaintext_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # encoder - processes transcription tokens
        self.encoder_embedding = nn.Embedding(transcription_vocab_size, embed_size, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                                    batch_first=True, dropout=dropout if num_layers > 1 else 0,
                                    bidirectional=True)
        
        # decoder - generates plaintext characters
        self.decoder_embedding = nn.Embedding(plaintext_vocab_size, embed_size, padding_idx=0)
        self.decoder_lstm = nn.LSTM(embed_size + hidden_size*2, hidden_size, num_layers,
                                    batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # attention mechanism
        self.attention = nn.Linear(hidden_size * 3, 1)
        
        # output layer
        self.fc_out = nn.Linear(hidden_size, plaintext_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # store vocab for SOS token
        self.plaintext_vocab = plaintext_vocab
        if plaintext_vocab:
            self.sos_idx = plaintext_vocab['char_to_idx']['< SOS >']
            self.eos_idx = plaintext_vocab['char_to_idx']['<EOS>']
    
    def forward(self, input_seq, target_seq=None, teacher_forcing_ratio=0.5, max_length=200):
        """
        Forward pass with optional teacher forcing
        Args:
            input_seq: (batch_size, input_seq_len) - transcription token indices
            target_seq: (batch_size, target_seq_len) - plaintext char indices (for training)
            teacher_forcing_ratio: probability of using teacher forcing
            max_length: maximum output length for inference
        """
        batch_size = input_seq.size(0)
        
        # encode input sequence
        encoder_outputs, (hidden, cell) = self.encode(input_seq)
        
        # prepare decoder input
        if target_seq is not None:
            # training mode
            target_len = target_seq.size(1)
            outputs = torch.zeros(batch_size, target_len, self.plaintext_vocab_size).to(input_seq.device)
            
            # first decoder input is SOS token
            decoder_input = target_seq[:, 0].unsqueeze(1)  # (batch_size, 1)
            
            # adjust hidden state dimensions for decoder
            hidden = self._adjust_hidden_state(hidden)
            cell = self._adjust_hidden_state(cell)
            
            for t in range(1, target_len):
                output, hidden, cell = self.decode_step(decoder_input, hidden, cell, encoder_outputs)
                outputs[:, t, :] = output.squeeze(1)
                
                # teacher forcing
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    decoder_input = target_seq[:, t].unsqueeze(1)
                else:
                    decoder_input = output.argmax(2)
            
            return outputs
        else:
            # inference mode
            decoder_input = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long).to(input_seq.device)
            
            # adjust hidden state dimensions for decoder
            hidden = self._adjust_hidden_state(hidden)
            cell = self._adjust_hidden_state(cell)
            
            outputs = []
            for _ in range(max_length):
                output, hidden, cell = self.decode_step(decoder_input, hidden, cell, encoder_outputs)
                predicted = output.argmax(2)
                outputs.append(predicted)
                
                # stop if all sequences have generated EOS
                if (predicted == self.eos_idx).all():
                    break
                
                decoder_input = predicted
            
            return torch.cat(outputs, dim=1)
    
    def encode(self, input_seq):
        """Encode input sequence"""
        embedded = self.dropout(self.encoder_embedding(input_seq))
        encoder_outputs, (hidden, cell) = self.encoder_lstm(embedded)
        return encoder_outputs, (hidden, cell)
    
    def _adjust_hidden_state(self, hidden):
        """Adjust bidirectional encoder hidden state for unidirectional decoder"""
        # hidden: (num_layers * 2, batch_size, hidden_size)
        # we need: (num_layers, batch_size, hidden_size)
        # combine forward and backward hidden states
        hidden = hidden.view(self.num_layers, 2, hidden.size(1), self.hidden_size)
        hidden = hidden.sum(dim=1)  # sum forward and backward
        return hidden
    
    def decode_step(self, decoder_input, hidden, cell, encoder_outputs):
        """Single decoder step with attention"""
        # embed decoder input
        embedded = self.dropout(self.decoder_embedding(decoder_input))
        
        # calculate attention
        # hidden[-1]: (batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size*2)
        hidden_last = hidden[-1].unsqueeze(1)  # (batch_size, 1, hidden_size)
        hidden_expanded = hidden_last.expand(-1, encoder_outputs.size(1), -1)
        
        # concatenate and calculate attention scores
        attention_input = torch.cat([hidden_expanded, encoder_outputs], dim=2)
        attention_scores = self.attention(attention_input)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # apply attention to encoder outputs
        context = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs)  # (batch_size, 1, hidden_size*2)
        
        # concatenate embedded input with context
        decoder_input_combined = torch.cat([embedded, context], dim=2)
        
        # decoder LSTM step
        output, (hidden, cell) = self.decoder_lstm(decoder_input_combined, (hidden, cell))
        
        # output layer
        output = self.fc_out(output)
        
        return output, hidden, cell


def collate_fn(batch):
    """Custom collate function for batching variable-length sequences"""
    input_seqs = [item['input_seq'] for item in batch]
    target_seqs = [item['target_seq'] for item in batch]
    
    # pad sequences
    input_padded = pad_sequence(input_seqs, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target_seqs, batch_first=True, padding_value=0)
    
    return {
        'input_seq': input_padded,
        'target_seq': target_padded,
        'input_lengths': torch.tensor([item['input_length'] for item in batch]),
        'target_lengths': torch.tensor([item['target_length'] for item in batch]),
        'transcriptions': [item['transcription'] for item in batch],
        'plaintexts': [item['plaintext'] for item in batch],
        'filenames': [item['filename'] for item in batch]
    }


class DecipherTrainer:
    """Trainer class for the decipher model"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device, project_name='copiale_decipher'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        self.use_wandb = True
        self.project_name = project_name
    
    def train_epoch(self, teacher_forcing_ratio=0.5):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            input_seq = batch['input_seq'].to(self.device)
            target_seq = batch['target_seq'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # forward pass
            outputs = self.model(input_seq, target_seq, teacher_forcing_ratio=teacher_forcing_ratio)
            
            # calculate loss (excluding SOS token)
            outputs = outputs[:, 1:, :].contiguous()
            target_seq = target_seq[:, 1:].contiguous()
            
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), target_seq.view(-1))
            
            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_chars = 0
        all_edit_distances = []
        all_ground_truths = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_seq = batch['input_seq'].to(self.device)
                target_seq = batch['target_seq'].to(self.device)
                
                # forward pass with teacher forcing for loss calculation
                outputs = self.model(input_seq, target_seq, teacher_forcing_ratio=1.0)
                
                # calculate loss
                outputs_loss = outputs[:, 1:, :].contiguous()
                target_loss = target_seq[:, 1:].contiguous()
                loss = self.criterion(outputs_loss.view(-1, outputs_loss.size(-1)), target_loss.view(-1))
                total_loss += loss.item()
                
                # calculate accuracy
                predictions = outputs.argmax(dim=-1)
                mask = target_seq != 0  # ignore padding
                correct = ((predictions == target_seq) & mask).sum().item()
                total_correct += correct
                total_chars += mask.sum().item()
                
                # calculate edit distance for inference (without teacher forcing)
                inference_outputs = self.model(input_seq, target_seq=None, max_length=target_seq.size(1))
                
                for i in range(input_seq.size(0)):
                    gt = batch['plaintexts'][i]
                    pred_indices = inference_outputs[i].cpu().numpy()
                    pred = self.train_loader.dataset.indices_to_plaintext(pred_indices)
                    
                    # calculate normalized edit distance
                    ed = editdistance.eval(gt, pred)
                    normalized_ed = ed / max(len(gt), len(pred)) if max(len(gt), len(pred)) > 0 else 0
                    all_edit_distances.append(normalized_ed)
                    
                    all_ground_truths.append(gt)
                    all_predictions.append(pred)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_chars if total_chars > 0 else 0
        avg_edit_distance = np.mean(all_edit_distances) if all_edit_distances else 1.0
        
        # calculate WER and CER
        val_wer = wer(all_ground_truths, all_predictions) if all_predictions else 1.0
        val_cer = cer(all_ground_truths, all_predictions) if all_predictions else 1.0
        
        return avg_loss, accuracy, avg_edit_distance, val_wer, val_cer
    
    def test(self):
        """Test the model"""
        self.model.eval()
        all_edit_distances = []
        all_ground_truths = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_seq = batch['input_seq'].to(self.device)
                target_seq = batch['target_seq'].to(self.device)
                
                # inference
                outputs = self.model(input_seq, target_seq=None, max_length=target_seq.size(1))
                
                for i in range(input_seq.size(0)):
                    gt = batch['plaintexts'][i]
                    pred_indices = outputs[i].cpu().numpy()
                    pred = self.test_loader.dataset.indices_to_plaintext(pred_indices)
                    
                    ed = editdistance.eval(gt, pred)
                    normalized_ed = ed / max(len(gt), len(pred)) if max(len(gt), len(pred)) > 0 else 0
                    all_edit_distances.append(normalized_ed)
                    
                    all_ground_truths.append(gt)
                    all_predictions.append(pred)
        
        avg_edit_distance = np.mean(all_edit_distances) if all_edit_distances else 1.0
        test_wer = wer(all_ground_truths, all_predictions) if all_predictions else 1.0
        test_cer = cer(all_ground_truths, all_predictions) if all_predictions else 1.0
        
        print("\n" + "="*50)
        print("TEST RESULTS:")
        print(f"  Test Edit Distance: {avg_edit_distance:.4f}")
        print(f"  Test WER: {test_wer:.4f}")
        print(f"  Test CER: {test_cer:.4f}")
        print("="*50 + "\n")
        
        # log test results
        if self.use_wandb:
            wandb.log({
                'test_edit_distance': avg_edit_distance,
                'test_wer': test_wer,
                'test_cer': test_cer
            })
        
        return {
            'test_edit_distance': avg_edit_distance,
            'test_wer': test_wer,
            'test_cer': test_cer
        }
    
    def train(self, num_epochs, save_path='copiale_integrated_decipher_model_CRNN_Oct25.pth'):
        """Train the model"""
        best_edit_distance = float('inf')
        best_metrics = None
        
        for epoch in range(num_epochs):
            # anneal teacher forcing ratio
            teacher_forcing_ratio = max(0.5 - epoch * 0.01, 0.0)
            
            train_loss = self.train_epoch(teacher_forcing_ratio)
            val_loss, val_accuracy, avg_edit_distance, val_wer, val_cer = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # update learning rate
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
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'teacher_forcing_ratio': teacher_forcing_ratio
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
            print(f"  Teacher Forcing Ratio: {teacher_forcing_ratio:.2f}")
            
            # track best model
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
            
            print(f"Training complete! Best model saved from epoch {best_metrics['epoch']} with val_edit_distance {best_metrics['val_edit_distance']:.4f}")
        
        # run final test evaluation
        test_results = self.test()
        
        if self.use_wandb:
            wandb.finish()
        
        return test_results


def main():
    """main training function; set paths and project name in config below."""
    
    torch.cuda.empty_cache()
    gc.collect()

    # configuration: set paths and project name for your environment
    config = {
        'train_data_file': 'path/to/train.json',
        'val_data_file': 'path/to/val.json',
        'test_data_file': 'path/to/test.json',
        'transcription_vocab_file': 'path/to/transcription_vocabulary.json',
        'plaintext_vocab_output': 'plaintext_vocabulary.json',
        'full_vocab_output': 'full_vocabularies.json',
        'model_save_path': 'decipher_model.pth',
        'batch_size': 16,
        'num_epochs': 15,
        'embed_size': 128,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.3,
        'max_seq_len': 200,
        'wandb_project': 'decipher_training',
        'wandb_run_name': 'decipher-crnn-vocab',
    }

    wandb.init(project=config['wandb_project'], name=config['wandb_run_name'])
    
    print("="*60)
    print("LOADING TRANSCRIPTION MODEL VOCABULARY")
    print("="*60)
    
    # load vocabulary from transcription model
    with open(config['transcription_vocab_file'], 'r') as f:
        transcription_vocab = json.load(f)
    
    print(f"Loaded transcription vocabulary with {transcription_vocab['vocab_size']} tokens")
    print(f"Sample tokens: {list(transcription_vocab['token_to_idx'].keys())[:10]}")
    
    print("\n" + "="*60)
    print("CREATING DECRYPTION DATASETS")
    print("="*60)
    
    # create training dataset with transcription vocabulary
    train_dataset = CopialeDecipherDataset(
        data_file=config['train_data_file'],
        vocab_transcription=transcription_vocab,
        vocab_plaintext=None,  # will be built from data
        max_seq_len=config['max_seq_len']
    )
    
    # get vocabularies from training dataset
    vocabs = train_dataset.get_vocabularies()
    
    # export vocabularies
    decryption_vocab = {
        'char_to_idx': vocabs['plaintext']['char_to_idx'],
        'idx_to_char': {str(k): v for k, v in vocabs['plaintext']['idx_to_char'].items()},
        'vocab_size': vocabs['plaintext']['vocab_size']
    }
    
    with open(config['plaintext_vocab_output'], 'w') as f:
        json.dump(decryption_vocab, f, indent=2)
    print(f"\nPlaintext vocabulary exported to {config['plaintext_vocab_output']}")
    
    # also export full vocabularies
    with open(config['full_vocab_output'], 'w') as f:
        # convert idx_to_token keys to strings for JSON serialization
        vocabs_to_save = {
            'transcription': {
                'token_to_idx': vocabs['transcription']['token_to_idx'],
                'idx_to_token': {str(k): v for k, v in vocabs['transcription']['idx_to_token'].items()},
                'vocab_size': vocabs['transcription']['vocab_size']
            },
            'plaintext': {
                'char_to_idx': vocabs['plaintext']['char_to_idx'],
                'idx_to_char': {str(k): v for k, v in vocabs['plaintext']['idx_to_char'].items()},
                'vocab_size': vocabs['plaintext']['vocab_size']
            }
        }
        json.dump(vocabs_to_save, f, indent=2)
    print(f"Full vocabularies exported to {config['full_vocab_output']}")
    
    # load validation and test datasets with shared vocabularies
    val_dataset = CopialeDecipherDataset(
        data_file=config['val_data_file'],
        vocab_transcription=vocabs['transcription'],
        vocab_plaintext=vocabs['plaintext'],
        max_seq_len=config['max_seq_len']
    )
    
    test_dataset = CopialeDecipherDataset(
        data_file=config['test_data_file'],
        vocab_transcription=vocabs['transcription'],
        vocab_plaintext=vocabs['plaintext'],
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
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    print("\n" + "="*60)
    print("CREATING DECRYPTION MODEL")
    print("="*60)
    
    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CopialeDecipherModel(
        transcription_vocab_size=train_dataset.transcription_vocab['vocab_size'],
        plaintext_vocab_size=train_dataset.plaintext_vocab['vocab_size'],
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        plaintext_vocab=train_dataset.plaintext_vocab
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Transcription vocab size (input): {train_dataset.transcription_vocab['vocab_size']}")
    print(f"Plaintext vocab size (output): {train_dataset.plaintext_vocab['vocab_size']}")
    
    print("\n" + "="*60)
    print("TRAINING DECRYPTION MODEL")
    print("="*60)
    
    # create trainer and train
    trainer = DecipherTrainer(
        model, 
        train_loader, 
        val_loader,
        test_loader,
        device, 
        project_name=config['project_name']
    )
    
    # train model
    test_results = trainer.train(config['num_epochs'], save_path=config['model_save_path'])
    
    print("\nTraining completed!")
    if test_results:
        print("Final test results:")
        for key, value in test_results.items():
            print(f"  {key}: {value:.4f}")

    # clean up memory
    del model, trainer, train_dataset, val_dataset, test_dataset
    del train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()


def decipher_text(model, dataset, transcription_text, device='cuda', max_length=200):
    """
    Decipher a single transcription text using the trained model
    Args:
        model: Trained CopialeDecipherModel
        dataset: Dataset instance (for vocabulary access)
        transcription_text: Input transcription string
        device: Device to run inference on
        max_length: Maximum output length
    """
    model.eval()
    
    # convert transcription to indices
    input_indices = dataset.transcription_to_indices(transcription_text)
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
    
    with torch.no_grad():
        # generate output sequence
        output_indices = model(input_tensor, max_length=max_length)
        
        # convert back to text
        output_text = dataset.indices_to_plaintext(output_indices[0].cpu().numpy())
    
    return output_text


def inference_example():
    """example of how to use the trained model for inference; set paths below."""
    
    # set paths for your environment
    full_vocab_path = 'path/to/full_vocabularies.json'
    model_checkpoint_path = 'path/to/decipher_model.pth'

    # load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load vocabularies
    with open(full_vocab_path, 'r') as f:
        vocabs_raw = json.load(f)
    
    # convert string keys back to integers for idx_to_token/idx_to_char
    vocabs = {
        'transcription': {
            'token_to_idx': vocabs_raw['transcription']['token_to_idx'],
            'idx_to_token': {int(k): v for k, v in vocabs_raw['transcription']['idx_to_token'].items()},
            'vocab_size': vocabs_raw['transcription']['vocab_size']
        },
        'plaintext': {
            'char_to_idx': vocabs_raw['plaintext']['char_to_idx'],
            'idx_to_char': {int(k): v for k, v in vocabs_raw['plaintext']['idx_to_char'].items()},
            'vocab_size': vocabs_raw['plaintext']['vocab_size']
        }
    }
    
    # create a dummy dataset instance for vocabulary access
    class InferenceDataset:
        def __init__(self, vocabs):
            self.transcription_vocab = vocabs['transcription']
            self.plaintext_vocab = vocabs['plaintext']
        
        def transcription_to_indices(self, transcription):
            indices = [self.transcription_vocab['token_to_idx']['< SOS >']]
            tokens = transcription.split()
            for token in tokens:
                indices.append(self.transcription_vocab['token_to_idx'].get(token, 
                             self.transcription_vocab['token_to_idx']['<UNK>']))
            indices.append(self.transcription_vocab['token_to_idx']['<EOS>'])
            return indices
        
        def indices_to_plaintext(self, indices):
            chars = []
            for idx in indices:
                char = self.plaintext_vocab['idx_to_char'].get(idx, '<UNK>')
                if char in ['< SOS >', '<EOS>', '<PAD>']:
                    continue
                chars.append(char)
            return ''.join(chars)
    
    # create model
    model = CopialeDecipherModel(
        transcription_vocab_size=vocabs['transcription']['vocab_size'],
        plaintext_vocab_size=vocabs['plaintext']['vocab_size'],
        embed_size=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        plaintext_vocab=vocabs['plaintext']
    )
    
    # load trained weights
    checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # create inference dataset
    inference_dataset = InferenceDataset(vocabs)
    
    # example transcription to decipher (replace with actual transcription from CRNN output)
    example_transcription = "m^. a^^ c^. NorthEastArrow a^^ SquaredPlus BigT h^. SquareP a^^ j r + o^^ RockSalt 3"
    
    # decipher
    try:
        deciphered_text = decipher_text(model, inference_dataset, example_transcription, device)
        
        print(f"Input transcription: {example_transcription}")
        print(f"Deciphered text: {deciphered_text}")
    except Exception as e:
        print(f"Error during decryption: {e}")
        print("Make sure you have a trained model and proper vocabulary files")

    # clean up memory
    del model, inference_dataset
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
    
    # uncomment to run inference example
    # inference_example()