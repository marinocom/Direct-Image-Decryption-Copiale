# --------------------------
# TRANSCRIPTION INFERENCE SCRIPT using Baro CRNN
# Input: model.pth, vocabulary.json, dataset.json, image_dir
# --------------------------
#
# OUTPUTS
# -------
# Single-image mode (SINGLE_IMAGE set):
#   - Console: the predicted transcription string for that image.
#
# Full-dataset mode (SINGLE_IMAGE is None):
#   Writes to OUTPUT_DIR (default: crnn_inference_results):
#
#   - detailed_results.csv
#     Per-sample table: filename, predicted_transcription, predicted_normalized,
#     true_transcription, true_normalized, edit_distance, normalized_edit_distance.
#
#   - metrics.json
#     Aggregate metrics: token_accuracy, sequence_accuracy, normalized_edit_distance,
#     word_error_rate, character_error_rate, transcription_success_rate (all in [0,1]).
#
#   - predictions_comparison.csv
#     Columns: filename, predicted, reference, predicted_raw, reference_raw
#     (side-by-side predicted vs reference transcriptions).
#
#   - best_examples.txt
#     Top N samples by lowest normalized edit distance: filename, true/predicted
#     (raw and normalized) and edit distance.
#
#   - worst_examples.txt
#     Bottom N samples by highest normalized edit distance; same format as above.
#
#   - analysis_plots.png / analysis_plots.pdf
#     2x3 figure: metrics bar chart, edit-distance histogram, length correlation
#     scatter, success rate by length, best/worst example summary, token frequency
#     comparison (top 15 tokens).

# --------------------------


import torch
import torch.nn as nn
import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import editdistance
from jiwer import wer, cer
import os
from tqdm import tqdm
import pandas as pd
from typing import Dict, List, Tuple
import seaborn as sns
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


def normalize_tokens(tokens):
    """Map BigA-Z to a-z, leave others unchanged."""
    normalized = []
    for t in tokens:
        if t.startswith("Big") and len(t) == 4 and t[3].isalpha():
            normalized.append(t[3].lower())
        else:
            normalized.append(t)
    return normalized


# import model classes from training script
class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for CRNN - optimized for fixed-size images"""
    
    def __init__(self, input_channels=1, output_channels=256):
        super(CNNFeatureExtractor, self).__init__()
        
        self.cnn = nn.Sequential(
            # first conv block
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # H/2, W/2
            
            # second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # H/4, W/4
            
            # third conv block - pool height more aggressively
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # H/8, W/4
            
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
    """CRNN model with CTC loss for sequence recognition"""
    
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
        
        # extract CNN features
        features = self.feature_extractor(images)  # (batch_size, seq_len, feature_dim)
        
        # lstm processing
        lstm_out, _ = self.lstm(features)  # (batch_size, seq_len, hidden_size*2)
        
        # classification
        output = self.classifier(self.dropout(lstm_out))  # (batch_size, seq_len, vocab_size)
        
        # for CTC, need log_softmax
        output = torch.nn.functional.log_softmax(output, dim=2)
        
        return output


def ctc_decode(predictions, blank_idx=0):
    """simple CTC greedy decoder"""
    # predictions: (batch_size, seq_len, vocab_size)
    pred_indices = torch.argmax(predictions, dim=-1)  # (batch_size, seq_len)
    
    decoded = []
    for batch_idx in range(pred_indices.size(0)):
        pred = pred_indices[batch_idx].cpu().numpy()
        
        # ctc collapse: remove repeated tokens and blanks
        prev_token = None
        decoded_seq = []
        for token in pred:
            if token != blank_idx and token != prev_token:
                decoded_seq.append(token)
            prev_token = token
        
        decoded.append(decoded_seq)
    
    return decoded


class BoundedMetrics:
    """class to compute bounded metrics for transcription evaluation"""
    
    def __init__(self):
        self.metrics = {}
    
    def token_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """calculate token-level accuracy (0-1)"""
        total_tokens = 0
        correct_tokens = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            # align lengths by taking minimum
            min_len = min(len(pred_tokens), len(ref_tokens))
            total_tokens += max(len(pred_tokens), len(ref_tokens))
            
            # count correct tokens at aligned positions
            for i in range(min_len):
                if pred_tokens[i] == ref_tokens[i]:
                    correct_tokens += 1
        
        return correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def normalized_edit_distance(self, predictions: List[str], references: List[str]) -> float:
        """calculate normalized edit distance on tokens (0-1, lower is better)"""
        distances = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            edit_dist = editdistance.eval(pred_tokens, ref_tokens)
            max_len = max(len(ref_tokens), 1)
            normalized_dist = edit_dist / max_len
            distances.append(normalized_dist)
        
        return np.mean(distances)
    
    def sequence_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """calculate exact sequence match accuracy (0-1)"""
        exact_matches = 0
        for pred, ref in zip(predictions, references):
            if pred.strip() == ref.strip():
                exact_matches += 1
        
        return exact_matches / len(predictions) if predictions else 0.0
    
    def bounded_wer_cer(self, predictions: List[str], references: List[str]) -> Tuple[float, float]:
        """calculate bounded WER and CER (0-1, lower is better)"""
        try:
            # standard WER/CER calculation
            word_error_rate = wer(references, predictions)
            char_error_rate = cer(references, predictions)
            
            # bound to [0, 1]
            word_error_rate = min(1.0, max(0.0, word_error_rate))
            char_error_rate = min(1.0, max(0.0, char_error_rate))
            
        except Exception as e:
            print(f"Warning: jiwer failed ({e}), using edit distance approximation")
            # fallback to edit distance
            char_error_rate = self.normalized_edit_distance(predictions, references)
            word_error_rate = char_error_rate
        
        return word_error_rate, char_error_rate
    
    def compute_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """compute all bounded metrics"""
        token_acc = self.token_accuracy(predictions, references)
        seq_acc = self.sequence_accuracy(predictions, references)
        edit_dist = self.normalized_edit_distance(predictions, references)
        wer_score, cer_score = self.bounded_wer_cer(predictions, references)
        
        return {
            'token_accuracy': token_acc,
            'sequence_accuracy': seq_acc,
            'normalized_edit_distance': edit_dist,
            'word_error_rate': wer_score,
            'character_error_rate': cer_score,
            'transcription_success_rate': 1.0 - edit_dist,
        }


class CRNNTranscriptionInference:
    """main inference class for CRNN transcription"""
    
    def __init__(self, model_path: str, vocab_path: str, device: str = 'cuda',
                 max_width: int = 800, target_height: int = 64):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.vocab_path = vocab_path
        self.max_width = max_width
        self.target_height = target_height
        
        # load vocabulary
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        
        self.token_to_idx = vocab['token_to_idx']
        self.idx_to_token = vocab['idx_to_token']
        self.vocab_size = vocab['vocab_size']
        
        # initialize model
        self.model = CRNNModel(
            vocab_size=self.vocab_size,
            hidden_size=256,
            num_layers=4,
            dropout=0.4
        )
        
        # load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # handle DataParallel saved models
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # initialize metrics calculator
        self.metrics_calculator = BoundedMetrics()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def load_and_preprocess_image(self, image_path: str):
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
            # crop if too wide
            image = image[:, :self.max_width]
        elif new_width < self.max_width:
            # pad with white (255) if too narrow
            pad_width = self.max_width - new_width
            image = np.pad(image, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
        
        # normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # convert to tensor and add batch and channel dimensions
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        return image
    
    def indices_to_text(self, indices):
        """convert indices back to text"""
        tokens = []
        for idx in indices:
            # handle both string and integer keys
            token = self.idx_to_token.get(str(idx), self.idx_to_token.get(idx, '<UNK>'))
            if token in ['<BLANK>', '<PAD>']:
                continue
            tokens.append(token)
        return ' '.join(tokens)
    
    def transcribe_single_image(self, image_path: str) -> str:
        """transcribe a single image"""
        try:
            # load and preprocess image
            image_tensor = self.load_and_preprocess_image(image_path).to(self.device)
            
            with torch.no_grad():
                # forward pass
                log_probs = self.model(image_tensor)  # (1, seq_len, vocab_size)
                
                # decode predictions
                decoded_preds = ctc_decode(log_probs, blank_idx=0)
                
                # convert to text
                transcription = self.indices_to_text(decoded_preds[0])
            
            return transcription.strip()
            
        except Exception as e:
            print(f"Error transcribing image '{image_path}': {e}")
            return ""
    
    def evaluate_dataset(self, data_file: str, image_dir: str,
                        save_results: bool = True, 
                        output_dir: str = 'crnn_inference_results') -> Dict:
        """Evaluate model on a complete dataset"""
        
        # load dataset
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        predictions = []
        references = []
        results = []
        
        print(f"Evaluating on {len(data)} samples...")
        
        # process each image
        for filename, item in tqdm(data.items(), desc="Processing images"):
            true_transcription = item['transcription']
            image_path = os.path.join(image_dir, filename)
            
            # transcribe image
            predicted_transcription = self.transcribe_single_image(image_path)
            
            # normalize tokens for comparison
            pred_tokens = normalize_tokens(predicted_transcription.split())
            true_tokens = normalize_tokens(true_transcription.split())
            
            pred_text_normalized = ' '.join(pred_tokens)
            true_text_normalized = ' '.join(true_tokens)
            
            predictions.append(pred_text_normalized)
            references.append(true_text_normalized)
            
            # store individual result
            edit_dist = editdistance.eval(pred_tokens, true_tokens)
            
            results.append({
                'filename': filename,
                'predicted_transcription': predicted_transcription,
                'predicted_normalized': pred_text_normalized,
                'true_transcription': true_transcription,
                'true_normalized': true_text_normalized,
                'edit_distance': edit_dist,
                'normalized_edit_distance': edit_dist / max(len(true_tokens), 1)
            })
        
        # calculate metrics
        metrics = self.metrics_calculator.compute_all_metrics(predictions, references)
        
        # save results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            
            # save detailed results
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
            
            # save metrics
            with open(f"{output_dir}/metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # save predictions vs references
            comparison_df = pd.DataFrame({
                'filename': [r['filename'] for r in results],
                'predicted': predictions,
                'reference': references,
                'predicted_raw': [r['predicted_transcription'] for r in results],
                'reference_raw': [r['true_transcription'] for r in results]
            })
            comparison_df.to_csv(f"{output_dir}/predictions_comparison.csv", index=False)
            
            print(f"Results saved to {output_dir}/")
        
        return {
            'metrics': metrics,
            'detailed_results': results,
            'predictions': predictions,
            'references': references
        }
    
    def generate_analysis_plots(self, results: Dict, output_dir: str = 'crnn_inference_results'):
        """generate analysis plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        detailed_results = results['detailed_results']
        metrics = results['metrics']
        
        # create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CRNN Transcription Model Analysis', fontsize=16)
        
        # 1. metrics overview bar chart
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        axes[0, 0].bar(range(len(metric_names)), metric_values, color='skyblue')
        axes[0, 0].set_xticks(range(len(metric_names)))
        axes[0, 0].set_xticklabels([name.replace('_', '\n') for name in metric_names], 
                                   rotation=45, ha='right', fontsize=8)
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('All Metrics Overview')
        axes[0, 0].set_ylim(0, 1)
        
        # add value labels on bars
        for i, v in enumerate(metric_values):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. edit distance distribution
        edit_distances = [r['normalized_edit_distance'] for r in detailed_results]
        axes[0, 1].hist(edit_distances, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(edit_distances), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(edit_distances):.3f}')
        axes[0, 1].set_xlabel('Normalized Edit Distance')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Edit Distance Distribution')
        axes[0, 1].legend()
        
        # 3. transcription length analysis (predicted vs reference)
        pred_lengths = [len(p.split()) for p in results['predictions']]
        ref_lengths = [len(r.split()) for r in results['references']]
        
        axes[0, 2].scatter(ref_lengths, pred_lengths, alpha=0.6, s=30, color='green')
        axes[0, 2].plot([0, max(max(pred_lengths), max(ref_lengths))], 
                       [0, max(max(pred_lengths), max(ref_lengths))], 'r--', alpha=0.8)
        axes[0, 2].set_xlabel('Reference Length (tokens)')
        axes[0, 2].set_ylabel('Prediction Length (tokens)')
        axes[0, 2].set_title('Length Correlation')
        
        # calculate correlation
        correlation = np.corrcoef(ref_lengths, pred_lengths)[0, 1]
        axes[0, 2].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[0, 2].transAxes, 
                       bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # 4. success rate by transcription length
        length_bins = np.arange(0, max(ref_lengths) + 5, 5)
        bin_indices = np.digitize(ref_lengths, length_bins)
        
        success_by_length = []
        length_centers = []
        
        for i in range(1, len(length_bins)):
            mask = bin_indices == i
            if np.any(mask):
                success_rate = 1 - np.mean([edit_distances[j] for j in range(len(edit_distances)) if mask[j]])
                success_by_length.append(success_rate)
                length_centers.append((length_bins[i-1] + length_bins[i]) / 2)
        
        if success_by_length:
            axes[1, 0].plot(length_centers, success_by_length, 'o-', 
                          color='purple', linewidth=2, markersize=6)
            axes[1, 0].set_xlabel('Reference Length (tokens)')
            axes[1, 0].set_ylabel('Transcription Success Rate')
            axes[1, 0].set_title('Success Rate by Input Length')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. top/bottom examples
        sorted_results = sorted(detailed_results, key=lambda x: x['normalized_edit_distance'])
        
        best_examples = sorted_results[:3]
        worst_examples = sorted_results[-3:]
        
        axes[1, 1].text(0.1, 0.9, "BEST EXAMPLES:", transform=axes[1, 1].transAxes, 
                       fontweight='bold', fontsize=10, color='green')
        y_pos = 0.8
        for i, ex in enumerate(best_examples):
            axes[1, 1].text(0.1, y_pos - i*0.15, 
                          f"File: {ex['filename'][:25]}...\nEdit Dist: {ex['normalized_edit_distance']:.3f}", 
                          transform=axes[1, 1].transAxes, fontsize=8)
        
        axes[1, 1].text(0.1, 0.4, "WORST EXAMPLES:", transform=axes[1, 1].transAxes, 
                       fontweight='bold', fontsize=10, color='red')
        y_pos = 0.3
        for i, ex in enumerate(worst_examples):
            axes[1, 1].text(0.1, y_pos - i*0.08, 
                          f"File: {ex['filename'][:25]}...\nEdit Dist: {ex['normalized_edit_distance']:.3f}", 
                          transform=axes[1, 1].transAxes, fontsize=8)
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Best/Worst Examples')
        
        # 6. token frequency analysis

        all_pred_tokens = ' '.join(results['predictions']).split()
        all_ref_tokens = ' '.join(results['references']).split()
        
        pred_token_freq = Counter(all_pred_tokens)
        ref_token_freq = Counter(all_ref_tokens)
        
        # get top 15 most common tokens
        common_tokens = sorted(ref_token_freq.keys(), 
                              key=lambda x: ref_token_freq[x], reverse=True)[:15]
        
        pred_freqs = [pred_token_freq.get(token, 0) for token in common_tokens]
        ref_freqs = [ref_token_freq[token] for token in common_tokens]
        
        x_pos = np.arange(len(common_tokens))
        width = 0.35
        
        axes[1, 2].bar(x_pos - width/2, ref_freqs, width, label='Reference', 
                      alpha=0.8, color='skyblue')
        axes[1, 2].bar(x_pos + width/2, pred_freqs, width, label='Predicted', 
                      alpha=0.8, color='lightcoral')
        
        axes[1, 2].set_xlabel('Tokens')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Token Frequency Comparison (Top 15)')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(common_tokens, rotation=45, ha='right', fontsize=7)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/analysis_plots.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/analysis_plots.pdf", bbox_inches='tight')
        plt.show()
        
        print(f"Analysis plots saved to {output_dir}/")
    
    def print_detailed_metrics(self, metrics: Dict):
        """print detailed metrics with explanations"""
        print("\n" + "="*80)
        print("CRNN TRANSCRIPTION METRICS REPORT")
        print("="*80)
        
        print(f"\nACCURACY METRICS:")
        print(f"   Token Accuracy:              {metrics['token_accuracy']:.4f} (higher is better, max=1.0)")
        print(f"   Sequence Accuracy:           {metrics['sequence_accuracy']:.4f} (exact matches, max=1.0)")
        print(f"   Transcription Success Rate:  {metrics['transcription_success_rate']:.4f} (overall success, max=1.0)")
        
        print(f"\nERROR METRICS:")
        print(f"   Normalized Edit Distance:    {metrics['normalized_edit_distance']:.4f} (lower is better, min=0.0)")
        print(f"   Word Error Rate (WER):       {metrics['word_error_rate']:.4f} (lower is better, min=0.0)")
        print(f"   Character Error Rate (CER):  {metrics['character_error_rate']:.4f} (lower is better, min=0.0)")
        
        print(f"\nPERFORMANCE SUMMARY:")
        avg_score = np.mean([
            metrics['token_accuracy'],
            metrics['sequence_accuracy'], 
            metrics['transcription_success_rate'],
            1 - metrics['normalized_edit_distance'],
            1 - metrics['word_error_rate'],
            1 - metrics['character_error_rate']
        ])
        print(f"   Overall Performance:         {avg_score:.4f} (composite score, max=1.0)")
        print("="*80)
    
    def save_examples(self, results: Dict, output_dir: str, num_examples: int = 10):
        """Save detailed examples for manual inspection"""
        detailed_results = results['detailed_results']
        
        # sort by edit distance
        sorted_results = sorted(detailed_results, key=lambda x: x['normalized_edit_distance'])
        
        # best and worst examples
        best_examples = sorted_results[:num_examples]
        worst_examples = sorted_results[-num_examples:]
        
        # save to text files for easy reading
        with open(f"{output_dir}/best_examples.txt", 'w', encoding='utf-8') as f:
            f.write("BEST TRANSCRIPTION EXAMPLES\n")
            f.write("="*80 + "\n\n")
            
            for i, ex in enumerate(best_examples):
                f.write(f"EXAMPLE {i+1} (Edit Distance: {ex['normalized_edit_distance']:.4f})\n")
                f.write(f"File: {ex['filename']}\n")
                f.write(f"True Transcription: {ex['true_transcription']}\n")
                f.write(f"Predicted:          {ex['predicted_transcription']}\n")
                f.write(f"True (normalized):  {ex['true_normalized']}\n")
                f.write(f"Pred (normalized):  {ex['predicted_normalized']}\n")
                f.write("-" * 80 + "\n\n")
        
        with open(f"{output_dir}/worst_examples.txt", 'w', encoding='utf-8') as f:
            f.write("WORST TRANSCRIPTION EXAMPLES\n")
            f.write("="*80 + "\n\n")
            
            for i, ex in enumerate(worst_examples):
                f.write(f"EXAMPLE {i+1} (Edit Distance: {ex['normalized_edit_distance']:.4f})\n")
                f.write(f"File: {ex['filename']}\n")
                f.write(f"True Transcription: {ex['true_transcription']}\n")
                f.write(f"Predicted:          {ex['predicted_transcription']}\n")
                f.write(f"True (normalized):  {ex['true_normalized']}\n")
                f.write(f"Pred (normalized):  {ex['predicted_normalized']}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"Example files saved to {output_dir}/")


def main():
    """Main function for CRNN inference - modify paths here"""
    
    # configuration - modify these paths as needed
    MODEL_PATH = ''
    VOCAB_PATH = ''
    DATA_FILE = ''
    IMAGE_DIR = ''
    OUTPUT_DIR = ''
    DEVICE = 'cuda'
    MAX_WIDTH = 800
    TARGET_HEIGHT = 64
    
    # OPTIONAL: set to a specific image path for single transcription
    SINGLE_IMAGE = None
    
    print("Starting CRNN Transcription Inference...")
    print(f"Model: {MODEL_PATH}")
    print(f"Vocabulary: {VOCAB_PATH}")
    print(f"Device: {DEVICE}")
    
    # initialize inference
    inference = CRNNTranscriptionInference(
        MODEL_PATH, 
        VOCAB_PATH, 
        DEVICE,
        max_width=MAX_WIDTH,
        target_height=TARGET_HEIGHT
    )
    
    # SINGLE IMAGE MODE 
    if SINGLE_IMAGE:
        # single image transcription
        print(f"Transcribing single image: {SINGLE_IMAGE}")
        result = inference.transcribe_single_image(SINGLE_IMAGE)
        print(f"Transcription: {result}")
    else:
        # full dataset evaluation
        print(f"Evaluating on dataset: {DATA_FILE}")
        results = inference.evaluate_dataset(
            DATA_FILE,
            IMAGE_DIR,
            save_results=True, 
            output_dir=OUTPUT_DIR
        )
        
        # print metrics
        inference.print_detailed_metrics(results['metrics'])
        
        # generate plots
        inference.generate_analysis_plots(results, OUTPUT_DIR)
        inference.save_examples(results, OUTPUT_DIR)
        
        print(f"\nEvaluation complete! Check {OUTPUT_DIR}/ for detailed results.")
        print(f"Key files:")
        print(f"   - detailed_results.csv: Full results with edit distances")
        print(f"   - predictions_comparison.csv: Side-by-side comparison")
        print(f"   - best_examples.txt: Best transcription examples")
        print(f"   - worst_examples.txt: Worst transcription examples")
        print(f"   - analysis_plots.png: Comprehensive analysis visualizations")


if __name__ == "__main__":
    main()