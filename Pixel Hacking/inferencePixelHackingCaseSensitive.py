"""
PIXEL HACKING inference (case-sensitive).

Runs the trained PIXEL HACKING (8-head attention) model. 
Supports single-image decryption or batch evaluation on a JSON
test set (filename → plaintext). All metrics are computed case-sensitively.
Set paths in main() before running.

All evaluation metrics (character/sequence accuracy, edit distance, BLEU, WER/CER,
decryption success rate) are computed in a case-sensitive way: predictions and
references are compared as-is (no lowercasing), so e.g. "CopiAle" and "copiale"
count as different.

Outputs (when save_results=True): CSV results, metrics summary, analysis plots,
and best/worst examples.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = ""


from pixelHackingmodelCreationOCT25 import (
    CopialeImageDecryptionModelWithCRNN, 
    CopialeImageDecryptionDataset,
    collate_fn
)


class InferenceDataset:
    """Simplified dataset class for inference"""
    
    def __init__(self, vocab_path: str, max_width=800, target_height=64):
        with open(vocab_path, 'r') as f:
            self.plaintext_vocab = json.load(f)
        self.max_width = max_width
        self.target_height = target_height
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image with fixed dimensions"""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # resize to fixed height while maintaining aspect ratio
        original_height, original_width = image.shape
        aspect_ratio = original_width / original_height
        new_width = int(self.target_height * aspect_ratio)
        
        # resize image
        image = cv2.resize(image, (new_width, self.target_height))
        
        # pad or crop to max_width
        if new_width > self.max_width:
            image = image[:, :self.max_width]
        elif new_width < self.max_width:
            pad_width = self.max_width - new_width
            image = np.pad(image, ((0, 0), (0, pad_width)), mode='constant', constant_values=255)
        
        # normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # convert to tensor and add channel dimension
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        
        return image
    
    def indices_to_plaintext(self, indices):
        """Convert indices back to plaintext"""
        chars = []
        skip_tokens = {'<SOS>', '< SOS >', '<EOS>', '<PAD>'}
        for idx in indices:
            char = self.plaintext_vocab['idx_to_char'].get(str(idx), '<UNK>')
            if char in skip_tokens:
                continue
            chars.append(char)
        return ''.join(chars)


class BoundedMetrics:
    """Class to compute bounded metrics for decryption evaluation with case-sensitive comparison"""
    
    def __init__(self):
        self.metrics = {}
    
    def character_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Calculate character-level accuracy (0-1) - case sensitive"""
        total_chars = 0
        correct_chars = 0
        
        for pred, ref in zip(predictions, references):
            pred_text = pred.strip()
            ref_text = ref.strip()
            
            min_len = min(len(pred_text), len(ref_text))
            total_chars += max(len(pred_text), len(ref_text))
            
            for i in range(min_len):
                if pred_text[i] == ref_text[i]:
                    correct_chars += 1
        
        return correct_chars / total_chars if total_chars > 0 else 0.0
    
    def normalized_edit_distance(self, predictions: List[str], references: List[str]) -> float:
        """Calculate normalized edit distance (0-1, lower is better) - case sensitive"""
        distances = []
        for pred, ref in zip(predictions, references):
            pred_text = pred.strip()
            ref_text = ref.strip()
            
            edit_dist = editdistance.eval(pred_text, ref_text)
            max_len = max(len(pred_text), len(ref_text), 1)
            normalized_dist = edit_dist / max_len
            distances.append(normalized_dist)
        
        return np.mean(distances)
    
    def sequence_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Calculate exact sequence match accuracy (0-1) - case sensitive"""
        exact_matches = 0
        for pred, ref in zip(predictions, references):
            pred_text = pred.strip()
            ref_text = ref.strip()
            if pred_text == ref_text:
                exact_matches += 1
        
        return exact_matches / len(predictions) if predictions else 0.0
    
    def bounded_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate character-level BLEU score approximation (0-1) - case sensitive"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smoothie = SmoothingFunction().method4
            
            scores = []
            for pred, ref in zip(predictions, references):
                pred_text = pred.strip()
                ref_text = ref.strip()
                pred_chars = list(pred_text)
                ref_chars = [list(ref_text)]
                
                if len(pred_chars) == 0:
                    scores.append(0.0)
                else:
                    score = sentence_bleu(ref_chars, pred_chars, smoothing_function=smoothie)
                    scores.append(score)
            
            return np.mean(scores)
        except ImportError:
            return self._simple_ngram_overlap(predictions, references)
    
    def _simple_ngram_overlap(self, predictions: List[str], references: List[str], n=4) -> float:
        """Simple n-gram overlap as BLEU approximation - case sensitive"""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_text = pred.strip()
            ref_text = ref.strip()
            
            if len(pred_text) == 0:
                scores.append(0.0)
                continue
            
            overlap_scores = []
            for gram_size in range(1, min(n + 1, len(pred_text) + 1)):
                pred_grams = set([pred_text[i:i+gram_size] for i in range(len(pred_text) - gram_size + 1)])
                ref_grams = set([ref_text[i:i+gram_size] for i in range(len(ref_text) - gram_size + 1)])
                
                if len(pred_grams) == 0:
                    overlap_scores.append(0.0)
                else:
                    overlap = len(pred_grams.intersection(ref_grams)) / len(pred_grams)
                    overlap_scores.append(overlap)
            
            scores.append(np.mean(overlap_scores) if overlap_scores else 0.0)
        
        return np.mean(scores)
    
    def bounded_wer_cer(self, predictions: List[str], references: List[str]) -> Tuple[float, float]:
        """Calculate bounded WER and CER (0-1, lower is better) - case sensitive"""
        try:
            pred_text = [p.strip() for p in predictions]
            ref_text = [r.strip() for r in references]
            
            word_error_rate = wer(ref_text, pred_text)
            char_error_rate = cer(ref_text, pred_text)
            
            word_error_rate = min(1.0, max(0.0, word_error_rate))
            char_error_rate = min(1.0, max(0.0, char_error_rate))
            
        except Exception as e:
            print(f"Warning: jiwer failed ({e}), using edit distance approximation")
            char_error_rate = self.normalized_edit_distance(predictions, references)
            word_error_rate = self.normalized_edit_distance(
                [' '.join(pred.split()) for pred in predictions],
                [' '.join(ref.split()) for ref in references]
            )
        
        return word_error_rate, char_error_rate
    
    def compute_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute all bounded metrics - case sensitive"""
        char_acc = self.character_accuracy(predictions, references)
        seq_acc = self.sequence_accuracy(predictions, references)
        edit_dist = self.normalized_edit_distance(predictions, references)
        bleu = self.bounded_bleu_score(predictions, references)
        wer_score, cer_score = self.bounded_wer_cer(predictions, references)
        
        return {
            'character_accuracy': char_acc,
            'sequence_accuracy': seq_acc,
            'normalized_edit_distance': edit_dist,
            'bleu_score': bleu,
            'word_error_rate': wer_score,
            'character_error_rate': cer_score,
            'decryption_success_rate': 1.0 - edit_dist,
        }


class DecryptionInference:
    """Main inference class for image decryption"""
    
    def __init__(self, model_path: str, vocab_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.vocab_path = vocab_path
        
        # load vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        # initialize model
        self.model = CopialeImageDecryptionModelWithCRNN(
            plaintext_vocab_size=self.vocab['vocab_size'],
            embed_size=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            plaintext_vocab=self.vocab
        )
        
        # load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # initialize inference dataset and metrics
        self.inference_dataset = InferenceDataset(vocab_path)
        self.metrics_calculator = BoundedMetrics()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Vocabulary size: {self.vocab['vocab_size']}")
    
    def decrypt_single_image(self, image_path: str, max_length: int = 200) -> str:
        """Decrypt a single image"""
        try:
            image = self.inference_dataset.load_and_preprocess_image(image_path)
            image_tensor = image.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output_indices = self.model(image_tensor, max_length=max_length)
                output_text = self.inference_dataset.indices_to_plaintext(
                    output_indices[0].cpu().numpy()
                )
            
            return output_text.strip()
            
        except Exception as e:
            print(f"Error decrypting {image_path}: {e}")
            return ""
    
    def evaluate_dataset(self, data_file: str, image_dir: str, max_length: int = 200, 
                        save_results: bool = True, output_dir: str = 'inference_results') -> Dict:
        """Evaluate model on a complete dataset"""
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        predictions = []
        references = []
        results = []
        
        print(f"Evaluating on {len(data)} samples...")
        
        for filename, item in tqdm(data.items(), desc="Processing images"):
            image_path = Path(image_dir) / filename
            
            if not image_path.exists():
                print(f"Warning: Image not found: {image_path}")
                continue
            
            predicted_text = self.decrypt_single_image(image_path, max_length)
            true_text = item['plaintext']
            
            predictions.append(predicted_text)
            references.append(true_text)
            
            case_sensitive_edit_dist = editdistance.eval(predicted_text, true_text)
            case_insensitive_edit_dist = editdistance.eval(predicted_text.lower().strip(), true_text.lower().strip())
            
            results.append({
                'filename': filename,
                'predicted_text': predicted_text,
                'true_text': true_text,
                'edit_distance_case_sensitive': case_sensitive_edit_dist,
                'edit_distance_case_insensitive': case_insensitive_edit_dist,
                'normalized_edit_distance_case_sensitive': case_sensitive_edit_dist / max(len(true_text), 1),
                'normalized_edit_distance_case_insensitive': case_insensitive_edit_dist / max(len(true_text), 1),
                'edit_distance': case_sensitive_edit_dist,
                'normalized_edit_distance': case_sensitive_edit_dist / max(len(true_text), 1)
            })
        
        metrics = self.metrics_calculator.compute_all_metrics(predictions, references)
        
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            
            results_df = pd.DataFrame(results)
            results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False)
            
            with open(f"{output_dir}/metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            comparison_df = pd.DataFrame({
                'filename': [r['filename'] for r in results],
                'predicted': predictions,
                'reference': references,
            })
            comparison_df.to_csv(f"{output_dir}/predictions_comparison.csv", index=False)
            
            print(f"Results saved to {output_dir}/")
        
        return {
            'metrics': metrics,
            'detailed_results': results,
            'predictions': predictions,
            'references': references
        }
    
    def generate_analysis_plots(self, results: Dict, output_dir: str = 'inference_results'):
        """Generate analysis plots"""
        os.makedirs(output_dir, exist_ok=True)
        
        detailed_results = results['detailed_results']
        metrics = results['metrics']
        
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle('Image Decryption Model Analysis (Case-Sensitive)', fontsize=16)
        
        # 1. Metrics overview
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        axes[0, 0].bar(range(len(metric_names)), metric_values, color='skyblue')
        axes[0, 0].set_xticks(range(len(metric_names)))
        axes[0, 0].set_xticklabels([name.replace('_', '\n') for name in metric_names], rotation=45, ha='right')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('All Metrics Overview')
        axes[0, 0].set_ylim(0, 1)
        
        for i, v in enumerate(metric_values):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. Edit distance distribution (case-sensitive)
        edit_distances = [r['normalized_edit_distance'] for r in detailed_results]
        axes[0, 1].hist(edit_distances, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(edit_distances), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(edit_distances):.3f}')
        axes[0, 1].set_xlabel('Normalized Edit Distance (Case-Sensitive)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Edit Distance Distribution')
        axes[0, 1].legend()
        
        # 3. Case-sensitive vs Case-insensitive comparison
        case_sensitive_distances = [r['normalized_edit_distance_case_sensitive'] for r in detailed_results]
        case_insensitive_distances = [r['normalized_edit_distance_case_insensitive'] for r in detailed_results]
        
        axes[0, 2].scatter(case_sensitive_distances, case_insensitive_distances, alpha=0.6, s=30, color='red')
        axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.8)
        axes[0, 2].set_xlabel('Case-Sensitive Edit Distance')
        axes[0, 2].set_ylabel('Case-Insensitive Edit Distance')
        axes[0, 2].set_title('Case Sensitivity Impact')
        
        diff = np.mean(np.array(case_insensitive_distances) - np.array(case_sensitive_distances))
        axes[0, 2].text(0.05, 0.95, f'Avg diff (insens - sens): {diff:.3f}', 
                       transform=axes[0, 2].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # 4. Text length analysis
        pred_lengths = [len(p) for p in results['predictions']]
        ref_lengths = [len(r) for r in results['references']]
        
        axes[0, 3].scatter(ref_lengths, pred_lengths, alpha=0.6, s=30, color='purple')
        axes[0, 3].plot([0, max(max(pred_lengths), max(ref_lengths))], 
                       [0, max(max(pred_lengths), max(ref_lengths))], 'r--', alpha=0.8)
        axes[0, 3].set_xlabel('Reference Length')
        axes[0, 3].set_ylabel('Prediction Length')
        axes[0, 3].set_title('Length Correlation')
        
        correlation = np.corrcoef(ref_lengths, pred_lengths)[0, 1]
        axes[0, 3].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[0, 3].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        # 5. Success rate by text length
        length_bins = np.arange(0, max(ref_lengths) + 20, 20)
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
            axes[1, 0].plot(length_centers, success_by_length, 'o-', color='green', linewidth=2, markersize=6)
            axes[1, 0].set_xlabel('Text Length')
            axes[1, 0].set_ylabel('Decryption Success Rate')
            axes[1, 0].set_title('Success Rate by Text Length')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 6. Top/Bottom examples
        sorted_results = sorted(detailed_results, key=lambda x: x['normalized_edit_distance'])
        best_examples = sorted_results[:3]
        worst_examples = sorted_results[-3:]
        
        axes[1, 1].text(0.1, 0.9, "BEST EXAMPLES:", transform=axes[1, 1].transAxes, 
                       fontweight='bold', fontsize=10, color='green')
        y_pos = 0.8
        for i, ex in enumerate(best_examples):
            axes[1, 1].text(0.1, y_pos - i*0.15, 
                          f"File: {ex['filename'][:20]}...\nEdit Dist: {ex['normalized_edit_distance']:.3f}", 
                          transform=axes[1, 1].transAxes, fontsize=8)
        
        axes[1, 1].text(0.1, 0.4, "WORST EXAMPLES:", transform=axes[1, 1].transAxes, 
                       fontweight='bold', fontsize=10, color='red')
        y_pos = 0.3
        for i, ex in enumerate(worst_examples):
            axes[1, 1].text(0.1, y_pos - i*0.08, 
                          f"File: {ex['filename'][:20]}...\nEdit Dist: {ex['normalized_edit_distance']:.3f}", 
                          transform=axes[1, 1].transAxes, fontsize=8)
        
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Best/Worst Examples')
        
        # 7. Character frequency (case-sensitive: no lowercasing)
        all_pred_chars = ''.join(results['predictions'])
        all_ref_chars = ''.join(results['references'])
        
        from collections import Counter
        pred_char_freq = Counter(all_pred_chars)
        ref_char_freq = Counter(all_ref_chars)
        
        common_chars = sorted(ref_char_freq.keys(), key=lambda x: ref_char_freq[x], reverse=True)[:20]
        pred_freqs = [pred_char_freq.get(char, 0) for char in common_chars]
        ref_freqs = [ref_char_freq[char] for char in common_chars]
        
        x_pos = np.arange(len(common_chars))
        width = 0.35
        
        axes[1, 2].bar(x_pos - width/2, ref_freqs, width, label='Reference', alpha=0.8, color='skyblue')
        axes[1, 2].bar(x_pos + width/2, pred_freqs, width, label='Predicted', alpha=0.8, color='lightcoral')
        axes[1, 2].set_xlabel('Characters')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Character Frequency Comparison (Case-Sensitive)')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels([repr(char) for char in common_chars], rotation=45)
        axes[1, 2].legend()
        
        # 8. Histogram: case-insensitive vs case-sensitive difference
        improvements = np.array(case_insensitive_distances) - np.array(case_sensitive_distances)
        axes[1, 3].hist(improvements, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 3].axvline(np.mean(improvements), color='green', linestyle='--', 
                          label=f'Mean: {np.mean(improvements):.3f}')
        axes[1, 3].set_xlabel('Edit Distance (Case-Insens - Case-Sens)')
        axes[1, 3].set_ylabel('Frequency')
        axes[1, 3].set_title('Case-Insensitive vs Case-Sensitive')
        axes[1, 3].legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/analysis_plots.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/analysis_plots.pdf", bbox_inches='tight')
        plt.show()
        
        print(f"Analysis plots saved to {output_dir}/")
    
    def print_detailed_metrics(self, metrics: Dict):
        """Print detailed metrics with explanations"""
        print("\n" + "="*80)
        print("IMAGE DECRYPTION METRICS REPORT (CASE-SENSITIVE)")
        print("="*80)
        
        print(f"\nACCURACY METRICS:")
        print(f"   Character Accuracy:       {metrics['character_accuracy']:.4f} (higher is better, max=1.0)")
        print(f"   Sequence Accuracy:        {metrics['sequence_accuracy']:.4f} (exact matches, max=1.0)")
        print(f"   Decryption Success Rate:  {metrics['decryption_success_rate']:.4f} (overall success, max=1.0)")
        
        print(f"\nDISTANCE METRICS:")
        print(f"   Normalized Edit Distance: {metrics['normalized_edit_distance']:.4f} (lower is better, min=0.0)")
        print(f"   Word Error Rate:          {metrics['word_error_rate']:.4f} (lower is better, min=0.0)")
        print(f"   Character Error Rate:     {metrics['character_error_rate']:.4f} (lower is better, min=0.0)")
        
        print(f"\nQUALITY METRICS:")
        print(f"   BLEU Score:               {metrics['bleu_score']:.4f} (higher is better, max=1.0)")
        
        print(f"\nPERFORMANCE SUMMARY:")
        avg_score = np.mean([
            metrics['character_accuracy'],
            metrics['sequence_accuracy'], 
            metrics['decryption_success_rate'],
            metrics['bleu_score'],
            1 - metrics['normalized_edit_distance'],
            1 - metrics['word_error_rate'],
            1 - metrics['character_error_rate']
        ])
        print(f"   Overall Performance:      {avg_score:.4f} (composite score, max=1.0)")
        print("\nNOTE: All metrics computed with case-sensitive comparison")
        print("      (e.g., 'CopiAle' and 'copiale' count as different)")
        print("="*80)
    
    def save_examples(self, results: Dict, output_dir: str, num_examples: int = 10):
        """Save detailed examples for manual inspection"""
        detailed_results = results['detailed_results']
        
        sorted_results = sorted(detailed_results, key=lambda x: x['normalized_edit_distance'])
        best_examples = sorted_results[:num_examples]
        worst_examples = sorted_results[-num_examples:]
        
        with open(f"{output_dir}/best_examples.txt", 'w', encoding='utf-8') as f:
            f.write("BEST IMAGE DECRYPTION EXAMPLES (Case-Sensitive Evaluation)\n")
            f.write("="*80 + "\n\n")
            
            for i, ex in enumerate(best_examples):
                f.write(f"EXAMPLE {i+1} (Case-Sensitive Edit Distance: {ex['normalized_edit_distance']:.4f})\n")
                f.write(f"           (Case-Insensitive Edit Distance: {ex['normalized_edit_distance_case_insensitive']:.4f})\n")
                f.write(f"File: {ex['filename']}\n")
                f.write(f"True Text: {ex['true_text']}\n")
                f.write(f"Predicted: {ex['predicted_text']}\n")
                f.write("-" * 80 + "\n\n")
        
        with open(f"{output_dir}/case_sensitivity_analysis.txt", 'w', encoding='utf-8') as f:
            f.write("CASE SENSITIVITY COMPARISON (IMAGE DECRYPTION)\n")
            f.write("="*80 + "\n\n")
            
            case_improvements = []
            for ex in detailed_results:
                diff = ex['normalized_edit_distance_case_insensitive'] - ex['normalized_edit_distance_case_sensitive']
                if diff < 0:  # case-insensitive was better (lower distance)
                    case_improvements.append((ex, -diff))
            
            case_improvements.sort(key=lambda x: x[1], reverse=True)
            
            f.write(f"EXAMPLES WHERE CASE-INSENSITIVE EVAL WOULD HELP: {len(case_improvements)}\n\n")
            f.write("TOP EXAMPLES (case-insensitive distance lower than case-sensitive):\n")
            f.write("-" * 80 + "\n\n")
            
            for i, (ex, improvement) in enumerate(case_improvements[:10]):
                f.write(f"EXAMPLE {i+1} (Improvement if case-insens: {improvement:.4f})\n")
                f.write(f"File: {ex['filename']}\n")
                f.write(f"True Text: {ex['true_text']}\n")
                f.write(f"Predicted: {ex['predicted_text']}\n")
                f.write(f"Case-Sensitive Distance: {ex['normalized_edit_distance_case_sensitive']:.4f}\n")
                f.write(f"Case-Insensitive Distance: {ex['normalized_edit_distance_case_insensitive']:.4f}\n")
                f.write("-" * 40 + "\n\n")
        
        print(f"Example files saved to {output_dir}/")


def main():
    """main function for inference; set paths below."""
    
    model_path = "path/to/decipher_model.pth"
    vocab_path = "path/to/plaintext_vocabulary.json"
    data_file = "path/to/test_data.json"
    image_dir = "path/to/test_images"
    output_dir = "./inference_results"
    max_length = 200
    device = "cuda"
    
    single_image = None
    
    print("Starting image decryption inference (case-sensitive)...")
    print(f"Model: {model_path}")
    print(f"Vocabulary: {vocab_path}")
    print(f"Device: {device}")
    print("Note: all evaluations are case-sensitive (e.g. 'CopiAle' != 'copiale').")
    
    inference = DecryptionInference(model_path, vocab_path, device)
    
    if single_image:
        print(f"Decrypting single image: {single_image}")
        result = inference.decrypt_single_image(single_image, max_length)
        print(f"Decrypted text: {result}")
    else:
        print(f"Evaluating on dataset: {data_file}")
        results = inference.evaluate_dataset(
            data_file,
            image_dir,
            max_length,
            save_results=True,
            output_dir=output_dir
        )
        
        inference.print_detailed_metrics(results['metrics'])
        
        detailed_results = results['detailed_results']
        case_insens_better = sum(1 for ex in detailed_results 
                                if ex['normalized_edit_distance_case_insensitive'] < ex['normalized_edit_distance_case_sensitive'])
        
        print(f"\nCASE SENSITIVITY SUMMARY:")
        print(f"   Examples where case-insensitive would be better: {case_insens_better}/{len(detailed_results)} ({100*case_insens_better/len(detailed_results):.1f}%)")
        
        inference.generate_analysis_plots(results, output_dir)
        inference.save_examples(results, output_dir)
        
        print(f"\nEvaluation complete. Check {output_dir}/ for detailed results.")


if __name__ == "__main__":
    main()
