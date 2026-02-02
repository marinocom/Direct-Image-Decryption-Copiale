"""
decryption inference (case-insensitive).

This script runs the 2-step architecture decryption model that maps cipher transcriptions
(space-separated token sequences) to plaintext. It supports:

- Single transcription decryption: one input string → one plaintext output.
- Batch evaluation: INPUT: a JSON test set (filename → {transcription, plaintext}) with
  metrics and optional plots.


All evaluation metrics (character/sequence accuracy, edit distance, BLEU, WER/CER,
decryption success rate) are computed in a case-insensitive way: predictions and
references are lowercased before comparison, so e.g. "CopiAle" and "copiale" count
as an exact match.

Outputs (when save_results=True): CSV results, metrics summary, analysis plots,
best/worst examples, and a short case-sensitivity analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
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

# import the model architecture from the model creation script
from decryptionModelCreation import (
    CopialeDecipherModel
)


class InferenceDataset:
    """Simplified dataset class for integrated model inference"""
    
    def __init__(self, vocab_path: str):
        with open(vocab_path, 'r') as f:
            self.vocabs = json.load(f)
        
        # handle both formats: full vocabularies or single vocab
        if 'transcription' in self.vocabs and 'plaintext' in self.vocabs:
            # full vocabulary file from integrated training
            self.transcription_vocab = self._convert_vocab_keys(self.vocabs['transcription'])
            self.plaintext_vocab = self._convert_vocab_keys(self.vocabs['plaintext'])
        else:
            # single vocabulary file (plaintext only)
            self.plaintext_vocab = self.vocabs
            self.transcription_vocab = None
    
    def _convert_vocab_keys(self, vocab_dict):
        """Convert string keys to int for idx_to_token/idx_to_char if needed"""
        converted = vocab_dict.copy()
        if 'idx_to_token' in converted:
            converted['idx_to_token'] = {int(k) if isinstance(k, str) else k: v 
                                         for k, v in converted['idx_to_token'].items()}
        if 'idx_to_char' in converted:
            converted['idx_to_char'] = {int(k) if isinstance(k, str) else k: v 
                                        for k, v in converted['idx_to_char'].items()}
        return converted
    
    def transcription_to_indices(self, transcription):
        """Convert transcription tokens to indices"""
        if self.transcription_vocab is None:
            raise ValueError("Transcription vocabulary not available")
        
        indices = [self.transcription_vocab['token_to_idx']['< SOS >']]
        tokens = transcription.split()
        for token in tokens:
            indices.append(self.transcription_vocab['token_to_idx'].get(
                token, self.transcription_vocab['token_to_idx']['<UNK>']))
        indices.append(self.transcription_vocab['token_to_idx']['<EOS>'])
        return indices
    
    def indices_to_plaintext(self, indices):
        """Convert indices back to plaintext"""
        chars = []
        for idx in indices:
            # handle both string and integer keys
            char = self.plaintext_vocab['idx_to_char'].get(
                str(idx), self.plaintext_vocab['idx_to_char'].get(idx, '<UNK>'))
            if char in ['< SOS >', '<EOS>', '<PAD>']:
                continue
            chars.append(char)
        return ''.join(chars)


class BoundedMetrics:
    """Class to compute bounded metrics for decryption evaluation with case-insensitive comparison"""
    
    def __init__(self):
        self.metrics = {}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for case-insensitive comparison"""
        return text.lower().strip()
    
    def character_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Calculate character-level accuracy (0-1) - case insensitive"""
        total_chars = 0
        correct_chars = 0
        
        for pred, ref in zip(predictions, references):
            # normalize both texts to lowercase
            pred_norm = self._normalize_text(pred)
            ref_norm = self._normalize_text(ref)
            
            # align lengths by taking minimum
            min_len = min(len(pred_norm), len(ref_norm))
            total_chars += max(len(pred_norm), len(ref_norm))
            
            # count correct characters at aligned positions
            for i in range(min_len):
                if pred_norm[i] == ref_norm[i]:
                    correct_chars += 1
        
        return correct_chars / total_chars if total_chars > 0 else 0.0
    
    def normalized_edit_distance(self, predictions: List[str], references: List[str]) -> float:
        """Calculate normalized edit distance (0-1, lower is better) - case insensitive"""
        distances = []
        for pred, ref in zip(predictions, references):
            # normalize both texts to lowercase
            pred_norm = self._normalize_text(pred)
            ref_norm = self._normalize_text(ref)
            
            edit_dist = editdistance.eval(pred_norm, ref_norm)
            max_len = max(len(pred_norm), len(ref_norm), 1)
            normalized_dist = edit_dist / max_len
            distances.append(normalized_dist)
        
        return np.mean(distances)
    
    def sequence_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Calculate exact sequence match accuracy (0-1) - case insensitive"""
        exact_matches = 0
        for pred, ref in zip(predictions, references):
            pred_norm = self._normalize_text(pred)
            ref_norm = self._normalize_text(ref)
            if pred_norm == ref_norm:
                exact_matches += 1
        
        return exact_matches / len(predictions) if predictions else 0.0
    
    def bounded_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate character-level BLEU score approximation (0-1) - case insensitive"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            smoothie = SmoothingFunction().method4
            
            scores = []
            for pred, ref in zip(predictions, references):
                # normalize and convert to character lists
                pred_norm = self._normalize_text(pred)
                ref_norm = self._normalize_text(ref)
                pred_chars = list(pred_norm)
                ref_chars = [list(ref_norm)]
                
                if len(pred_chars) == 0:
                    scores.append(0.0)
                else:
                    score = sentence_bleu(ref_chars, pred_chars, smoothing_function=smoothie)
                    scores.append(score)
            
            return np.mean(scores)
        except ImportError:
            # fallback to simple n-gram overlap
            return self._simple_ngram_overlap(predictions, references)
    
    def _simple_ngram_overlap(self, predictions: List[str], references: List[str], n=4) -> float:
        """Simple n-gram overlap as BLEU approximation - case insensitive"""
        scores = []
        for pred, ref in zip(predictions, references):
            pred_norm = self._normalize_text(pred)
            ref_norm = self._normalize_text(ref)
            
            if len(pred_norm) == 0:
                scores.append(0.0)
                continue
            
            overlap_scores = []
            for gram_size in range(1, min(n + 1, len(pred_norm) + 1)):
                pred_grams = set([pred_norm[i:i+gram_size] for i in range(len(pred_norm) - gram_size + 1)])
                ref_grams = set([ref_norm[i:i+gram_size] for i in range(len(ref_norm) - gram_size + 1)])
                
                if len(pred_grams) == 0:
                    overlap_scores.append(0.0)
                else:
                    overlap = len(pred_grams & ref_grams) / len(pred_grams)
                    overlap_scores.append(overlap)
            
            scores.append(np.mean(overlap_scores))
        
        return np.mean(scores)
    
    def decryption_success_rate(self, predictions: List[str], references: List[str], 
                                threshold: float = 0.3) -> float:
        """Calculate percentage of decryptions with edit distance below threshold"""
        successes = 0
        for pred, ref in zip(predictions, references):
            pred_norm = self._normalize_text(pred)
            ref_norm = self._normalize_text(ref)
            edit_dist = editdistance.eval(pred_norm, ref_norm)
            max_len = max(len(pred_norm), len(ref_norm), 1)
            if edit_dist / max_len < threshold:
                successes += 1
        
        return successes / len(predictions) if predictions else 0.0
    
    def compute_all_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute all metrics at once"""
        # case-insensitive metrics
        metrics = {
            'character_accuracy': self.character_accuracy(predictions, references),
            'normalized_edit_distance': self.normalized_edit_distance(predictions, references),
            'sequence_accuracy': self.sequence_accuracy(predictions, references),
            'bleu_score': self.bounded_bleu_score(predictions, references),
            'decryption_success_rate': self.decryption_success_rate(predictions, references),
        }
        
        # add WER and CER using jiwer (case-insensitive)
        try:
            # normalize for case-insensitive comparison
            pred_norm = [self._normalize_text(p) for p in predictions]
            ref_norm = [self._normalize_text(r) for r in references]
            metrics['word_error_rate'] = wer(ref_norm, pred_norm)
            metrics['character_error_rate'] = cer(ref_norm, pred_norm)
        except Exception as e:
            print(f"Warning: Could not compute WER/CER: {e}")
            metrics['word_error_rate'] = 1.0
            metrics['character_error_rate'] = 1.0
        
        return metrics


class IntegratedDecryptionInference:
    """Inference class for the integrated decryption model"""
    
    def __init__(self, model_path: str, vocab_path: str, transcription_vocab_path: str = None, device: str = 'cuda'):
        """
        Initialize inference
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to plaintext vocabulary file
            transcription_vocab_path: Path to transcription vocabulary file (optional)
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # load vocabularies
        print(f"Loading plaintext vocabulary from {vocab_path}...")
        self.dataset = InferenceDataset(vocab_path)
        
        # load transcription vocabulary if provided separately
        if transcription_vocab_path:
            print(f"Loading transcription vocabulary from {transcription_vocab_path}...")
            with open(transcription_vocab_path, 'r') as f:
                transcription_vocab = json.load(f)
            self.dataset.transcription_vocab = self.dataset._convert_vocab_keys(transcription_vocab)
            print(f"Transcription vocab loaded: {self.dataset.transcription_vocab['vocab_size']} tokens")
        
        # load model
        print(f"Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        if self.dataset.transcription_vocab:
            print(f"Transcription vocab size: {self.dataset.transcription_vocab['vocab_size']}")
        print(f"Plaintext vocab size: {self.dataset.plaintext_vocab['vocab_size']}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model"""
        # load checkpoint first to get vocab sizes
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # extract actual vocab sizes from the model checkpoint
        embedding_weight = checkpoint['model_state_dict']['encoder_embedding.weight']
        actual_transcription_vocab_size = embedding_weight.shape[0]
        
        decoder_weight_key = None
        for key in checkpoint['model_state_dict'].keys():
            if 'decoder' in key and 'weight' in key and len(checkpoint['model_state_dict'][key].shape) == 2:
                decoder_weight_key = key
                break
        
        if decoder_weight_key:
            actual_plaintext_vocab_size = checkpoint['model_state_dict'][decoder_weight_key].shape[0]
        else:
            actual_plaintext_vocab_size = self.dataset.plaintext_vocab['vocab_size']
        
        print(f"\nModel checkpoint vocabulary sizes:")
        print(f"  Transcription vocab (from model): {actual_transcription_vocab_size}")
        print(f"  Plaintext vocab (from model): {actual_plaintext_vocab_size}")
        
        # check if we have transcription vocab
        if self.dataset.transcription_vocab is not None:
            loaded_transcription_vocab_size = self.dataset.transcription_vocab['vocab_size']
            print(f"\nLoaded vocabulary sizes:")
            print(f"  Transcription vocab (from file): {loaded_transcription_vocab_size}")
            print(f"  Plaintext vocab (from file): {self.dataset.plaintext_vocab['vocab_size']}")
            
            if loaded_transcription_vocab_size != actual_transcription_vocab_size:
                print(f"\n⚠️  WARNING: Vocabulary size mismatch!")
                print(f"  Model expects {actual_transcription_vocab_size} transcription tokens")
                print(f"  Vocabulary file has {loaded_transcription_vocab_size} tokens")
                print(f"  Difference: {actual_transcription_vocab_size - loaded_transcription_vocab_size} tokens")
                print(f"\n  Using model's vocabulary size ({actual_transcription_vocab_size}) to load the model.")
                print(f"  Note: This may cause issues if the missing tokens are needed during inference.")
                
                # check what tokens might be missing
                if 'token_to_idx' in self.dataset.transcription_vocab:
                    print(f"\n  Tokens in vocabulary file: {len(self.dataset.transcription_vocab['token_to_idx'])}")
                    print(f"  Sample tokens: {list(self.dataset.transcription_vocab['token_to_idx'].keys())[:10]}")
            
            transcription_vocab_size = actual_transcription_vocab_size  # use model's size
            plaintext_vocab_size = self.dataset.plaintext_vocab['vocab_size']
        else:
            print(f"Warning: Transcription vocab not found.")
            print(f"Inferred transcription_vocab_size={actual_transcription_vocab_size} from model checkpoint")
            print(f"Note: You won't be able to decrypt new transcriptions without the transcription vocab!")
            transcription_vocab_size = actual_transcription_vocab_size
            plaintext_vocab_size = self.dataset.plaintext_vocab['vocab_size']
        
        # create model architecture
        model = CopialeDecipherModel(
            transcription_vocab_size=transcription_vocab_size,
            plaintext_vocab_size=plaintext_vocab_size,
            embed_size=128,  # match training config
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            plaintext_vocab=self.dataset.plaintext_vocab
        )
        
        # load checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def decrypt_single_transcription(self, transcription: str, max_length: int = 200) -> str:
        """
        Decrypt a single transcription
        Args:
            transcription: Input transcription string (space-separated tokens)
            max_length: Maximum output length
        Returns:
            Decrypted plaintext string
        """
        # convert to indices
        input_indices = self.dataset.transcription_to_indices(transcription)
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # generate output
            output_indices = self.model(input_tensor, max_length=max_length)
            
            # convert to text
            output_text = self.dataset.indices_to_plaintext(output_indices[0].cpu().numpy())
        
        return output_text
    
    def evaluate_dataset(self, data_file: str, max_length: int = 200, 
                        save_results: bool = True, output_dir: str = './results') -> Dict:
        """
        Evaluate model on a dataset
        Args:
            data_file: JSON file with test data
            max_length: Maximum output length
            save_results: Whether to save detailed results
            output_dir: Directory to save results
        Returns:
            Dictionary with metrics and detailed results
        """
        # load test data
        with open(data_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"Evaluating on {len(test_data)} samples...")
        
        predictions = []
        references = []
        detailed_results = []
        
        # process each sample
        for filename, item in tqdm(test_data.items(), desc="Decrypting"):
            transcription = item['transcription']
            true_plaintext = item['plaintext']
            
            # decrypt
            predicted_plaintext = self.decrypt_single_transcription(transcription, max_length)
            
            predictions.append(predicted_plaintext)
            references.append(true_plaintext)
            
            # calculate per-sample metrics (both case-sensitive and case-insensitive)
            # case-insensitive (main metric)
            pred_norm = predicted_plaintext.lower().strip()
            true_norm = true_plaintext.lower().strip()
            edit_dist_insensitive = editdistance.eval(pred_norm, true_norm)
            normalized_ed_insensitive = edit_dist_insensitive / max(len(pred_norm), len(true_norm), 1)
            
            # case-sensitive (for comparison)
            edit_dist_sensitive = editdistance.eval(predicted_plaintext, true_plaintext)
            normalized_ed_sensitive = edit_dist_sensitive / max(len(predicted_plaintext), len(true_plaintext), 1)
            
            detailed_results.append({
                'filename': filename,
                'transcription': transcription,
                'true_text': true_plaintext,
                'predicted_text': predicted_plaintext,
                'normalized_edit_distance': normalized_ed_insensitive,  # case-insensitive (primary)
                'normalized_edit_distance_case_insensitive': normalized_ed_insensitive,
                'normalized_edit_distance_case_sensitive': normalized_ed_sensitive,
            })
        
        # compute overall metrics (case-insensitive)
        metric_calculator = BoundedMetrics()
        metrics = metric_calculator.compute_all_metrics(predictions, references)
        
        # save results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            self._save_results(detailed_results, metrics, output_dir)
        
        return {
            'metrics': metrics,
            'detailed_results': detailed_results,
            'predictions': predictions,
            'references': references
        }
    
    def _save_results(self, detailed_results: List[Dict], metrics: Dict, output_dir: str):
        """Save detailed results and metrics"""
        # save detailed results csv
        df = pd.DataFrame(detailed_results)
        df.to_csv(f"{output_dir}/detailed_results.csv", index=False, encoding='utf-8')
        
        # save predictions comparison
        comparison_df = pd.DataFrame({
            'filename': [r['filename'] for r in detailed_results],
            'transcription': [r['transcription'] for r in detailed_results],
            'true_text': [r['true_text'] for r in detailed_results],
            'predicted_text': [r['predicted_text'] for r in detailed_results],
            'true_text_normalized': [r['true_text'].lower().strip() for r in detailed_results],
            'predicted_text_normalized': [r['predicted_text'].lower().strip() for r in detailed_results],
            'edit_distance_case_insensitive': [r['normalized_edit_distance_case_insensitive'] for r in detailed_results],
            'edit_distance_case_sensitive': [r['normalized_edit_distance_case_sensitive'] for r in detailed_results],
        })
        comparison_df.to_csv(f"{output_dir}/predictions_comparison.csv", index=False, encoding='utf-8')
        
        # save metrics summary
        with open(f"{output_dir}/metrics_summary.txt", 'w') as f:
            f.write("INTEGRATED MODEL DECRYPTION METRICS (Case-Insensitive)\n")
            f.write("="*80 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        
        # save metrics as json
        with open(f"{output_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Results saved to {output_dir}/")
    
    def generate_analysis_plots(self, results: Dict, output_dir: str):
        """Generate visualization plots"""
        os.makedirs(output_dir, exist_ok=True)
        detailed_results = results['detailed_results']
        
        # set style
        sns.set_style("whitegrid")
        
        # 1. edit distance distribution (case-insensitive)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        edit_distances = [r['normalized_edit_distance_case_insensitive'] for r in detailed_results]
        axes[0, 0].hist(edit_distances, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Normalized Edit Distance (Case-Insensitive)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Edit Distances')
        axes[0, 0].axvline(np.mean(edit_distances), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(edit_distances):.3f}')
        axes[0, 0].legend()
        
        # 2. case sensitivity impact
        case_sensitive = [r['normalized_edit_distance_case_sensitive'] for r in detailed_results]
        case_insensitive = [r['normalized_edit_distance_case_insensitive'] for r in detailed_results]
        
        axes[0, 1].scatter(case_sensitive, case_insensitive, alpha=0.5)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Equal')
        axes[0, 1].set_xlabel('Case-Sensitive Edit Distance')
        axes[0, 1].set_ylabel('Case-Insensitive Edit Distance')
        axes[0, 1].set_title('Case Sensitivity Impact')
        axes[0, 1].legend()
        
        # 3. Success Rate by Threshold
        thresholds = np.linspace(0, 1, 50)
        success_rates = []
        for thresh in thresholds:
            success = sum(1 for ed in edit_distances if ed < thresh) / len(edit_distances)
            success_rates.append(success)
        
        axes[1, 0].plot(thresholds, success_rates, linewidth=2)
        axes[1, 0].set_xlabel('Edit Distance Threshold')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title('Decryption Success Rate vs Threshold')
        axes[1, 0].grid(True)
        
        # 4. length analysis
        pred_lengths = [len(r['predicted_text']) for r in detailed_results]
        true_lengths = [len(r['true_text']) for r in detailed_results]
        
        axes[1, 1].scatter(true_lengths, pred_lengths, alpha=0.5)
        axes[1, 1].plot([0, max(true_lengths)], [0, max(true_lengths)], 'r--', label='Perfect')
        axes[1, 1].set_xlabel('True Text Length')
        axes[1, 1].set_ylabel('Predicted Text Length')
        axes[1, 1].set_title('Length Comparison')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/analysis_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Metrics Summary Plot
        metrics = results['metrics']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        colors = ['green' if 'accuracy' in name or 'bleu' in name or 'success' in name 
                 else 'red' for name in metric_names]
        
        ax.barh(metric_names, metric_values, color=colors, alpha=0.7)
        ax.set_xlabel('Score')
        ax.set_title('Integrated Model Performance Metrics (Case-Insensitive)')
        ax.set_xlim(0, 1)
        
        for i, v in enumerate(metric_values):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plots saved to {output_dir}/")
    
    def print_detailed_metrics(self, metrics: Dict):
        """Print detailed metrics report"""
        print("\n")
        print("="*80)
        print("INTEGRATED MODEL DECRYPTION EVALUATION RESULTS (Case-Insensitive)")
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
        print(f"   BLEU Score:              {metrics['bleu_score']:.4f} (higher is better, max=1.0)")
        
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
        print("\nNOTE: All metrics computed with case-insensitive comparison")
        print("      (e.g., 'StudentS' vs 'students' counts as exact match)")
        print("="*80)
    
    def save_examples(self, results: Dict, output_dir: str, num_examples: int = 10):
        """Save detailed examples for manual inspection"""
        detailed_results = results['detailed_results']
        
        # sort by case-insensitive edit distance
        sorted_results = sorted(detailed_results, key=lambda x: x['normalized_edit_distance'])
        
        # best and worst examples
        best_examples = sorted_results[:num_examples]
        worst_examples = sorted_results[-num_examples:]
        
        # save to text files for easy reading
        with open(f"{output_dir}/best_examples.txt", 'w', encoding='utf-8') as f:
            f.write("BEST DECRYPTION EXAMPLES (Case-Insensitive Evaluation)\n")
            f.write("="*80 + "\n\n")
            
            for i, ex in enumerate(best_examples):
                f.write(f"EXAMPLE {i+1} (Case-Insensitive Edit Distance: {ex['normalized_edit_distance']:.4f})\n")
                f.write(f"           (Case-Sensitive Edit Distance: {ex['normalized_edit_distance_case_sensitive']:.4f})\n")
                f.write(f"File: {ex['filename']}\n")
                f.write(f"Transcription: {ex['transcription']}\n")
                f.write(f"True Text: {ex['true_text']}\n")
                f.write(f"Predicted: {ex['predicted_text']}\n")
                f.write(f"True Text (normalized): {ex['true_text'].lower().strip()}\n")
                f.write(f"Predicted (normalized): {ex['predicted_text'].lower().strip()}\n")
                f.write("-" * 80 + "\n\n")
        
        with open(f"{output_dir}/worst_examples.txt", 'w', encoding='utf-8') as f:
            f.write("WORST DECRYPTION EXAMPLES (Case-Insensitive Evaluation)\n")
            f.write("="*80 + "\n\n")
            
            for i, ex in enumerate(worst_examples):
                f.write(f"EXAMPLE {i+1} (Case-Insensitive Edit Distance: {ex['normalized_edit_distance']:.4f})\n")
                f.write(f"           (Case-Sensitive Edit Distance: {ex['normalized_edit_distance_case_sensitive']:.4f})\n")
                f.write(f"File: {ex['filename']}\n")
                f.write(f"Transcription: {ex['transcription']}\n")
                f.write(f"True Text: {ex['true_text']}\n")
                f.write(f"Predicted: {ex['predicted_text']}\n")
                f.write(f"True Text (normalized): {ex['true_text'].lower().strip()}\n")
                f.write(f"Predicted (normalized): {ex['predicted_text'].lower().strip()}\n")
                f.write("-" * 80 + "\n\n")
        
        # additional analysis file for case sensitivity impact
        with open(f"{output_dir}/case_sensitivity_analysis.txt", 'w', encoding='utf-8') as f:
            f.write("CASE SENSITIVITY IMPACT ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            # find examples where case sensitivity made a big difference
            case_improvements = []
            for ex in detailed_results:
                improvement = ex['normalized_edit_distance_case_sensitive'] - ex['normalized_edit_distance_case_insensitive']
                if improvement > 0:  # cases where case-insensitive helped
                    case_improvements.append((ex, improvement))
            
            # sort by improvement amount
            case_improvements.sort(key=lambda x: x[1], reverse=True)
            
            f.write(f"TOTAL EXAMPLES WHERE CASE-INSENSITIVE HELPED: {len(case_improvements)}\n")
            f.write(f"AVERAGE IMPROVEMENT: {np.mean([imp for _, imp in case_improvements]) if case_improvements else 0:.4f}\n\n")
            
            f.write("TOP EXAMPLES WHERE CASE-INSENSITIVE EVALUATION HELPED MOST:\n")
            f.write("-" * 80 + "\n\n")
            
            for i, (ex, improvement) in enumerate(case_improvements[:10]):
                f.write(f"EXAMPLE {i+1} (Improvement: {improvement:.4f})\n")
                f.write(f"File: {ex['filename']}\n")
                f.write(f"True Text: {ex['true_text']}\n")
                f.write(f"Predicted: {ex['predicted_text']}\n")
                f.write(f"Case-Sensitive Distance: {ex['normalized_edit_distance_case_sensitive']:.4f}\n")
                f.write(f"Case-Insensitive Distance: {ex['normalized_edit_distance_case_insensitive']:.4f}\n")
                f.write("-" * 40 + "\n\n")
        
        print(f"Example files saved to {output_dir}/")


def main():
    """main function for integrated model inference; set paths below."""
    
    # configuration: set these paths for your environment
    model_path = ""
    vocab_path = ""
    transcription_vocab_path = ""
    data_file = ""
    output_dir = ""
    max_length = 200
    device = "cuda"
    
    # optional: set to a string for single-transcription decryption
    single_transcription = None  # e.g. "m^. a^^ c^. NorthEastArrow a^^ SquaredPlus BigT"
    
    print("Starting integrated model decryption inference (case-insensitive)...")
    print(f"Model: {model_path}")
    print(f"Plaintext vocabulary: {vocab_path}")
    print(f"Transcription vocabulary: {transcription_vocab_path}")
    print(f"Device: {device}")
    print("Note: all evaluations are case-insensitive (e.g. 'CopiAle' == 'copiale').")
    print("Note: using integrated model with transcription vocab from CRNN training.")
    
    # initialize inference
    inference = IntegratedDecryptionInference(
        model_path,
        vocab_path,
        transcription_vocab_path=transcription_vocab_path,
        device=device
    )
    
    if single_transcription:
        # single transcription decryption
        print(f"Decrypting single transcription: {single_transcription}")
        result = inference.decrypt_single_transcription(single_transcription, max_length)
        print(f"Decrypted text: {result}")
    else:
        # full dataset evaluation
        print(f"Evaluating on dataset: {data_file}")
        results = inference.evaluate_dataset(
            data_file,
            max_length,
            save_results=True,
            output_dir=output_dir
        )
        
        # print metrics
        inference.print_detailed_metrics(results['metrics'])
        
        # print case sensitivity impact summary
        detailed_results = results['detailed_results']
        case_improvements = []
        for ex in detailed_results:
            improvement = ex['normalized_edit_distance_case_sensitive'] - ex['normalized_edit_distance_case_insensitive']
            if improvement > 0:
                case_improvements.append(improvement)
        
        print(f"\nCASE SENSITIVITY IMPACT SUMMARY:")
        print(f"   Examples improved by case-insensitive eval: {len(case_improvements)}/{len(detailed_results)} ({100*len(case_improvements)/len(detailed_results):.1f}%)")
        if case_improvements:
            print(f"   Average improvement: {np.mean(case_improvements):.4f}")
            print(f"   Maximum improvement: {max(case_improvements):.4f}")
        
        # generate plots
        inference.generate_analysis_plots(results, output_dir)
        
        # save detailed examples
        inference.save_examples(results, output_dir)
        
        print(f"\nEvaluation complete. Check {output_dir}/ for detailed results.")
        print(f"Key files:")
        print(f"   - detailed_results.csv: Full results with both case-sensitive and case-insensitive metrics")
        print(f"   - case_sensitivity_analysis.txt: Analysis of case sensitivity impact")
        print(f"   - predictions_comparison.csv: Includes normalized (lowercase) versions of texts")
        print(f"   - metrics_summary.txt: Summary of all metrics")
        print(f"   - analysis_plots.png: Visualization of results")


if __name__ == "__main__":
    main()