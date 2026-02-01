"""
mismatch analyzer
-----------------
compares vocabulary and character sets between a training dataset and a test
dataset to detect mismatches that can hurt inference (e.g. tokens/characters
in test that never appear in training).

what it does:
  - loads two json datasets (train and test). each entry is expected to have
    a 'plaintext' field with the transcription; tokens are obtained by
    splitting on whitespace, matching typical training pipelines.
  - builds token vocabularies and character frequency counters for both sets.
  - compares token sets: tokens only in train, only in test, common; computes
    vocabulary overlap ratio and test coverage by training vocab.
  - does the same at character level (unique chars, overlap, test coverage).
  - identifies "problematic" tokens: tokens that appear in test but not in
    train, ranked by frequency in test (these are the ones most likely to be
    mapped to <unk> and cause errors).
  - generates token-to-index mappings for both vocabularies (for alignment).
  - optionally prints a detailed mismatch report (tokens/chars only in train
    or test, with counts).
  - saves a single json report containing: summary stats, token and character
    analysis, vocabulary mappings, and text recommendations (e.g. low coverage,
    many test-only tokens).
  - prints a final summary to the console (including a warning if test has
    tokens not in training).

configure train_json, test_json, output_file, and verbose in main(), then run
the script. output: vocabulary_mismatch_analysis.json (or path in output_file).
"""

import json
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import Counter

class VocabularyAnalyzer:
    """analyze vocabulary differences between training and test datasets"""
    
    def __init__(self):
        self.train_vocab = set()
        self.test_vocab = set()
        self.train_tokens = []
        self.test_tokens = []
        self.train_char_freq = Counter()
        self.test_char_freq = Counter()
        
    def build_vocabulary_from_json(self, json_file: str, dataset_type: str = "train") -> Set[str]:
        """
        build vocabulary from json file exactly like the training code
        returns set of tokens found in the dataset
        """
        print(f"Building {dataset_type} vocabulary from {json_file}...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokens = set()
        all_tokens = []
        char_freq = Counter()
        
        for filename, item in data.items():
            # get transcription text (same field as used in training)
            transcription = item.get('plaintext', '')
            
            if not transcription:
                print(f"Warning: Empty transcription for {filename}")
                continue
                
            # split by whitespace to get individual tokens (exactly like training)
            token_list = transcription.split()
            tokens.update(token_list)
            all_tokens.extend(token_list)
            
            # also count character frequencies
            for char in transcription:
                char_freq[char] += 1
        
        # add special tokens (exactly like training)
        special_tokens = {'<PAD>', '< SOS >', '<EOS>', '<UNK>'}
        tokens.update(special_tokens)
        
        # store results
        if dataset_type == "train":
            self.train_vocab = tokens
            self.train_tokens = all_tokens
            self.train_char_freq = char_freq
        else:
            self.test_vocab = tokens
            self.test_tokens = all_tokens
            self.test_char_freq = char_freq
        
        print(f"✓ {dataset_type.capitalize()} vocabulary size: {len(tokens)}")
        print(f"✓ Total {dataset_type} tokens: {len(all_tokens)}")
        print(f"✓ Unique characters in {dataset_type}: {len(char_freq)}")
        
        return tokens
    
    def analyze_vocabulary_mismatch(self) -> Dict:
        """
        analyze differences between training and test vocabularies
        """
        print("\n" + "="*60)
        print("VOCABULARY MISMATCH ANALYSIS")
        print("="*60)
        
        # calculate set differences
        train_only = self.train_vocab - self.test_vocab
        test_only = self.test_vocab - self.train_vocab
        common = self.train_vocab & self.test_vocab
        
        # basic statistics
        analysis = {
            'train_vocab_size': len(self.train_vocab),
            'test_vocab_size': len(self.test_vocab),
            'common_vocab_size': len(common),
            'train_only_count': len(train_only),
            'test_only_count': len(test_only),
            'train_only_tokens': sorted(list(train_only)),
            'test_only_tokens': sorted(list(test_only)),
            'vocabulary_overlap_ratio': len(common) / len(self.train_vocab | self.test_vocab),
            'test_coverage_by_train': len(common) / len(self.test_vocab) if self.test_vocab else 0
        }
        
        print(f"Train vocabulary size: {analysis['train_vocab_size']}")
        print(f"Test vocabulary size: {analysis['test_vocab_size']}")
        print(f"Common tokens: {analysis['common_vocab_size']}")
        print(f"Tokens only in training: {analysis['train_only_count']}")
        print(f"Tokens only in test: {analysis['test_only_count']}")
        print(f"Vocabulary overlap ratio: {analysis['vocabulary_overlap_ratio']:.4f}")
        print(f"Test coverage by training: {analysis['test_coverage_by_train']:.4f}")
        
        return analysis
    
    def analyze_character_differences(self) -> Dict:
        """
        analyze character-level differences between datasets
        """
        print("\n" + "="*60)
        print("CHARACTER-LEVEL ANALYSIS")
        print("="*60)
        
        # get all characters from each dataset
        train_chars = set(self.train_char_freq.keys())
        test_chars = set(self.test_char_freq.keys())
        
        # calculate differences
        train_only_chars = train_chars - test_chars
        test_only_chars = test_chars - train_chars
        common_chars = train_chars & test_chars
        
        char_analysis = {
            'train_unique_chars': len(train_chars),
            'test_unique_chars': len(test_chars),
            'common_chars': len(common_chars),
            'train_only_chars': sorted(list(train_only_chars)),
            'test_only_chars': sorted(list(test_only_chars)),
            'char_overlap_ratio': len(common_chars) / len(train_chars | test_chars),
            'test_char_coverage': len(common_chars) / len(test_chars) if test_chars else 0
        }
        
        print(f"Train unique characters: {char_analysis['train_unique_chars']}")
        print(f"Test unique characters: {char_analysis['test_unique_chars']}")
        print(f"Common characters: {char_analysis['common_chars']}")
        print(f"Characters only in training: {len(train_only_chars)}")
        print(f"Characters only in test: {len(test_only_chars)}")
        print(f"Character overlap ratio: {char_analysis['char_overlap_ratio']:.4f}")
        print(f"Test character coverage: {char_analysis['test_char_coverage']:.4f}")
        
        return char_analysis
    
    def print_detailed_differences(self, analysis: Dict, char_analysis: Dict):
        """
        print detailed information about the differences
        """
        print("\n" + "="*60)
        print("DETAILED MISMATCH REPORT")
        print("="*60)
        
        # token-level differences
        if analysis['train_only_tokens']:
            print(f"\nTOKENS ONLY IN TRAINING ({len(analysis['train_only_tokens'])}):")
            for i, token in enumerate(analysis['train_only_tokens']):
                print(f"  {i+1:3d}. '{token}' (appears {self.train_tokens.count(token)} times)")
                if i >= 19:  # show only first 20
                    print(f"  ... and {len(analysis['train_only_tokens']) - 20} more")
                    break
        
        if analysis['test_only_tokens']:
            print(f"\nTOKENS ONLY IN TEST ({len(analysis['test_only_tokens'])}):")
            for i, token in enumerate(analysis['test_only_tokens']):
                print(f"  {i+1:3d}. '{token}' (appears {self.test_tokens.count(token)} times)")
                if i >= 19:  # show only first 20
                    print(f"  ... and {len(analysis['test_only_tokens']) - 20} more")
                    break
        
        # character-level differences
        if char_analysis['train_only_chars']:
            print(f"\nCHARACTERS ONLY IN TRAINING ({len(char_analysis['train_only_chars'])}):")
            for char in char_analysis['train_only_chars']:
                freq = self.train_char_freq[char]
                char_repr = repr(char) if not char.isprintable() or char.isspace() else f"'{char}'"
                print(f"  {char_repr} (frequency: {freq})")
        
        if char_analysis['test_only_chars']:
            print(f"\nCHARACTERS ONLY IN TEST ({len(char_analysis['test_only_chars'])}):")
            for char in char_analysis['test_only_chars']:
                freq = self.test_char_freq[char]
                char_repr = repr(char) if not char.isprintable() or char.isspace() else f"'{char}'"
                print(f"  {char_repr} (frequency: {freq})")
    
    def analyze_problematic_tokens(self, analysis: Dict) -> List[str]:
        """
        identify tokens that might cause the most problems during inference
        """
        print("\n" + "="*60)
        print("PROBLEMATIC TOKENS ANALYSIS")
        print("="*60)
        
        problematic = []
        
        # tokens in test but not in training are problematic
        if analysis['test_only_tokens']:
            test_token_freq = Counter(self.test_tokens)
            most_frequent_missing = sorted(
                analysis['test_only_tokens'], 
                key=lambda x: test_token_freq[x], 
                reverse=True
            )
            
            print("MOST PROBLEMATIC: Tokens in test but not in training (sorted by frequency):")
            for i, token in enumerate(most_frequent_missing[:10]):
                freq = test_token_freq[token]
                print(f"  {i+1:2d}. '{token}' (appears {freq} times in test)")
                problematic.append(token)
            
            if len(most_frequent_missing) > 10:
                print(f"  ... and {len(most_frequent_missing) - 10} more tokens")
        
        return problematic
    
    def generate_vocabulary_mapping(self) -> Dict:
        """
        generate mapping information for vocabulary alignment
        """
        # create sorted token-to-index mapping like in training
        train_sorted_tokens = sorted(list(self.train_vocab))
        test_sorted_tokens = sorted(list(self.test_vocab))
        
        train_token_to_idx = {token: idx for idx, token in enumerate(train_sorted_tokens)}
        test_token_to_idx = {token: idx for idx, token in enumerate(test_sorted_tokens)}
        
        mapping_info = {
            'train_token_to_idx': train_token_to_idx,
            'test_token_to_idx': test_token_to_idx,
            'train_idx_to_token': {idx: token for token, idx in train_token_to_idx.items()},
            'test_idx_to_token': {idx: token for token, idx in test_token_to_idx.items()},
            'train_vocab_size': len(train_sorted_tokens),
            'test_vocab_size': len(test_sorted_tokens)
        }
        
        return mapping_info
    
    def save_analysis_report(self, output_file: str, analysis: Dict, char_analysis: Dict, 
                           mapping_info: Dict, problematic_tokens: List[str]):
        """
        save comprehensive analysis report to json file
        """
        report = {
            'summary': {
                'train_vocab_size': analysis['train_vocab_size'],
                'test_vocab_size': analysis['test_vocab_size'],
                'vocabulary_size_difference': analysis['test_vocab_size'] - analysis['train_vocab_size'],
                'tokens_only_in_train': analysis['train_only_count'],
                'tokens_only_in_test': analysis['test_only_count'],
                'vocabulary_overlap_ratio': analysis['vocabulary_overlap_ratio'],
                'test_coverage_by_train': analysis['test_coverage_by_train']
            },
            'token_analysis': {
                'train_only_tokens': analysis['train_only_tokens'],
                'test_only_tokens': analysis['test_only_tokens'],
                'most_problematic_tokens': problematic_tokens
            },
            'character_analysis': {
                'train_unique_chars': char_analysis['train_unique_chars'],
                'test_unique_chars': char_analysis['test_unique_chars'],
                'train_only_chars': char_analysis['train_only_chars'],
                'test_only_chars': char_analysis['test_only_chars'],
                'char_overlap_ratio': char_analysis['char_overlap_ratio'],
                'test_char_coverage': char_analysis['test_char_coverage']
            },
            'vocabulary_mappings': mapping_info,
            'recommendations': self._generate_recommendations(analysis, char_analysis)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nComprehensive analysis saved to: {output_file}")
    
    def _generate_recommendations(self, analysis: Dict, char_analysis: Dict) -> List[str]:
        """
        generate recommendations based on the analysis
        """
        recommendations = []
        
        if analysis['test_only_count'] > 0:
            recommendations.append(
                f"WARNING: {analysis['test_only_count']} tokens in test set are not in training vocabulary. "
                "These will be mapped to <UNK> during inference, potentially reducing accuracy."
            )
        
        if analysis['test_coverage_by_train'] < 0.95:
            recommendations.append(
                f"WARNING: Training vocabulary only covers {analysis['test_coverage_by_train']:.1%} of test vocabulary. "
                "Consider expanding training data or preprocessing test data."
            )
        
        if char_analysis['test_char_coverage'] < 0.98:
            recommendations.append(
                f"WARNING: Training data only covers {char_analysis['test_char_coverage']:.1%} of characters in test set. "
                "Some characters in test will be completely unknown."
            )
        
        if analysis['vocabulary_overlap_ratio'] < 0.9:
            recommendations.append(
                "WARNING: Low vocabulary overlap between train and test. Consider data augmentation or "
                "ensuring train and test come from similar domains."
            )
        
        if not recommendations:
            recommendations.append("Vocabulary alignment looks good! No major issues detected.")
        
        return recommendations


def main():
    """main function to run vocabulary analysis"""
    
    # configuration - modify these paths as needed
    TRAIN_JSON = ""
    TEST_JSON = ""
    OUTPUT_FILE = ""
    VERBOSE = True
    
    # validate input files
    if not Path(TRAIN_JSON).exists():
        print(f"Training JSON file not found: {TRAIN_JSON}")
        return
    
    if not Path(TEST_JSON).exists():
        print(f"Test JSON file not found: {TEST_JSON}")
        return
    
    # initialize analyzer
    analyzer = VocabularyAnalyzer()
    
    # build vocabularies
    print("Analyzing vocabulary mismatch between datasets...")
    analyzer.build_vocabulary_from_json(TRAIN_JSON, "train")
    analyzer.build_vocabulary_from_json(TEST_JSON, "test")
    
    # perform analysis
    analysis = analyzer.analyze_vocabulary_mismatch()
    char_analysis = analyzer.analyze_character_differences()
    
    # print detailed differences
    if VERBOSE:
        analyzer.print_detailed_differences(analysis, char_analysis)
    
    # analyze problematic tokens
    problematic_tokens = analyzer.analyze_problematic_tokens(analysis)
    
    # generate vocabulary mappings
    mapping_info = analyzer.generate_vocabulary_mapping()
    
    # save detailed report
    analyzer.save_analysis_report(OUTPUT_FILE, analysis, char_analysis, mapping_info, problematic_tokens)
    
    # print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Vocabulary size difference: {analysis['test_vocab_size'] - analysis['train_vocab_size']}")
    print(f"Tokens only in training: {analysis['train_only_count']}")
    print(f"Tokens only in test: {analysis['test_only_count']}")
    print(f"Most problematic tokens: {len(problematic_tokens)}")
    
    if analysis['test_only_count'] > 0:
        print(f"\nATTENTION: {analysis['test_only_count']} tokens in test set will be mapped to <UNK>")
        print("This may significantly impact inference accuracy!")
    else:
        print("\nAll test tokens are covered by training vocabulary!")


if __name__ == "__main__":
    main()