# Utils/metrics.py

import torch
import editdistance  # for calculating edit distance (Levenshtein) for WER and CER
import time  # for calculating RTF (Real-Time Factor)
from difflib import SequenceMatcher


# --- New Class-Based Structure ---

class BaseCalculator:
    """
    Base class for metric calculators that use edit distance on sequences.
    This is suitable for WER, PER, VER where sequences are split into tokens.
    """
    def _calculate_from_tokens(self, predictions: list[list[str]], ground_truths: list[list[str]]) -> float:
        """
        Calculates error rate from pre-tokenized lists.
        """
        total_distance = 0
        total_items = 0

        for pred_tokens, gt_tokens in zip(predictions, ground_truths):
            distance = editdistance.eval(pred_tokens, gt_tokens)
            total_distance += distance
            total_items += len(gt_tokens)

        if total_items == 0:
            return 0.0 if total_distance == 0 else 1.0
        
        return total_distance / total_items

class WERCalculator(BaseCalculator):
    """Calculates Word Error Rate (WER)."""

    def _preprocess(self, sequences: list[str], ignore_tokens: list[str] = None) -> list[list[str]]:
        """Removes specified tokens and splits into words."""
        if ignore_tokens is None:
            ignore_tokens = ['§']
        processed_sequences = []
        for seq in sequences:
            tokens = seq.split()
            filtered = [t for t in tokens if t not in ignore_tokens]
            processed_sequences.append(filtered)
        return processed_sequences

    def calculate_legacy(self, predictions: list[str], ground_truths: list[str]) -> float:
        pred_words = [p.split() for p in predictions]
        gt_words = [g.split() for g in ground_truths]
        return self._calculate_from_tokens(pred_words, gt_words)

    def calculate_standard(self, predictions: list[str], ground_truths: list[str], ignore_tokens: list[str] = None) -> float:
        if ignore_tokens is None:
            ignore_tokens = ['§']
        pred_words = self._preprocess(predictions, ignore_tokens)
        gt_words = self._preprocess(ground_truths, ignore_tokens)
        return self._calculate_from_tokens(pred_words, gt_words)


class PERCalculator(BaseCalculator):
    """Calculates Phoneme Error Rate (PER)."""

    def _preprocess(self, sequences: list[str], ignore_tokens: list[str] = None) -> list[list[str]]:
        """Removes specified tokens and splits the sequence."""
        if ignore_tokens is None:
            ignore_tokens = ['sil', '<wb>']
        
        processed_sequences = []
        for seq in sequences:
            tokens = seq.split()
            # Filter out the tokens to be ignored
            filtered_tokens = [token for token in tokens if token not in ignore_tokens]
            processed_sequences.append(filtered_tokens)
        return processed_sequences

    def calculate_legacy(self, predictions: list[str], ground_truths: list[str]) -> float:
        """Calculates PER on raw sequences, treating all tokens as valid."""
        if not predictions and not ground_truths:
            return 0.0
        if not predictions or not ground_truths:
            return 1.0

        pred_phonemes = [p.split() for p in predictions]
        gt_phonemes = [g.split() for g in ground_truths]
        
        return self._calculate_from_tokens(pred_phonemes, gt_phonemes)

    def calculate_standard(self, predictions: list[str], ground_truths: list[str], ignore_tokens: list[str] = None) -> float:
        """Calculates PER on sequences after removing the ignore_tokens."""
        if ignore_tokens is None:
            ignore_tokens = ['sil', '<wb>']
        
        if not predictions and not ground_truths:
            return 0.0
        if not predictions or not ground_truths:
            return 1.0
            
        pred_phonemes_processed = self._preprocess(predictions, ignore_tokens)
        gt_phonemes_processed = self._preprocess(ground_truths, ignore_tokens)

        return self._calculate_from_tokens(pred_phonemes_processed, gt_phonemes_processed)


class VERCalculator(BaseCalculator):
    """Calculates Viseme Error Rate (VER)."""

    def _preprocess(self, sequences: list[str], ignore_token: str = "viseme_S") -> list[list[str]]:
        """Removes the ignore_token and splits the sequence into tokens."""
        processed_sequences = []
        for seq in sequences:
            tokens = seq.split()
            # Filter out the token to be ignored
            filtered_tokens = [token for token in tokens if token != ignore_token]
            processed_sequences.append(filtered_tokens)
        return processed_sequences

    def calculate_legacy(self, predictions: list[str], ground_truths: list[str]) -> float:
        """Calculates VER on raw sequences, treating all tokens as valid."""
        if not predictions and not ground_truths:
            return 0.0
        if not predictions or not ground_truths:
            return 1.0

        pred_visemes = [p.split() for p in predictions]
        gt_visemes = [g.split() for g in ground_truths]
        
        return self._calculate_from_tokens(pred_visemes, gt_visemes)

    def calculate_standard(self, predictions: list[str], ground_truths: list[str], ignore_token: str = "viseme_S") -> float:

        if ignore_tokens is None:
            ignore_tokens = ['sil']
            
        """Calculates VER on sequences after removing the ignore_token."""
        if not predictions and not ground_truths:
            return 0.0
        if not predictions or not ground_truths:
            return 1.0
            
        pred_visemes_processed = self._preprocess(predictions, ignore_token)
        gt_visemes_processed = self._preprocess(ground_truths, ignore_token)

        return self._calculate_from_tokens(pred_visemes_processed, gt_visemes_processed)


class CERCalculator:
    """
    Calculates Character Error Rate (CER).
    """

    def calculate_legacy(self, predictions: list[str], ground_truths: list[str]) -> float:
        total_distance = 0
        total_chars = 0
        for pred, gt in zip(predictions, ground_truths):
            total_distance += editdistance.eval(pred, gt)
            total_chars += len(gt)
        return total_distance / total_chars if total_chars > 0 else 0.0

    def calculate_standard(self, predictions: list[str], ground_truths: list[str], ignore_tokens: list[str] = None) -> float:
        if ignore_tokens is None:
            ignore_tokens = ['<wb>', ' ', '§']
        total_distance = 0
        total_chars = 0
        for pred, gt in zip(predictions, ground_truths):
            for tok in ignore_tokens:
                pred = pred.replace(tok, "")
                gt = gt.replace(tok, "")
            total_distance += editdistance.eval(pred, gt)
            total_chars += len(gt)
        return total_distance / total_chars if total_chars > 0 else 0.0
    
# --- Legacy Functions (for backward compatibility) ---

def calculate_cer(predictions: list[str], ground_truths: list[str], standard: bool = True) -> float:
    calculator = CERCalculator()
    if standard:
        return calculator.calculate_standard(predictions, ground_truths)
    else:
        return calculator.calculate_legacy(predictions, ground_truths)


def calculate_wer(predictions: list[str], ground_truths: list[str], standard: bool = True) -> float:
    calculator = WERCalculator()
    if standard:
        return calculator.calculate_standard(predictions, ground_truths)
    else:
        return calculator.calculate_legacy(predictions, ground_truths)


def calculate_per(predictions: list[str], ground_truths: list[str], standard: bool = True) -> float:
    """
    Calculates Phoneme Error Rate (PER).

    Args:
        predictions (list[str]): List of predicted phoneme sequences.
        ground_truths (list[str]): List of ground truth phoneme sequences.
        standard (bool): If True, calculates the standard PER by ignoring silence ('sil')
                         and separator ('<wb>') tokens. If False, calculates the legacy PER
                         where all tokens are considered.

    Returns:
        float: The calculated Phoneme Error Rate.
    """
    calculator = PERCalculator()
    if standard:
        return calculator.calculate_standard(predictions, ground_truths)
    else:
        return calculator.calculate_legacy(predictions, ground_truths)


def calculate_ver(predictions: list[str], ground_truths: list[str], standard: bool = True) -> float:
    """
    Calculates Viseme Error Rate (VER).

    Args:
        predictions (list[str]): List of predicted viseme sequences.
        ground_truths (list[str]): List of ground truth viseme sequences.
        standard (bool): If True, calculates the standard VER by ignoring silence/separator
                         tokens ('viseme_S'). If False, calculates the legacy VER where
                         all tokens are considered.

    Returns:
        float: The calculated Viseme Error Rate.
    """
    calculator = VERCalculator()
    if standard:
        return calculator.calculate_standard(predictions, ground_truths)
    else:
        return calculator.calculate_legacy(predictions, ground_truths)


# --- Other Utility Functions (unchanged for now) ---

def calculate_rtf(processing_time_seconds: float, num_frames: int, frames_per_second: float) -> float:
    """
    فاکتور زمان واقعی (Real-Time Factor - RTF) یک مرحله پردازش را محاسبه می‌کند.
    """
    if frames_per_second <= 0 or num_frames <= 0:
        return 0.0
    real_video_duration_seconds = num_frames / frames_per_second
    if real_video_duration_seconds == 0:
        return 0.0
    rtf = processing_time_seconds / real_video_duration_seconds
    return rtf

def update_confusion_matrix(matrix, predictions, ground_truths, class_list):
    """
    Updates the confusion matrix using Python's built-in difflib.
    """
    class_to_idx = {cls: i for i, cls in enumerate(class_list)}
    
    for pred_seq, gt_seq in zip(predictions, ground_truths):
        pred_phonemes = pred_seq.split()
        gt_phonemes = gt_seq.split()
        
        matcher = SequenceMatcher(None, gt_phonemes, pred_phonemes)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for i in range(i1, i2):
                    gt_char = gt_phonemes[i]
                    if gt_char in class_to_idx:
                        matrix[class_to_idx[gt_char], class_to_idx[gt_char]] += 1
            elif tag == 'replace':
                for i in range(i1, i2):
                    for j in range(j1, j2):
                        gt_char = gt_phonemes[i]
                        pred_char = pred_phonemes[j]
                        if gt_char in class_to_idx and pred_char in class_to_idx:
                            matrix[class_to_idx[gt_char], class_to_idx[pred_char]] += 1

def format_confusion_matrix(matrix, class_list, top_k=10, hard_phonemes=None):
    """Formats the confusion matrix, highlights hard phonemes, and returns metrics."""
    if hard_phonemes is None:
        hard_phonemes = []
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)

    off_diagonal = matrix.clone()
    off_diagonal.fill_diagonal_(0)
    
    flat_indices = torch.topk(off_diagonal.flatten(), k=min(top_k, off_diagonal.numel())).indices
    rows = flat_indices // matrix.shape[1]
    cols = flat_indices % matrix.shape[1]
    
    output_str = "\n--- Phoneme Confusion Matrix ---\n"
    output_str += f"Top {top_k} Most Confused Pairs (GT -> Pred):\n"
    confusion_pairs = {}
    for i in range(len(rows)):
        gt_idx, pred_idx = rows[i].item(), cols[i].item()
        count = int(matrix[gt_idx, pred_idx].item())
        if count > 0:
            gt_phoneme = class_list[gt_idx]
            pred_phoneme = class_list[pred_idx]
            output_str += f"  - '{gt_phoneme}' -> '{pred_phoneme}': {count} times\n"
            if gt_phoneme not in confusion_pairs:
                confusion_pairs[gt_phoneme] = pred_phoneme
            
    correct_predictions = matrix.diag()
    total_per_class = matrix.sum(dim=1)
    accuracies = correct_predictions / total_per_class.clamp(min=1)
    
    output_str += "\nPer-Phoneme Accuracy (Correct / Total):\n"
    for i, acc in enumerate(accuracies):
        class_name = class_list[i]
        correct = int(correct_predictions[i].item())
        total = int(total_per_class[i].item())
        if total > 0:
            marker = ">>" if class_name in hard_phonemes else "  -"
            output_str += f"{marker} '{class_name}': {acc:.2%} ({correct}/{total})\n"
            
    return output_str, confusion_pairs

def calculate_targeted_metrics(preds_per: list, gts_per: list, hard_phonemes: list) -> dict:
    """Calculates PER specifically for samples containing hard phonemes."""
    if not hard_phonemes:
        return {}

    hard_samples_preds = []
    hard_samples_gts = []

    for pred_seq, gt_seq in zip(preds_per, gts_per):
        gt_phonemes = gt_seq.split()
        if any(p in hard_phonemes for p in gt_phonemes):
            hard_samples_preds.append(pred_seq)
            hard_samples_gts.append(gt_seq)
    
    if not hard_samples_gts:
        return {"hard_samples_per": 0.0, "hard_samples_count": 0}

    hard_per = calculate_wer(hard_samples_preds, hard_samples_gts)

    return {
        "hard_samples_per": hard_per,
        "hard_samples_count": len(hard_samples_gts)
    }

def correct_with_lexicon(predictions: list[str], lexicon: list[list[str]]) -> list[str]:
    """Corrects a list of predicted sequences using a lexicon."""
    corrected_predictions = []
    
    for pred_seq in predictions:
        pred_phonemes = pred_seq.split()
        best_match = min(lexicon, key=lambda lex_phonemes: editdistance.eval(pred_phonemes, lex_phonemes))
        corrected_predictions.append(" ".join(best_match))
        
    return corrected_predictions


# --- Block for testing the evaluation metrics ---
if __name__ == "__main__":
    print("--- Testing Evaluation Metrics ---")

    # --- Test WER ---
    preds_wer = ["this is a test", "hello world", "speech recognition"]
    gts_wer = ["this is a test", "hello", "speech recognition system"]
    
    wer_score = calculate_wer(preds_wer, gts_wer)
    print(f"WER Score: {wer_score:.4f}")
    # Expected: (0 + 1 + 1) / (4 + 1 + 3) = 2 / 8 = 0.25
    assert abs(wer_score - 0.25) < 0.001, f"WER test failed. Expected ~0.25, got {wer_score}"

    # --- Test CER ---
    preds_cer = ["apple", "banana", "kiwi"]
    gts_cer = ["aple", "bannana", "kiwi"]
    
    cer_score = calculate_cer(preds_cer, gts_cer)
    print(f"CER Score: {cer_score:.4f}")
    # Expected: (1 + 1 + 0) / (4 + 7 + 4) = 2 / 15 = 0.1333...
    assert abs(cer_score - 0.1333) < 0.001, f"CER test failed. Expected ~0.1333, got {cer_score}"

    # --- Test RTF ---
    # (Leaving RTF tests as they were)
    print("\n--- Testing RTF ---")
    processing_time_1 = 1.0
    num_frames_1 = 300
    frames_per_second_1 = 30.0
    rtf_1 = calculate_rtf(processing_time_1, num_frames_1, frames_per_second_1)
    print(f"RTF Scenario 1 (Faster): {rtf_1:.4f}")
    assert abs(rtf_1 - 0.1) < 0.001, "RTF test 1 failed"

    # --- Test VER ---
    print("\n--- Testing VER ---")
    preds_ver = ["viseme_A viseme_B viseme_S viseme_C", "viseme_D viseme_E"]
    gts_ver = ["viseme_A viseme_B viseme_C", "viseme_D viseme_F viseme_E"]

    # Test legacy VER (standard=False)
    ver_legacy = calculate_ver(preds_ver, gts_ver, standard=False)
    print(f"VER Score (Legacy): {ver_legacy:.4f}")
    # Expected: 
    # Seq1: "A B S C" vs "A B C" -> 1 deletion of S.
    # Seq2: "D E" vs "D F E" -> 1 insertion of F.
    # Total distance = 1 + 1 = 2
    # Total visemes in GT = 3 + 3 = 6
    # VER = 2 / 6 = 0.3333
    assert abs(ver_legacy - 0.3333) < 0.001, f"Legacy VER test failed. Expected ~0.3333, got {ver_legacy}"

    # Test standard VER (standard=True)
    ver_standard = calculate_ver(preds_ver, gts_ver, standard=True)
    print(f"VER Score (Standard): {ver_standard:.4f}")
    # Expected:
    # Seq1: "A B C" (after removing S) vs "A B C" -> 0 distance
    # Seq2: "D E" vs "D F E" -> 1 insertion of F
    # Total distance = 0 + 1 = 1
    # Total visemes in GT (after removing S) = 3 + 3 = 6 
    # VER = 1 / 6 = 0.1667
    assert abs(ver_standard - 0.1667) < 0.001, f"Standard VER test failed. Expected ~0.1667, got {ver_standard}"

    # --- Test PER ---
    print("\n--- Testing PER ---")
    preds_per = ["P _ AE AH", "sil K AE T"]
    gts_per = ["P AE", "K AE T S"]

    # Test legacy PER (standard=False)
    per_legacy = calculate_per(preds_per, gts_per, standard=False)
    print(f"PER Score (Legacy): {per_legacy:.4f}")
    # Expected:
    # Seq1: "P _ AE AH" vs "P AE" -> 2 errors (_ and AH)
    # Seq2: "sil K AE T" vs "K AE T S" -> 2 errors (sil and S)
    # Total distance = 2 + 2 = 4
    # Total phonemes in GT = 2 + 4 = 6
    # PER = 4 / 6 = 0.6667
    assert abs(per_legacy - 0.6667) < 0.001, f"Legacy PER test failed. Expected ~0.6667, got {per_legacy}"

    # Test standard PER (standard=True)
    per_standard = calculate_per(preds_per, gts_per, standard=True)
    print(f"PER Score (Standard): {per_standard:.4f}")
    # Expected:
    # Seq1: "P AE AH" vs "P AE" -> 1 error (AH)
    # Seq2: "K AE T" vs "K AE T S" -> 1 error (S)
    # Total distance = 1 + 1 = 2
    # Total phonemes in GT (after removing sil/_) = 2 + 4 = 6
    # PER = 2 / 6 = 0.3333
    assert abs(per_standard - 0.3333) < 0.001, f"Standard PER test failed. Expected ~0.3333, got {per_standard}"

    print("--- All tests passed successfully! ---")
