import torch
import torch.nn as nn
import logging

logger = logging.getLogger("ULBiTPBaseline")


class LabelSmoothedCTCLoss(nn.Module):
    """
    A custom loss function that applies label smoothing to the CTC loss.
    
    Label smoothing encourages the model to be less confident in its predictions,
    which can improve generalization and prevent overfitting. It achieves this by
    mixing the model's predicted probabilities with a uniform distribution.
    """
    def __init__(self, num_classes: int, blank: int, alpha: float = 0.1, reduction: str = 'mean', zero_infinity: bool = True):
        """
        Args:
            num_classes (int): The number of classes (including the blank token).
            blank (int): The index of the blank label.
            alpha (float, optional): The label smoothing factor (e.g., 0.1). Defaults to 0.1.
            reduction (str, optional): Specifies the reduction to apply to the output:
                                       'none' | 'mean' | 'sum'. Defaults to 'mean'.
            zero_infinity (bool, optional): Whether to zero infinite losses and grad. Defaults to True.
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        
        # Instantiate the standard CTC loss, which we will use internally
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculates the label-smoothed CTC loss.
        
        Args:
            log_probs (torch.Tensor): Tensor of shape (T, N, C) from the model (log-probabilities).
            targets (torch.Tensor): Tensor of shape (S) or (N, S_max) representing the target sequences.
            input_lengths (torch.Tensor): Tensor of shape (N) representing the lengths of the inputs.
            target_lengths (torch.Tensor): Tensor of shape (N) representing the lengths of the targets.
            
        Returns:
            torch.Tensor: The calculated loss.
        """
        if self.alpha == 0:
            # If no smoothing, just use the standard CTC loss
            return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # The core of label smoothing: mix the model's log-probabilities with a uniform distribution.
        # Use logaddexp for numerical stability: log( (1-alpha)*p + alpha*u ) 
        # where u is the uniform probability 1/num_classes
        
        # Calculate the log of a uniform distribution over all classes
        log_uniform = torch.full_like(log_probs, -float(torch.log(torch.tensor(self.num_classes))))
        
        # Combine the original log-probabilities with the uniform distribution
        smoothed_log_probs = torch.logaddexp(
            log_probs + torch.log(torch.tensor(1.0 - self.alpha)),
            log_uniform + torch.log(torch.tensor(self.alpha))
        )

        # Pass the smoothed log-probabilities to the standard CTC loss function
        return self.ctc_loss(smoothed_log_probs, targets, input_lengths, target_lengths)

    def update_alpha(self, new_alpha: float):
        """Updates the label smoothing factor (alpha) for annealing."""
        if self.alpha != new_alpha:
            logger.info(f"Label smoothing alpha updated from {self.alpha:.2f} to {new_alpha:.2f}")
            self.alpha = new_alpha


