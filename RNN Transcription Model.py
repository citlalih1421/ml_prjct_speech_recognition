import torch
import torch.nn as nn
from torch.nn import functional as F

class CTCTranscriptionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, bidirectional=True):
        super(CTCTranscriptionModel, self).__init__()
        
        # Choose LSTM or GRU here (replace 'LSTM' with 'GRU' if desired)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=bidirectional)
        
        # Fully connected layer to output probabilities for each class
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)

    def forward(self, x):
        # Pass input through RNN
        output, _ = self.rnn(x)
        
        # Apply fully connected layer and softmax activation
        output = self.fc(output)  # Output shape: (batch_size, seq_len, num_classes)
        output = F.log_softmax(output, dim=2)  # Log-softmax for CTC compatibility
        
        # Transpose for CTC: (seq_len, batch_size, num_classes)
        return output.transpose(0, 1)

    def compute_ctc_loss(self, logits, targets, input_lengths, target_lengths):
        """
        Computes the CTC loss given the model's logits and target sequences.
        
        Args:
        - logits (torch.Tensor): Model output logits (seq_len, batch_size, num_classes).
        - targets (torch.Tensor): Concatenated target sequences.
        - input_lengths (torch.Tensor): Lengths of the input sequences.
        - target_lengths (torch.Tensor): Lengths of each target sequence.
        
        Returns:
        - loss (torch.Tensor): Computed CTC loss.
        """
        ctc_loss = nn.CTCLoss(blank=0)
        return ctc_loss(logits, targets, input_lengths, target_lengths)
    
    def transcribe(self, logits):
        """
        Decodes the logits to a sequence of classes using greedy decoding.
        
        Args:
        - logits (torch.Tensor): Logits from the model (seq_len, batch_size, num_classes).
        
        Returns:
        - transcriptions (List[List[int]]): List of decoded transcriptions for each batch item.
        """
        # Get the index with the highest probability at each timestep
        best_path = torch.argmax(logits, dim=2)  # Shape: (seq_len, batch_size)
        
        transcriptions = []
        for batch in best_path.transpose(0, 1):  # Transpose to iterate over batch dimension
            transcription = []
            prev_class = None
            for current_class in batch:
                if current_class != prev_class and current_class != 0:  # Avoid duplicates and blanks
                    transcription.append(current_class.item())
                prev_class = current_class
            transcriptions.append(transcription)
        
        return transcriptions


if __name__ == "__main__":
    # Model parameters
    input_size = 40  # Feature dimension, e.g., MFCCs
    hidden_size = 128
    num_classes = 29  # Number of output classes, including blank
    num_layers = 2
    batch_size = 2
    seq_len = 50

    # Instantiate the model
    model = CTCTranscriptionModel(input_size, hidden_size, num_classes, num_layers)

    # Generate random input data and example target sequences
    input_data = torch.randn(batch_size, seq_len, input_size)
    targets = torch.tensor([1, 2, 3, 4, 5, 6])  # Example flattened target sequence
    input_lengths = torch.tensor([seq_len, seq_len])  # Both sequences are full length
    target_lengths = torch.tensor([3, 3])  # Target sequence lengths for CTC loss

    # Forward pass
    logits = model(input_data)  # Output shape: (seq_len, batch_size, num_classes)

    # Compute CTC loss
    loss = model.compute_ctc_loss(logits, targets, input_lengths, target_lengths)
    print("CTC Loss:", loss.item())

    # Transcribe (Inference)
    transcriptions = model.transcribe(logits)
    print("Transcriptions:", transcriptions)
