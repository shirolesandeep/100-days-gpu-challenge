# mixed_precision_training.py
import torch
import numpy as np
import time

class MixedPrecisionTrainer:
    def __init__(self, use_amp=True):
        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cuda':
            print("Warning: CUDA not available. Tensor Cores won't be used.")
        
        # Check if Tensor Cores are available
        if self.device.type == 'cuda':
            capability = torch.cuda.get_device_capability()
            if capability[0] < 7:
                print(f"Warning: GPU capability {capability} may not support Tensor Cores.")
        
        # Set up amp for mixed precision training
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    def create_model(self, input_dim, hidden_dim, output_dim):
        # Simple neural network with two linear layers
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        ).to(self.device)
        return model
    
    def train_step(self, model, optimizer, inputs, targets, criterion):
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision when enabled
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Optimizer step with unscaling
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def benchmark(self, input_dim=1024, hidden_dim=4096, output_dim=512, batch_size=128, num_iterations=100):
        # Create data
        inputs = torch.randn(batch_size, input_dim, device=self.device)
        targets = torch.randn(batch_size, output_dim, device=self.device)
        
        # Create model, optimizer, and loss function
        model = self.create_model(input_dim, hidden_dim, output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Warmup
        for _ in range(10):
            self.train_step(model, optimizer, inputs, targets, criterion)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_iterations):
            loss = self.train_step(model, optimizer, inputs, targets, criterion)
            
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Results
        elapsed_time = end_time - start_time
        iterations_per_sec = num_iterations / elapsed_time
        
        print(f"Mixed precision mode: {'Enabled' if self.use_amp else 'Disabled'}")
        print(f"Total time for {num_iterations} iterations: {elapsed_time:.4f} seconds")
        print(f"Throughput: {iterations_per_sec:.2f} iterations/second")
        
        return elapsed_time

def compare_precision_modes():
    print("Running benchmark with FP32 precision...")
    trainer_fp32 = MixedPrecisionTrainer(use_amp=False)
    time_fp32 = trainer_fp32.benchmark()
    
    print("\nRunning benchmark with mixed precision (FP16)...")
    trainer_fp16 = MixedPrecisionTrainer(use_amp=True)
    time_fp16 = trainer_fp16.benchmark()
    
    speedup = time_fp32 / time_fp16
    print(f"\nSpeedup with mixed precision: {speedup:.2f}x")

def train_model(epochs=10):
    # Set up the trainer with mixed precision
    trainer = MixedPrecisionTrainer(use_amp=True)
    
    # Data dimensions
    input_dim = 784  # e.g., MNIST flattened
    hidden_dim = 1024
    output_dim = 10
    batch_size = 256
    
    # Create synthetic dataset
    train_inputs = torch.randn(10000, input_dim, device=trainer.device)
    train_targets = torch.randint(0, output_dim, (10000,), device=trainer.device)
    
    # Create model and optimizer
    model = trainer.create_model(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        num_batches = train_inputs.size(0) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_inputs = train_inputs[start_idx:end_idx]
            batch_targets = train_targets[start_idx:end_idx]
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=trainer.use_amp):
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            trainer.scaler.scale(loss).backward()
            trainer.scaler.step(optimizer)
            trainer.scaler.update()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    print("Training complete!")

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit()
    
    # Check if the GPU supports Tensor Cores
    capability = torch.cuda.get_device_capability()
    if capability[0] < 7:
        print(f"Warning: Your GPU has compute capability {capability}.")
        print("Tensor Cores require compute capability 7.0+ (Volta, Turing, Ampere, or newer).")
    else:
        print(f"GPU has compute capability {capability}, which supports Tensor Cores.")
    
    # Compare FP32 vs Mixed Precision (FP16)
    compare_precision_modes()
    
    # Train a model with mixed precision
    print("\nTraining a model with mixed precision:")
    train_model(epochs=3)