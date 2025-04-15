# fp8_matrix_multiplication.py
import torch
import numpy as np
import time
import ctypes
from typing import Tuple, Optional

# Check if transformer-engine is importable (it provides FP8 support)
try:
    import transformer_engine as te
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False
    print("Warning: transformer-engine not found. We'll simulate FP8 using FP16.")

class FP8MatrixMultiplier:
    def __init__(self):
        # Check for CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type != 'cuda':
            print("Warning: CUDA not available, using CPU instead.")
            print("FP8 operations will be simulated.")
        
        # Check CUDA compute capability
        if self.device.type == 'cuda':
            capability = torch.cuda.get_device_capability()
            self.hopper_or_newer = capability[0] >= 9
            if not self.hopper_or_newer:
                print(f"Warning: GPU compute capability is {capability}.")
                print("Native FP8 support requires Hopper (9.0) or newer architecture.")
                print("FP8 will be simulated using FP16.")

    def simulate_fp8_quantization(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simulate FP8 quantization by quantizing and dequantizing to 8 bits.
        This is a rough approximation of E4M3 dynamic range.
        """
        # FP8 E4M3 has range approximately [-448, 448] with precision of 2^-7
        scale = torch.max(torch.abs(x)) / 240.0
        quant_x = torch.round(x / scale * 127) / 127 * scale
        return quant_x

    def actual_fp8_matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Perform matrix multiplication using actual FP8 if available through transformer-engine.
        This uses the transformer-engine library's FP8 implementation.
        """
        if not TRANSFORMER_ENGINE_AVAILABLE:
            raise RuntimeError("transformer-engine not available for actual FP8 computation")
        
        # Convert to fp8 format and compute
        a_fp8 = te.fp8_autocast.cast_to_fp8(a, te.DType.kFloat8E4M3)
        b_fp8 = te.fp8_autocast.cast_to_fp8(b, te.DType.kFloat8E4M3)
        
        with te.fp8_autocast():
            result = torch.matmul(a_fp8, b_fp8)
        
        return result

    def simulated_fp8_matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Simulate FP8 matrix multiplication by quantizing inputs to FP8,
        performing the computation in FP16/FP32, and then quantizing the result.
        """
        a_fp8 = self.simulate_fp8_quantization(a)
        b_fp8 = self.simulate_fp8_quantization(b)
        result = torch.matmul(a_fp8, b_fp8)
        return result

    def matmul(self, a: torch.Tensor, b: torch.Tensor, force_simulation: bool = False) -> torch.Tensor:
        """
        Perform matrix multiplication using FP8 precision.
        Uses actual FP8 if available, otherwise simulates it.
        
        Args:
            a: First input tensor
            b: Second input tensor
            force_simulation: If True, forces simulation even if hardware FP8 is available
            
        Returns:
            Result of matrix multiplication
        """
        a = a.to(self.device)
        b = b.to(self.device)
        
        # Determine if we can use actual FP8
        use_actual_fp8 = (
            not force_simulation and
            TRANSFORMER_ENGINE_AVAILABLE and
            self.device.type == 'cuda' and
            self.hopper_or_newer
        )
        
        if use_actual_fp8:
            return self.actual_fp8_matmul(a, b)
        else:
            return self.simulated_fp8_matmul(a, b)

    def benchmark(self, 
                  a_shape: Tuple[int, int], 
                  b_shape: Tuple[int, int], 
                  runs: int = 100, 
                  warmup: int = 10) -> dict:
        """
        Benchmark FP8 matrix multiplication against FP16 and FP32.
        
        Args:
            a_shape: Shape of first matrix
            b_shape: Shape of second matrix
            runs: Number of benchmark runs
            warmup: Number of warmup runs
            
        Returns:
            Dictionary containing benchmark results
        """
        assert a_shape[1] == b_shape[0], "Matrix dimensions must be compatible for multiplication"
        
        # Create random matrices
        a_fp32 = torch.randn(a_shape, device=self.device)
        b_fp32 = torch.randn(b_shape, device=self.device)
        
        # Convert to different precisions
        a_fp16 = a_fp32.half()
        b_fp16 = b_fp32.half()
        
        # Warmup
        for _ in range(warmup):
            # FP32
            _ = torch.matmul(a_fp32, b_fp32)
            # FP16
            _ = torch.matmul(a_fp16, b_fp16)
            # FP8 (simulated)
            _ = self.matmul(a_fp32, b_fp32, force_simulation=True)
            # FP8 (actual if available)
            if TRANSFORMER_ENGINE_AVAILABLE and self.device.type == 'cuda' and self.hopper_or_newer:
                _ = self.matmul(a_fp32, b_fp32, force_simulation=False)
        
        # Synchronize before timing
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark FP32
        fp32_times = []
        for _ in range(runs):
            start = time.time()
            _ = torch.matmul(a_fp32, b_fp32)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            fp32_times.append(time.time() - start)
        
        # Benchmark FP16
        fp16_times = []
        for _ in range(runs):
            start = time.time()
            _ = torch.matmul(a_fp16, b_fp16)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            fp16_times.append(time.time() - start)
        
        # Benchmark simulated FP8
        sim_fp8_times = []
        for _ in range(runs):
            start = time.time()
            _ = self.matmul(a_fp32, b_fp32, force_simulation=True)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            sim_fp8_times.append(time.time() - start)
        
        # Benchmark actual FP8 if available
        actual_fp8_times = []
        if TRANSFORMER_ENGINE_AVAILABLE and self.device.type == 'cuda' and self.hopper_or_newer:
            for _ in range(runs):
                start = time.time()
                _ = self.matmul(a_fp32, b_fp32, force_simulation=False)
                torch.cuda.synchronize()
                actual_fp8_times.append(time.time() - start)
        
        # Calculate accuracy loss compared to FP32
        result_fp32 = torch.matmul(a_fp32, b_fp32)
        result_fp16 = torch.matmul(a_fp16, b_fp16).float()
        result_sim_fp8 = self.matmul(a_fp32, b_fp32, force_simulation=True)
        
        rel_error_fp16 = torch.norm(result_fp32 - result_fp16) / torch.norm(result_fp32)
        rel_error_sim_fp8 = torch.norm(result_fp32 - result_sim_fp8) / torch.norm(result_fp32)
        
        results = {
            "matrix_sizes": f"{a_shape} × {b_shape}",
            "fp32": {
                "avg_time": np.mean(fp32_times),
                "min_time": np.min(fp32_times),
                "rel_error": 0.0
            },
            "fp16": {
                "avg_time": np.mean(fp16_times),
                "min_time": np.min(fp16_times),
                "rel_error": rel_error_fp16.item(),
                "speedup": np.mean(fp32_times) / np.mean(fp16_times)
            },
            "simulated_fp8": {
                "avg_time": np.mean(sim_fp8_times),
                "min_time": np.min(sim_fp8_times),
                "rel_error": rel_error_sim_fp8.item(),
                "speedup": np.mean(fp32_times) / np.mean(sim_fp8_times)
            }
        }
        
        # Add actual FP8 results if available
        if TRANSFORMER_ENGINE_AVAILABLE and self.device.type == 'cuda' and self.hopper_or_newer:
            result_actual_fp8 = self.matmul(a_fp32, b_fp32, force_simulation=False)
            rel_error_actual_fp8 = torch.norm(result_fp32 - result_actual_fp8) / torch.norm(result_fp32)
            
            results["actual_fp8"] = {
                "avg_time": np.mean(actual_fp8_times),
                "min_time": np.min(actual_fp8_times),
                "rel_error": rel_error_actual_fp8.item(),
                "speedup": np.mean(fp32_times) / np.mean(actual_fp8_times)
            }
        
        return results

def main():
    """
    Run sample benchmarks to demonstrate FP8 matrix multiplication performance.
    """
    fp8_mm = FP8MatrixMultiplier()
    
    # Test with different matrix sizes
    matrix_sizes = [
        ((128, 128), (128, 128)),
        ((512, 512), (512, 512)),
        ((1024, 1024), (1024, 1024)),
        ((2048, 2048), (2048, 2048)),
        ((1024, 2048), (2048, 1024)),
        ((4096, 1024), (1024, 4096))
    ]
    
    results = []
    for a_shape, b_shape in matrix_sizes:
        print(f"\nBenchmarking {a_shape} × {b_shape}...")
        result = fp8_mm.benchmark(a_shape, b_shape, runs=10, warmup=2)
        results.append(result)
        print(f"FP32 avg time: {result['fp32']['avg_time']:.6f}s")
        print(f"FP16 avg time: {result['fp16']['avg_time']:.6f}s, speedup: {result['fp16']['speedup']:.2f}x, rel error: {result['fp16']['rel_error']:.6f}")
        print(f"Simulated FP8 avg time: {result['simulated_fp8']['avg_time']:.6f}s, speedup: {result['simulated_fp8']['speedup']:.2f}x, rel error: {result['simulated_fp8']['rel_error']:.6f}")
        
        if "actual_fp8" in result:
            print(f"Actual FP8 avg time: {result['actual_fp8']['avg_time']:.6f}s, speedup: {result['actual_fp8']['speedup']:.2f}x, rel error: {result['actual_fp8']['rel_error']:.6f}")
    
    return results

if __name__ == "__main__":
    main()
