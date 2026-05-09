#!/usr/bin/env python3
"""
generate_dram_for_input.py - Generate complete DRAM state for the N-th MNIST test image

This script creates a dram.hex file containing:
1. Instructions (machine code for MLP inference)
2. Weights and biases (quantized from pre-trained model)
3. Input data (the N-th quantized MNIST test image)

Usage:
    python generate_dram_for_input.py --input-index 42
    python generate_dram_for_input.py --input-index 42 --output dram_42.hex
    python generate_dram_for_input.py --input-index 42 --show-image
"""

import argparse
import numpy as np
import torch
from torchvision import datasets, transforms

from model import create_mlp_model
from compile import generate_assembly
from assembler import assemble_file
from dram import save_all_initializers_to_dram, save_input_to_dram, save_dram_to_file, read_from_dram
from helper_functions import quantize_tensor_f32_int8
from accelerator_config import AcceleratorConfig


def generate_dram_for_input(
    input_index: int,
    output_file: str = "dram.hex",
    model_weights: str = "digit_model_weights.pth",
    show_image: bool = False
):
    """
    Generates complete DRAM state for the N-th MNIST test image.
    
    Args:
        input_index: Index of the MNIST test image (0 to 9999)
        output_file: Output hex file path
        model_weights: Path to pre-trained PyTorch weights
        show_image: If True, display the input image using matplotlib
    
    Returns:
        Tuple of (expected_label, quantized_input)
    """
    print(f"=" * 60)
    print(f"Generating DRAM for input #{input_index}")
    print(f"=" * 60)
    
    # Step 1: Create and export ONNX model
    print("[1/5] Creating ONNX model...")
    create_mlp_model()
    model_path = "mlp_model.onnx"
    
    # Step 2: DRAM configuration
    dram_offsets = {
        "inputs":       AcceleratorConfig.DRAM_ADDR_INPUTS,
        "biases":       AcceleratorConfig.DRAM_ADDR_BIASES,
        "outputs":      AcceleratorConfig.DRAM_ADDR_OUTPUTS,
        "weights":      AcceleratorConfig.DRAM_ADDR_WEIGHTS,
        "conv_weights": AcceleratorConfig.DRAM_ADDR_CONV_WEIGHTS,
    }

    # Step 3: Save weights/biases to DRAM (the walker owns layout; its returned
    # maps drive the assembler in step 4).
    print("[2/5] Writing weights and biases to DRAM...")
    weight_map, bias_map, conv_weight_map = save_all_initializers_to_dram(
        model_path, dram_offsets)

    # Step 4: Generate and assemble instructions
    print("[3/5] Generating and assembling instructions...")
    generate_assembly(model_path, "model_assembly.asm",
                      weight_map, bias_map, conv_weight_map)
    assemble_file("model_assembly.asm")
    
    # Step 5: Load the N-th MNIST test image
    print(f"[4/5] Loading MNIST test image #{input_index}...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    if input_index < 0 or input_index >= len(test_dataset):
        raise ValueError(f"Input index must be between 0 and {len(test_dataset)-1}")
    
    image, label = test_dataset[input_index]
    
    # Quantize input
    input_numpy = image.numpy().squeeze()
    scale = np.max(np.abs(input_numpy)) / 127 if np.max(np.abs(input_numpy)) > 0 else 1.0
    quantized_input = quantize_tensor_f32_int8(input_numpy, scale).flatten()
    
    # Write to DRAM
    save_input_to_dram(quantized_input, dram_offsets["inputs"])
    
    # Verify write
    written_input = read_from_dram(dram_offsets["inputs"], len(quantized_input))
    if not np.array_equal(quantized_input, written_input):
        raise ValueError("Input data mismatch after writing to DRAM")
    
    # Step 6: Export DRAM to hex file
    print(f"[5/5] Exporting DRAM to {output_file}...")
    save_dram_to_file(output_file)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"✅ Generated {output_file}")
    print(f"   Input index: {input_index}")
    print(f"   Expected label: {label}")
    print(f"   Input size: {len(quantized_input)} bytes")
    print(f"   Input address: 0x{dram_offsets['inputs']:04X}")
    print(f"{'=' * 60}")
    
    # Optionally show the image
    if show_image:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4, 4))
            plt.imshow(image.squeeze().numpy(), cmap='gray')
            plt.title(f"Input #{input_index}, Label: {label}")
            plt.axis('off')
            plt.show()
        except ImportError:
            print("Note: matplotlib not installed, skipping image display")
    
    return label, quantized_input


def main():
    parser = argparse.ArgumentParser(
        description="Generate DRAM state for N-th MNIST test image"
    )
    parser.add_argument(
        "--input-index", "-n",
        type=int,
        required=True,
        help="Index of MNIST test image (0-9999)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dram.hex",
        help="Output hex file path (default: dram.hex)"
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Display the input image using matplotlib"
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default="digit_model_weights.pth",
        help="Path to pre-trained model weights"
    )
    
    args = parser.parse_args()
    
    generate_dram_for_input(
        input_index=args.input_index,
        output_file=args.output,
        model_weights=args.model_weights,
        show_image=args.show_image
    )


if __name__ == "__main__":
    main()
