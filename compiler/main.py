"""
main.py - Main script to evaluate a design using a custom architecture. 
It uses the golden model of hardware design to run the pre-trained model.
Created by Sayat A. at TEXER.AI
"""
import numpy as np
import torch
from torchvision import datasets, transforms

from compile import generate_assembly
from model import create_mlp_model
from golden_model import execute_program
from dram import save_all_initializers_to_dram, save_input_to_dram, save_dram_to_file, read_from_dram
from helper_functions import quantize_tensor_f32_int8
from accelerator_config import AcceleratorConfig

def evaluate_design(seed, torch_input, label):
    
    torch.manual_seed(seed)
    # 1. Save dummy input to DRAM
    # Use proper quantization for input to preserve information
    input_numpy = torch_input.numpy().squeeze()
    scale = np.max(np.abs(input_numpy)) / 127 if np.max(np.abs(input_numpy)) > 0 else 1.0
    dummy_input = quantize_tensor_f32_int8(input_numpy, scale).flatten()
    
    save_input_to_dram(dummy_input, dram_offsets["inputs"])
    written_input = read_from_dram(dram_offsets["inputs"], len(dummy_input))
    if not np.array_equal(dummy_input, written_input):
        print("The length of the input tensor is", len(dummy_input))
        print("The input data is: ", dummy_input)
        print("The written input data is: ", written_input)
        raise ValueError("Input data mismatch after writing to DRAM")

    # 2. Save DRAM to hex file
    save_dram_to_file("dram.hex")
    # print("DRAM saved to dram.hex")
    

    # # 3. Optional: print the ordered weights and biases
    # print_weights_in_order(model_path)

    # 4. Get the output of the design
    output_design = execute_program("dram.hex")
    max_index = np.argmax(output_design)
    # print("Output from the design:", output_design)
    # print("Expected label:", label)
    # print("Max index from the design:", max_index)
    return max_index == label.item()


if __name__ == "__main__":
    # Model initialization:
    # 1. Create and save the model
    create_mlp_model()
    model_path = "mlp_model.onnx"

    # 2. DRAM configuration

    dram_offsets = {
        "inputs":       AcceleratorConfig.DRAM_ADDR_INPUTS,
        "biases":       AcceleratorConfig.DRAM_ADDR_BIASES,
        "outputs":      AcceleratorConfig.DRAM_ADDR_OUTPUTS,
        "weights":      AcceleratorConfig.DRAM_ADDR_WEIGHTS,
        "conv_weights": AcceleratorConfig.DRAM_ADDR_CONV_WEIGHTS,
    }

    # 3. Save weights/biases to DRAM (the walker is the single source of truth
    #    for initializer DRAM layout — its returned maps drive step 4).
    weight_map, bias_map, conv_weight_map = save_all_initializers_to_dram(
        model_path, dram_offsets)

    # 4. Generate assembly, looking up initializer addresses in the maps.
    generate_assembly(model_path, "model_assembly.asm",
                      weight_map, bias_map, conv_weight_map)
    from assembler import assemble_file
    assemble_file("model_assembly.asm")

    sum = 0
    total_elements = 0
    # Getting the test data:
    # 1. Define transformations (e.g., convert to tensor and normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL image to Tensor [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and std from MNIST dataset
    ])

    # 2. Load the training and test datasets
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 3. Wrap them in DataLoader for batching and shuffling
    test_images = torch.stack([img for img, _ in test_dataset])  # shape: [10000, 1, 28, 28]
    test_labels = torch.tensor([label for _, label in test_dataset])  # shape: [10000]

    # Testing
    for i in range(len(test_labels)):
        output_design = evaluate_design(i, test_images[i], test_labels[i])
        sum += output_design
        if i % 10 == 0:
            print(f"{i+1} runs completed, current accuracy: {sum / (i + 1) * 100}%")

    accuracy = sum / len(test_labels) * 100  # Convert to percentage
    print("Average accuracy over all runs:", accuracy)