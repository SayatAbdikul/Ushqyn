import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import tqdm

from accelerator_config import AcceleratorConfig
import compile
import assembler
import dram
import golden_model

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 4,  kernel_size=3, stride=1, padding=0, bias=True)
        self.relu    = nn.ReLU()
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2   = nn.Conv2d(4, 8,  kernel_size=3, stride=1, padding=0, bias=True)
        self.fc      = nn.Linear(8 * 5 * 5, AcceleratorConfig.OUT_N, bias=True)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # → [4, 13, 13]
        x = self.pool(self.relu(self.conv2(x)))   # → [8,  5,  5]
        x = x.view(x.size(0), -1)                 # flatten → 200
        return self.fc(x)

def train_cnn(epochs=5):
    device = torch.device("cpu")
    model = SmallCNN().to(device)
    
    # The golden model expects int8 inputs. So we train with inputs in [-128, 127]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255.0 - 128.0).round().clamp(-128, 127))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training SmallCNN...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
                
    torch.save(model.state_dict(), "small_cnn_weights.pth")
    print("Saved weights to small_cnn_weights.pth")
    
    # Export to ONNX
    dummy_input = torch.zeros(1, 1, 28, 28)
    onnx_filename = "small_cnn_model.onnx"
    torch.onnx.export(
        model, dummy_input, onnx_filename,
        input_names=["input"], output_names=["output"],
    )
    print(f"Exported ONNX model to {onnx_filename}")
    return model

def evaluate_models(model, num_test_images=50):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255.0 - 128.0).round().clamp(-128, 127))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 1. Compile ONNX model — DRAM walker first (owns initializer layout),
    #    then assembler consumes its returned maps. Post-P2 the two walks are
    #    unified; the legacy save_conv_weights_to_dram call is no longer needed.
    asm_file = "small_cnn_model.asm"
    hex_file = "small_cnn_model.hex"

    dram_offsets = {
        "weights":      AcceleratorConfig.DRAM_ADDR_WEIGHTS,
        "conv_weights": AcceleratorConfig.DRAM_ADDR_CONV_WEIGHTS,
        "biases":       AcceleratorConfig.DRAM_ADDR_BIASES,
    }
    weight_map, bias_map, conv_weight_map = dram.save_all_initializers_to_dram(
        "small_cnn_model.onnx", dram_offsets)

    compile.generate_assembly("small_cnn_model.onnx", asm_file,
                              weight_map, bias_map, conv_weight_map)
    assembler.assemble_file(asm_file, output_file=hex_file)
    
    # Read the instructions from hex
    instr_words = []
    with open(hex_file, 'r') as f:
        for line in f:
            if line.strip():
                instr_words.append(int(line.strip(), 16))
                
    # Evaluate over PyTorch and Golden Model
    pt_correct = 0
    gm_correct = 0
    total = min(num_test_images, len(test_dataset))
    
    model.eval()
    print(f"\nEvaluating PyTorch vs Golden Model on {total} images...")
    
    for i in tqdm.tqdm(range(total)):
        image, label = test_dataset[i]
        
        # PyTorch Inference
        with torch.no_grad():
            img_batch = image.unsqueeze(0)
            pt_out = model(img_batch).numpy().flatten()
            pt_pred = np.argmax(pt_out)
            if pt_pred == label:
                pt_correct += 1
                
        # Golden Model Inference
        # Reset memory to pristine state (weights loaded, inputs clear)
        golden_model.memory = dram.get_dram()
        golden_model.buffers = {} # clear internal buffers
        
        # Load input image
        flat_input = image.numpy().flatten().astype(np.int8)
        golden_model.memory[AcceleratorConfig.DRAM_ADDR_INPUTS : AcceleratorConfig.DRAM_ADDR_INPUTS + len(flat_input)] = flat_input
        
        # Manual LOAD_V for input — encode via the ISA spec rather than open-coding
        # the bit layout (see compiler/isa_spec.py for the canonical opcode table).
        from isa_spec import encode
        load_v_word = encode("LOAD_V",
                             dest=9,
                             addr=AcceleratorConfig.DRAM_ADDR_INPUTS,
                             length=len(flat_input))
        golden_model.i_decoder(load_v_word)
        
        # Execute Instructions
        for word in instr_words:
            golden_model.i_decoder(word)
            
        # Get result
        # The output of FC layered with GEMV goes to output buffer (usually 1 or 2, based compile.py)
        # We can find it by looking for STORE
        # The last STORE writes outputs to DRAM_ADDR_OUTPUTS
        
        # Wait, the assembly script should have a STORE instruction at the end!
        gm_out_memory = golden_model.memory[AcceleratorConfig.DRAM_ADDR_OUTPUTS : AcceleratorConfig.DRAM_ADDR_OUTPUTS + 10]
        # Memory values might be int8 if they were stored from int8 buffers, wait:
        # GEMV output buffer is int32 natively? No, `golden_model` GEMV output quantizes to int8. 
        # But wait! We need to cast it signed back from unsigned bytes? The `memory` array in get_dram is a bytearray, so it holds [0, 255].
        # We must unpack it as signed 8-bit integers.
        gm_out = np.array(gm_out_memory, dtype=np.int8)
        gm_pred = np.argmax(gm_out)
        
        if gm_pred == label:
            gm_correct += 1

    print(f"\nResults on {total} test images:")
    print(f"PyTorch Accuracy:     {pt_correct/total * 100:.2f}%")
    print(f"Golden Model Accuracy: {gm_correct/total * 100:.2f}%")

if __name__ == "__main__":
    trained_model = train_cnn(epochs=2)
    evaluate_models(trained_model, num_test_images=50)
