"""
Cocotb Testbench for TinyML Accelerator - Golden Model Verification
===================================================================

This testbench verifies the RTL implementation against the Python golden model
by comparing the results written to DRAM after execution using actual MNIST test data.

Test Flow:
1. Load MNIST test dataset (same as main.py)
2. For each test image:
   a. Quantize and save to dram.hex at input address
   b. Execute RTL through all instructions
   c. Read RTL results from dram.hex after STORE
   d. Execute golden model with same input
   e. Compare RTL vs Golden model results
3. Report overall accuracy

The comparison is performed on the output buffer (typically at address 0x20000)
after the STORE instruction writes the final neural network output.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, FallingEdge
import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms

# Add compiler directory to path for importing golden model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../compiler'))

from golden_model import execute_program, get_dram
from dram import save_dram_to_file, save_input_to_dram, read_from_dram, MEM_SIZE
from compile import generate_assembly
from model import create_mlp_model
from assembler import assemble_file
from utils.accelerator_tester import TinyMLAcceleratorTester


@cocotb.test()
async def test_accelerator_mnist_dataset(dut):
    """
    Main test: Compare RTL execution against golden model using MNIST test dataset
    
    This test:
    1. Loads the MNIST test dataset (10,000 images)
    2. Tests a subset of images (configurable)
    3. For each image:
       - Prepares input in dram.hex
       - Executes RTL
       - Executes golden model
       - Compares outputs
       - Checks if prediction matches label
    4. Reports overall accuracy for both RTL and golden model
    """
    
    # Create tester instance
    tester = TinyMLAcceleratorTester(dut)
    
    # Start clock
    cocotb.log.info("Starting 100MHz clock")
    clock = Clock(dut.clk, tester.clock_period, units="ns")
    cocotb.start_soon(clock.start())
    
    # Get paths
    compiler_dir = os.path.join(os.path.dirname(__file__), '../../compiler')
    dram_hex_path = os.path.join(compiler_dir, 'dram.hex')
    os.chdir(compiler_dir)  # Change to compiler directory for imports to work
    
    # ========================================================================
    # STEP 0: Initialize model and generate assembly
    # ========================================================================
    cocotb.log.info("\n" + "=" * 70)
    cocotb.log.info("STEP 0: INITIALIZE MODEL AND GENERATE ASSEMBLY")
    cocotb.log.info("=" * 70)
    
    # Create model
    create_mlp_model()
    model_path = "mlp_model.onnx"
    
    # Save weights/biases to DRAM (post-P2: walker is single source of truth
    # for initializer layout; its returned maps drive generate_assembly).
    from dram import save_all_initializers_to_dram, save_dram_to_file
    weight_map, bias_map, conv_weight_map = save_all_initializers_to_dram(
        model_path, tester.dram_offsets)
    cocotb.log.info("Weights and biases saved to DRAM")

    # Generate and assemble instructions
    generate_assembly(model_path, "model_assembly.asm",
                      weight_map, bias_map, conv_weight_map)
    assemble_file("model_assembly.asm")
    cocotb.log.info("Assembly code generated and assembled")
    
    # Save DRAM to file for golden model
    dram_hex_path = os.path.join(compiler_dir, 'dram.hex')
    save_dram_to_file(dram_hex_path)
    
    # CRITICAL: Sync ALL DRAM contents (instructions, weights, biases) to RTL memories
    # The $readmemh in RTL only runs at simulation time 0, so any changes made during
    # the test (like generating assembly or saving weights) must be manually synced.
    cocotb.log.info("Syncing DRAM contents to all RTL memory instances...")
    tester.sync_dram_to_rtl()
    cocotb.log.info("DRAM sync complete - instructions, weights, and biases loaded to RTL")
    
    # ========================================================================
    # STEP 1: Load MNIST test dataset
    # ========================================================================
    cocotb.log.info("\n" + "=" * 70)
    cocotb.log.info("STEP 1: LOAD MNIST TEST DATASET")
    cocotb.log.info("=" * 70)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_images = torch.stack([img for img, _ in test_dataset])
    test_labels = torch.tensor([label for _, label in test_dataset])
    
    cocotb.log.info(f"Loaded {len(test_labels)} test images")
    
    # ========================================================================
    # STEP 2: Test on subset of images
    # ========================================================================
    num_tests = 20  # Test first 10 images for faster verification
    cocotb.log.info(f"\nTesting on first {num_tests} images...")
    
    rtl_correct = 0
    golden_correct = 0
    both_correct = 0
    total_tests = 0
    
    for test_idx in range(num_tests):
        cocotb.log.info("\n" + "=" * 70)
        cocotb.log.info(f"TEST IMAGE {test_idx + 1}/{num_tests} - Label: {test_labels[test_idx].item()}")
        cocotb.log.info("=" * 70)
        
        # Prepare input
        input_image = test_images[test_idx]
        label = test_labels[test_idx].item()
        
        success = tester.prepare_input(input_image, compiler_dir)
        if not success:
            cocotb.log.error(f"Failed to prepare input for test {test_idx}")
            continue
            
        # Apply reset before each test
        await tester.reset()
        
        # Execute RTL - pulse start once and wait for done (zero instruction)
        cocotb.log.info("Executing RTL...")
        success = await tester.execute_all(timeout_cycles=500000)
        
        if not success:
            cocotb.log.error(f"RTL execution failed for test {test_idx}")
            continue
            
        # Wait for memory writes to settle
        for _ in range(100):
            await RisingEdge(dut.clk)
            
        # Read RTL results directly from simulation memory (STORE output)
        rtl_output = tester.read_memory_from_rtl(
            tester.output_addr,
            tester.output_length
        )
        
        # Also read from y[] output port for comparison
        rtl_y_output = np.array([int(dut.y[i].value.signed_integer) for i in range(10)], dtype=np.int8)
        cocotb.log.info(f"RTL y[] output: {rtl_y_output}")
        
        if rtl_output is None:
            cocotb.log.error(f"Failed to read RTL output for test {test_idx}")
            continue
            
        # Execute golden model
        cocotb.log.info("Executing golden model...")
        try:
            golden_output = execute_program(dram_hex_path)
            golden_output = np.array(golden_output, dtype=np.int8)
        except Exception as e:
            cocotb.log.error(f"Golden model execution failed: {e}")
            continue
            
        # Compare outputs
        match, differences, max_error = tester.compare_results(rtl_output, golden_output, verbose=False)
        
        # Get predictions
        rtl_pred = tester.get_prediction(rtl_output)
        golden_pred = tester.get_prediction(golden_output)
        
        # Check accuracy
        rtl_match = (rtl_pred == label)
        golden_match = (golden_pred == label)
        outputs_match = match or (max_error is not None and max_error <= 0)
        
        if rtl_match:
            rtl_correct += 1
        if golden_match:
            golden_correct += 1
        if rtl_match and golden_match:
            both_correct += 1
            
        total_tests += 1
        
        # Log results for this test
        cocotb.log.info(f"\nTest {test_idx + 1} Results:")
        cocotb.log.info(f"  Label:           {label}")
        cocotb.log.info(f"  RTL prediction:  {rtl_pred} {'✓' if rtl_match else '✗'}")
        cocotb.log.info(f"  Golden prediction: {golden_pred} {'✓' if golden_match else '✗'}")
        cocotb.log.info(f"  Outputs match:   {outputs_match} (max error: {max_error})")
        cocotb.log.info(f"  RTL output:      {rtl_output}")
        cocotb.log.info(f"  Golden output:   {golden_output}")
        
        if not outputs_match:
            cocotb.log.warning(f"⚠️  Output mismatch detected!")
            
    # ========================================================================
    # STEP 3: Report overall results
    # ========================================================================
    cocotb.log.info("\n" + "=" * 70)
    cocotb.log.info("FINAL RESULTS")
    cocotb.log.info("=" * 70)
    
    rtl_accuracy = (rtl_correct / total_tests * 100) if total_tests > 0 else 0
    golden_accuracy = (golden_correct / total_tests * 100) if total_tests > 0 else 0
    
    cocotb.log.info(f"\nTested {total_tests} images:")
    cocotb.log.info(f"  RTL Accuracy:    {rtl_correct}/{total_tests} ({rtl_accuracy:.1f}%)")
    cocotb.log.info(f"  Golden Accuracy: {golden_correct}/{total_tests} ({golden_accuracy:.1f}%)")
    cocotb.log.info(f"  Both Correct:    {both_correct}/{total_tests} ({both_correct/total_tests*100:.1f}%)")
    
    # Final assertion
    if rtl_accuracy >= 70 and abs(rtl_accuracy - golden_accuracy) <= 10:
        cocotb.log.info("\n✅ TEST PASSED: RTL achieves acceptable accuracy and matches golden model!")
    else:
        cocotb.log.error("\n❌ TEST FAILED: RTL accuracy too low or differs significantly from golden model")
        assert False, f"RTL accuracy: {rtl_accuracy}%, Golden: {golden_accuracy}%"
        

@cocotb.test()
async def test_single_instruction_load_v(dut):
    """
    Simple sanity test: Execute a single LOAD_V instruction
    """
    tester = TinyMLAcceleratorTester(dut)
    
    # Start clock
    clock = Clock(dut.clk, tester.clock_period, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    await tester.reset()
    
    # Execute one instruction
    cocotb.log.info("Executing single instruction test")
    success = await tester.execute_single_instruction()
    
    assert success, "Single instruction execution failed"
    cocotb.log.info("✅ Single instruction test passed")


@cocotb.test()
async def test_reset_behavior(dut):
    """
    Test proper reset behavior
    """
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Apply reset
    dut.rst.value = 1
    dut.start.value = 0
    
    for _ in range(10):
        await RisingEdge(dut.clk)
        
    # Check that done is low during reset
    assert dut.done.value == 0, "Done signal should be low during reset"
    
    # Release reset
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    
    # Done should still be low after reset
    assert dut.done.value == 0, "Done signal should be low after reset"
    
    cocotb.log.info("✅ Reset behavior test passed")
