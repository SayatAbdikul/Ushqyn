import os
import sys
import numpy as np
import torch
import cocotb
from cocotb.triggers import RisingEdge

# Ensure compiler path is available (works when imported directly)
compiler_dir = os.path.join(os.path.dirname(__file__), '../../compiler')
compiler_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../compiler'))
if compiler_dir not in sys.path:
    sys.path.insert(0, compiler_dir)

from dram import save_dram_to_file, save_input_to_dram, read_from_dram, get_dram
from helper_functions import quantize_tensor_f32_int8


class TinyMLAcceleratorTester:
    """Helper class to manage the accelerator testbench"""

    def __init__(self, dut):
        self.dut = dut
        self.clock_period = 10  # 10ns = 100MHz
        self.output_addr = 0x8C0  # Updated output address
        self.output_length = 10  # Default output length for MNIST (10 classes)
        self.input_addr = 0xC0  # Updated input address
        self.dram_offsets = {
            "inputs":       0xC0,
            "biases":       0x4C0,
            "outputs":      0x8C0,
            "weights":      0x940,
            "conv_weights": 0x3000,
        }

    async def reset(self):
        """Apply reset to the DUT"""
        self.dut.rst.value = 1
        self.dut.start.value = 0
        await RisingEdge(self.dut.clk)
        await RisingEdge(self.dut.clk)
        self.dut.rst.value = 0
        await RisingEdge(self.dut.clk)
        cocotb.log.info("Reset complete")

    async def wait_for_done(self, timeout_cycles=200000):
        """Wait for the done signal with timeout"""
        cycle_count = 0
        while cycle_count < timeout_cycles:
            await RisingEdge(self.dut.clk)
            if self.dut.done.value == 1:
                cocotb.log.info(f"Done signal received after {cycle_count} cycles")
                return True
            cycle_count += 1

            # Progress indicator every 10000 cycles
            if cycle_count % 10000 == 0:
                cocotb.log.info(f"  ... still executing (cycle {cycle_count})")

        cocotb.log.error(f"Timeout waiting for done after {timeout_cycles} cycles")
        return False

    async def execute_single_instruction(self):
        """Execute a single instruction by pulsing start"""
        # Pulse start signal
        self.dut.start.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.start.value = 0

        # Wait for completion
        success = await self.wait_for_done()
        return success

    async def execute_program(self, max_instructions=20):
        """
        Execute all instructions in the program sequentially.
        DEPRECATED: Use execute_all() instead for correct behavior.
        The program ends when an all-zero instruction is fetched.
        """
        cocotb.log.info("=" * 70)
        cocotb.log.info("Starting program execution on RTL")
        cocotb.log.info("=" * 70)

        for instr_num in range(max_instructions):
            cocotb.log.info(f"\n--- Executing instruction #{instr_num} ---")

            # Execute one instruction
            success = await self.execute_single_instruction()

            if not success:
                cocotb.log.error(f"Failed to execute instruction #{instr_num}")
                return False

            # Small delay between instructions
            await RisingEdge(self.dut.clk)
            await RisingEdge(self.dut.clk)

        cocotb.log.info("\n" + "=" * 70)
        cocotb.log.info("Program execution complete")
        cocotb.log.info("=" * 70 + "\n")
        return True

    async def execute_all(self, timeout_cycles=500000):
        """
        Execute all instructions by pulsing start once and waiting for done.
        The RTL will run continuously until it hits a zero instruction,
        at which point it pulses done. This matches the C++ testbench behavior.
        """
        cocotb.log.info("=" * 70)
        cocotb.log.info("Starting program execution on RTL (single start pulse)")
        cocotb.log.info("=" * 70)

        # Pulse start signal once
        self.dut.start.value = 1
        await RisingEdge(self.dut.clk)
        self.dut.start.value = 0

        # Wait for completion - done will pulse when zero instruction is hit
        success = await self.wait_for_done(timeout_cycles=timeout_cycles)

        if success:
            cocotb.log.info("\n" + "=" * 70)
            cocotb.log.info("Program execution complete (zero instruction hit)")
            cocotb.log.info("=" * 70 + "\n")
        else:
            cocotb.log.error("Program execution failed or timed out")

        return success

    def read_memory_from_rtl(self, start_addr, length):
        """
        Read memory contents directly from RTL simulation memory.
        Accesses the top-level unified memory array.
        """
        try:
            # Access unified memory in top module
            # Path: top -> main_memory -> memory
            main_mem = self.dut.main_memory.memory

            # Read memory values
            result = []
            for addr in range(start_addr, start_addr + length):
                val = main_mem[addr].value.integer
                if val > 127:
                    val = val - 256
                result.append(val)

            result = np.array(result, dtype=np.int8)
            cocotb.log.info(f"Read {length} bytes from RTL memory at address 0x{start_addr:06X}")
            return result

        except AttributeError as e:
            cocotb.log.error(f"Cannot access RTL memory hierarchy: {e}")
            cocotb.log.error("Memory path may be incorrect. Check module hierarchy.")
            return None
        except Exception as e:
            cocotb.log.error(f"Error reading RTL memory: {e}")
            return None

    def read_memory_from_file(self, hex_file, start_addr, length):
        """
        Read memory contents from hex file after RTL execution.
        Each line in the hex file is one byte in hex format.
        NOTE: This method is deprecated - use read_memory_from_rtl() instead.
        """
        try:
            with open(hex_file, 'r') as f:
                lines = f.readlines()

            # Convert hex strings to integers
            memory = []
            for line in lines:
                line = line.strip()
                if line:
                    val = int(line, 16)
                    if val > 127:
                        val = val - 256
                    memory.append(val)

            end_addr = start_addr + length
            if end_addr > len(memory):
                cocotb.log.error(f"Requested memory region [{start_addr}:{end_addr}] exceeds file size {len(memory)}")
                return None

            result = np.array(memory[start_addr:end_addr], dtype=np.int8)
            cocotb.log.info(f"Read {length} bytes from {hex_file} at address 0x{start_addr:06X}")
            return result

        except FileNotFoundError:
            cocotb.log.error(f"Memory file {hex_file} not found")
            return None
        except Exception as e:
            cocotb.log.error(f"Error reading memory file: {e}")
            return None

    def write_to_all_rtl_memories(self, address, value):
        """
        Write a single byte to the unified RTL memory instance at the given address.
        """
        # Convert signed int8 to unsigned for writing
        unsigned_val = value if value >= 0 else value + 256
        
        try:
            # Write to unified memory instance
            self.dut.main_memory.memory[address].value = unsigned_val
        except AttributeError as e:
            # Log error only once for first failure
            if address == 0:
                cocotb.log.error(f"Failed to write to RTL memory - hierarchy path may be incorrect: {e}")
                cocotb.log.error("Available attributes on dut:")
                try:
                    cocotb.log.error(f"  dut children: {dir(self.dut)[:20]}...")
                except:
                    pass
            raise
        except Exception as e:
            cocotb.log.error(f"Failed to write to RTL memory at address 0x{address:06X}: {e}")
            raise
    
    def verify_memory_write(self, address, expected_value):
        """
        Verify that a value was correctly written to the unified memory instance.
        """
        unsigned_expected = expected_value if expected_value >= 0 else expected_value + 256
        
        try:
            mem_val = self.dut.main_memory.memory[address].value.integer
            
            match = (mem_val == unsigned_expected)
            
            if not match:
                cocotb.log.warning(f"Memory mismatch at 0x{address:06X}: "
                                  f"expected={unsigned_expected}, "
                                  f"actual={mem_val}")
            return match
        except Exception as e:
            cocotb.log.error(f"Error verifying memory at 0x{address:06X}: {e}")
            return False
    
    def load_dram_to_all_rtl_memories(self, dram_array, sparse=True):
        """
        Load the entire DRAM contents to all RTL memory instances.
        This should be called after preparing the dram.hex file to ensure
        all RTL memories have the same data.
        
        If sparse=True (default), only writes non-zero values for faster loading.
        """
        if sparse:
            # Only write non-zero values (much faster)
            non_zero_count = 0
            for addr in range(len(dram_array)):
                if dram_array[addr] != 0:
                    self.write_to_all_rtl_memories(addr, int(dram_array[addr]))
                    non_zero_count += 1
            cocotb.log.info(f"Loaded {non_zero_count} non-zero bytes to all RTL memory instances (sparse mode)")
        else:
            cocotb.log.info(f"Loading {len(dram_array)} bytes to all RTL memory instances...")
            for addr in range(len(dram_array)):
                self.write_to_all_rtl_memories(addr, int(dram_array[addr]))
            cocotb.log.info(f"Successfully loaded DRAM to all 4 RTL memory instances")
    
    def sync_dram_to_rtl(self):
        """
        Sync the current Python DRAM state to all RTL memory instances.
        
        This is crucial because the RTL has 4 SEPARATE memory instances that
        each initialize from dram.hex at time 0 via $readmemh. After the initial
        load, updates to dram.hex are NOT reflected in RTL memory.
        
        Call this method after:
        - save_initializers_to_dram() - to sync weights and biases
        - assemble_file() - to sync instructions
        - Any changes to DRAM via Python
        """
        dram_array = get_dram()
        self.load_dram_to_all_rtl_memories(dram_array)
        
        # Verify some key locations were written correctly
        cocotb.log.info("Verifying memory writes...")
        # Check input address area
        for i in range(min(10, len(dram_array) - self.dram_offsets["inputs"])):
            addr = self.dram_offsets["inputs"] + i
            if addr < len(dram_array):
                self.verify_memory_write(addr, int(dram_array[addr]))
    
    def prepare_input(self, input_tensor, compiler_dir):
        """
        Prepare input data by writing it to DRAM at the input address.
        Also writes directly to all RTL memory instances to ensure consistency.
        Uses improved quantization with dynamic scaling.
        """
        from dram import dram as dram_array_ref
        
        # Improved quantization logic
        input_numpy = input_tensor.numpy().squeeze()
        scale = np.max(np.abs(input_numpy)) / 127 if np.max(np.abs(input_numpy)) > 0 else 1.0
        dummy_input = quantize_tensor_f32_int8(input_numpy, scale).flatten()

        cocotb.log.info(f"Preparing input: shape={input_tensor.shape}, quantized_length={len(dummy_input)}")

        # Clear the input area in Python DRAM first (to allow overwriting)
        input_addr = self.dram_offsets["inputs"]
        input_len = len(dummy_input)
        dram_array_ref[input_addr:input_addr + input_len] = 0
        
        save_input_to_dram(dummy_input, self.dram_offsets["inputs"])

        written_input = read_from_dram(self.dram_offsets["inputs"], len(dummy_input))
        if not np.array_equal(dummy_input, written_input):
            cocotb.log.error("Input data mismatch after writing to DRAM")
            return False

        dram_hex_path = os.path.join(compiler_dir, 'dram.hex')
        save_dram_to_file(dram_hex_path)
        cocotb.log.info(f"Input saved to {dram_hex_path}")
        
        # CRITICAL: Write input data directly to all RTL memory instances
        # The $readmemh in RTL only runs once at simulation start, so we must
        # use backdoor writes to update the memories with new input data
        for i, val in enumerate(dummy_input):
            self.write_to_all_rtl_memories(input_addr + i, int(val))
        cocotb.log.info(f"Input data written to all RTL memory instances at 0x{input_addr:06X}")

        return True

    def compare_results(self, rtl_output, golden_output, verbose=True):
        """
        Compare RTL and golden model outputs element-by-element.
        Returns (match, differences, max_error)
        """
        if rtl_output is None or golden_output is None:
            cocotb.log.error("Cannot compare - one or both outputs are None")
            return False, None, None

        if len(rtl_output) != len(golden_output):
            cocotb.log.error(f"Length mismatch: RTL={len(rtl_output)}, Golden={len(golden_output)}")
            return False, None, None

        differences = rtl_output - golden_output
        max_error = np.max(np.abs(differences))
        num_mismatches = np.count_nonzero(differences)

        match = np.array_equal(rtl_output, golden_output)

        if verbose:
            cocotb.log.info("\n" + "=" * 70)
            cocotb.log.info("RESULTS COMPARISON")
            cocotb.log.info("=" * 70)
            cocotb.log.info(f"Output length: {len(rtl_output)} elements")
            cocotb.log.info(f"Exact match: {match}")
            cocotb.log.info(f"Mismatches: {num_mismatches}/{len(rtl_output)} elements")
            cocotb.log.info(f"Max absolute error: {max_error}")

            cocotb.log.info("\nElement-by-element comparison:")
            cocotb.log.info("-" * 70)
            cocotb.log.info(f"{'Index':<8} {'RTL':<12} {'Golden':<12} {'Diff':<12} {'Match'}")
            cocotb.log.info("-" * 70)

            for i in range(len(rtl_output)):
                rtl_val = int(rtl_output[i])
                golden_val = int(golden_output[i])
                diff = rtl_val - golden_val
                match_str = "✓" if diff == 0 else "✗"
                cocotb.log.info(f"{i:<8} {rtl_val:<12} {golden_val:<12} {diff:<12} {match_str}")

            cocotb.log.info("-" * 70)

            if match:
                cocotb.log.info("\n✅ PASS: RTL output matches golden model exactly!")
            else:
                cocotb.log.warning(f"\n⚠️  MISMATCH: {num_mismatches} element(s) differ (max error: {max_error})")

            cocotb.log.info("=" * 70 + "\n")

        return match, differences, max_error

    def get_prediction(self, output):
        """Get predicted class from output vector"""
        return np.argmax(output)
