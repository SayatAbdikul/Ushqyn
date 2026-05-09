#!/usr/bin/env python3
"""
Run cocotb testbenches for Ushqyn with easy test selection.

Usage:
    python3 run_test.py                    # Run full accelerator tests
    python3 run_test.py --gemv             # Run GEMV module tests only
    python3 run_test.py --all              # Run both full and GEMV tests
    python3 run_test.py --clean            # Clean and run full tests
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    # Change to test directory
    test_dir = Path(__file__).parent
    os.chdir(test_dir)
    
    # Parse arguments
    run_full = True
    run_gemv = False
    clean = False
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == '--gemv':
            run_full = False
            run_gemv = True
        elif arg == '--all':
            run_full = True
            run_gemv = True
        elif arg == '--clean':
            clean = True
            run_full = True
        elif arg == '--help' or arg == '-h':
            print(__doc__)
            return 0
    
    # Clean if requested
    if clean:
        print("Cleaning build artifacts...")
        run_command("make clean_all", "Clean all generated files")
    
    # Prepare dram.hex
    print("\nPreparing test environment...")
    compiler_dir = Path(__file__).parent / "../../compiler"
    if not (compiler_dir / "dram.hex").exists():
        print("Generating dram.hex...")
        if not run_command(f"cd {compiler_dir} && python3 main.py", 
                          "Generate dram.hex from compiler"):
            print("WARNING: dram.hex generation failed (may be optional)")
    
    # Run full accelerator tests
    if run_full:
        success = run_command(
            "make TEST_TARGET=full_accelerator run_test",
            "Running Full Accelerator Tests"
        )
        if not success:
            print("❌ Full accelerator tests failed")
            return 1
        print("✅ Full accelerator tests passed")
    
    # Run GEMV tests
    if run_gemv:
        success = run_command(
            "make TEST_TARGET=top_gemv run_test",
            "Running GEMV Module Tests"
        )
        if not success:
            print("❌ GEMV tests failed")
            return 1
        print("✅ GEMV tests passed")
    
    print(f"\n{'='*60}")
    print("  All selected tests completed successfully!")
    print(f"{'='*60}\n")
    
    # Print next steps
    if not clean and not run_gemv:
        print("Waveform saved to: dump.vcd")
        print("View with: gtkwave dump.vcd")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
