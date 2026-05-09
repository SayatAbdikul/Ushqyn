#!/usr/bin/env python3
"""
Quick runner script for Ushqyn Cocotb tests
Provides a simple interface to prepare and run verification tests
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, cwd=None, description=None):
    """Run a shell command and handle errors"""
    if description:
        print(f"\n{'='*70}")
        print(f"  {description}")
        print('='*70)
    
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    
    if result.returncode != 0:
        print(f"\n❌ Error: Command failed with exit code {result.returncode}")
        return False
    return True


def check_dependencies():
    """Check if required tools are installed"""
    print("\n🔍 Checking dependencies...")
    
    dependencies = {
        'verilator': 'Verilator',
        'cocotb-config': 'Cocotb',
        'python3': 'Python 3'
    }
    
    missing = []
    for cmd, name in dependencies.items():
        result = subprocess.run(f"which {cmd}", shell=True, capture_output=True)
        if result.returncode != 0:
            missing.append(name)
            print(f"  ❌ {name} not found")
        else:
            print(f"  ✓ {name} found")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nInstallation instructions:")
        if 'Verilator' in missing:
            print("  Verilator: brew install verilator  (macOS)")
            print("             sudo apt-get install verilator  (Ubuntu)")
        if 'Cocotb' in missing:
            print("  Cocotb: pip install cocotb")
        return False
    
    print("\n✓ All dependencies found")
    return True


def prepare_test_data(compiler_dir):
    """Generate dram.hex if it doesn't exist"""
    dram_hex = compiler_dir / 'dram.hex'
    
    if dram_hex.exists():
        print(f"\n✓ dram.hex already exists at {dram_hex}")
        return True
    
    print(f"\n⚠️  dram.hex not found. Generating it...")
    
    # Check if we have the required files
    model_weights = compiler_dir / 'digit_model_weights.pth'
    if not model_weights.exists():
        print(f"❌ Error: {model_weights} not found")
        print("Please ensure you have the pre-trained model weights")
        return False
    
    # Run the compiler
    if not run_command('python3 main.py', cwd=compiler_dir, 
                      description="Running compiler to generate dram.hex"):
        return False
    
    if not dram_hex.exists():
        print(f"❌ Error: dram.hex was not generated")
        return False
    
    print(f"\n✓ dram.hex generated successfully")
    return True


def run_tests(test_dir, clean=False, verbose=False):
    """Run the cocotb tests"""
    
    # Clean if requested
    if clean:
        print("\n🧹 Cleaning previous build...")
        run_command('make clean_all', cwd=test_dir)
    
    # Run the test
    test_cmd = 'make'
    if verbose:
        test_cmd += ' VERBOSE=1'
    
    success = run_command(test_cmd, cwd=test_dir, 
                         description="Running Cocotb verification tests")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='Ushqyn Cocotb Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    Run tests with default settings
  %(prog)s --clean            Clean and run tests
  %(prog)s --prepare-only     Only prepare test data, don't run tests
  %(prog)s --skip-checks      Skip dependency checks (faster)
        """
    )
    
    parser.add_argument('--clean', action='store_true',
                       help='Clean previous build before running')
    parser.add_argument('--prepare-only', action='store_true',
                       help='Only prepare test data (generate dram.hex)')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip dependency checks')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output during tests')
    
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent
    compiler_dir = script_dir / '../../compiler'
    compiler_dir = compiler_dir.resolve()
    
    print("=" * 70)
    print("  Ushqyn — Cocotb Test Runner")
    print("=" * 70)
    print(f"\nTest directory: {script_dir}")
    print(f"Compiler directory: {compiler_dir}")
    
    # Check dependencies (unless skipped)
    if not args.skip_checks:
        if not check_dependencies():
            print("\n❌ Dependency check failed. Please install missing tools.")
            return 1
    
    # Prepare test data
    if not prepare_test_data(compiler_dir):
        print("\n❌ Failed to prepare test data")
        return 1
    
    if args.prepare_only:
        print("\n✓ Test data prepared successfully")
        print("Run without --prepare-only to execute tests")
        return 0
    
    # Run tests
    if not run_tests(script_dir, clean=args.clean, verbose=args.verbose):
        print("\n❌ Tests failed")
        return 1
    
    # Success!
    print("\n" + "=" * 70)
    print("  ✅ All tests completed successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print(f"  - Waveform: {script_dir}/dump.vcd")
    print(f"  - Results: {script_dir}/results.xml")
    print("\nView waveforms with: gtkwave dump.vcd")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
