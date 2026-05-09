# UART Tools for Ushqyn

C++ utilities for communicating with the FPGA via UART.

## Building

```bash
make
```

## Tools

### uart_loader
Uploads a hex file to FPGA memory via UART.

```bash
./uart_loader <serial_port> <hex_file>

# Example (macOS):
./uart_loader /dev/cu.usbserial-XXX ../../compiler/dram.hex

# Example (Linux):
./uart_loader /dev/ttyUSB0 ../../compiler/dram.hex
```

**Requirements:**
- FPGA must be in **Load Mode** (S2 button held/active)
- Baud rate: 115200

### uart_reader
Reads data from FPGA memory via UART. 

```bash
./uart_reader <serial_port> <start_address> <length>

# Example - Read 10 bytes from output area (0x8C0):
./uart_reader /dev/ttyUSB0 0x8C0 10
```

**Note:** The reader requires RTL support for read requests. The current `simple_memory.sv` only supports writes. This tool is provided as a template for when read support is added to the FPGA design.

## Serial Port Detection

**macOS:**
```bash
ls /dev/cu.usbserial-*
```

**Linux:**
```bash
ls /dev/ttyUSB*
```

## Workflow

1. Connect FPGA to computer via USB-UART
2. Put FPGA in Load Mode (hold S2 button)
3. Upload the hex file: `./uart_loader /dev/ttyUSB0 ../../compiler/dram.hex`
4. Release S2 to switch to Run Mode
5. Press S1 (Reset) to start execution
6. (Future) Read results: `./uart_reader /dev/ttyUSB0 0x8C0 10`
