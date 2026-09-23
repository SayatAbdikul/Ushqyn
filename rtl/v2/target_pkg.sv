// Generated from hardware/targets/tang_nano_20k_v2.json; do not edit.
package target_pkg;
    localparam integer TARGET_ID = 8196;
    localparam integer NUMERICS = 2;
    localparam integer DESC_VERSION = 2;
    localparam integer PROTOCOL_VERSION = 2;
    localparam integer CLOCK_HZ = 27000000;
    localparam integer BAUD = 115200;
    localparam integer ADDR_BITS = 24;
    localparam integer MEM_BYTES = 32768;
    localparam integer LANES = 8;
    localparam integer DESC_BYTES = 64;
    localparam integer MAX_TRANSFER = 64;
    localparam integer RX_TIMEOUT = 540000;
    localparam integer MAX_DESCRIPTORS = 256;
    localparam integer WATCHDOG = 27000000;
    localparam logic [7:0] OP_HALT = 0;
    localparam logic [7:0] OP_GEMM = 1;
    localparam logic [7:0] OP_RELU = 2;
    localparam logic [7:0] OP_COPY = 3;
    localparam logic [7:0] OP_CONV = 4;
    localparam logic [7:0] OP_MAXPOOL = 5;
    localparam logic [7:0] OP_DWCONV = 6;
    localparam logic [7:0] OP_AVGPOOL = 7;
    localparam logic [7:0] OP_CLIP = 8;
endpackage
