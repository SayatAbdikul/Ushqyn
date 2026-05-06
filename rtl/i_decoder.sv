module i_decoder (
    input  logic [63:0] instr,
    output logic [4:0]  opcode,
    output logic [4:0]  dest,
    output logic [9:0]  length_or_cols,
    output logic [9:0]  rows,
    output logic [23:0] addr,
    output logic [4:0]  b,
    output logic [4:0]  x,
    output logic [4:0]  w,

    // CNN Geometry Extensions
    output logic [5:0] fmap_h, fmap_w, in_channels, out_channels,
    output logic [3:0] kernel_h, kernel_w,
    output logic [2:0] stride_val, pad_val, pool_size,

    // CONV2D_RUN fused-ReLU bit (instr[25])
    output logic       relu_flag
);

    always_comb begin
        opcode = instr[4:0];

        // Defaults
        dest   = 0;
        length_or_cols = 0;
        rows   = 0;
        addr   = 0;
        b = 0;
        x = 0;
        w = 0;
        fmap_h = 0;
        fmap_w = 0;
        in_channels = 0;
        out_channels = 0;
        kernel_h = 0;
        kernel_w = 0;
        stride_val = 0;
        pad_val = 0;
        pool_size = 0;
        relu_flag = 0;

        case (opcode)
            5'h00: begin // NOP
            end

            5'h01, 5'h03: begin // LOAD_V or STORE
                dest           = instr[9:5];
                length_or_cols = instr[19:10];
                addr           = instr[63:40];
            end

            5'h02: begin // LOAD_M
                dest           = instr[9:5];
                length_or_cols = instr[19:10];  // cols
                rows           = instr[29:20];
                addr           = instr[63:40];
            end

            5'h04: begin // GEMV
                dest = instr[9:5];
                length_or_cols = instr[19:10]; // cols
                rows = instr[29:20];
                b    = instr[34:30];
                x    = instr[39:35];
                w    = instr[44:40];
            end

            5'h05: begin // RELU
                dest           = instr[9:5];
                x              = instr[14:10];
                length_or_cols = instr[29:20];  // RELU length field
            end
            
            5'h06: begin // CONV2D_CFG
                dest        = instr[9:5];
                fmap_h      = instr[15:10];
                fmap_w      = instr[21:16];
                in_channels = instr[27:22];
                out_channels= instr[33:28];
                kernel_h    = instr[37:34];
                kernel_w    = instr[41:38];
                stride_val  = instr[44:42];
                pad_val     = instr[47:45];
            end
            
            5'h07: begin // CONV2D_RUN
                dest      = instr[9:5];
                x         = instr[14:10];
                w         = instr[19:15];
                b         = instr[24:20];
                relu_flag = instr[25];   // fused-ReLU activation flag
            end
            
            5'h08: begin // MAXPOOL
                dest      = instr[9:5];
                x         = instr[14:10];
                pool_size = instr[17:15];
                stride_val= instr[20:18];
                fmap_h    = instr[26:21];
                fmap_w    = instr[32:27];
                // MAXPOOL's `channels` field (5 bits at [37:33]) is muxed onto
                // the `in_channels` output port. CONV2D_CFG and MAXPOOL never
                // coexist in flight, so the same wire serves both.
                in_channels = instr[37:33];
            end

            default: begin
                
            end
            
        endcase
    end
endmodule
