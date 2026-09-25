// UART packet-to-tile bridge. The v2 framing and CRC remain unchanged.
// 0x000000..007fff: scratchpad; 0x400000..40001f: DMA registers;
// 0x800000..ffffff: byte-addressed 8-MiB SDRAM window.
module v2_tiled_host_bridge #(
    parameter integer TIMEOUT_CYCLES=target_pkg::RX_TIMEOUT
)(
    input logic clk, rst_n,
    input logic rx_valid,
    input logic [7:0] rx_data,
    output logic tx_valid,
    output logic [7:0] tx_data,
    input logic tx_ready,
    output logic ext_req, ext_wr,
    output logic [23:0] ext_addr,
    output logic [63:0] ext_wdata,
    output logic [7:0] ext_wstrb,
    input logic ext_ready, ext_rvalid,
    input logic [63:0] ext_rdata,
    input logic memory_initialized,
    input logic memory_port_busy,
    output logic engine_busy, dma_busy
);
    logic start_engine, abort_command, clear_counters;
    logic [23:0] start_pc;
    logic [7:0] engine_error, dma_error;
    logic [31:0] engine_elapsed, compute_cycles, wait_cycles, control_cycles;
    logic [31:0] useful_macs, read_bytes, write_bytes, layer_count;
    logic [31:0] protocol_errors, dma_bytes_copied;
    logic [31:0] dma_elapsed_cycles;
    logic dma_pending;
    logic [15:0] engine_live_limit, overlap_cycles;
    logic [7:0] guard_error;
    logic command_req, command_wr, command_ready, command_rvalid;
    logic [23:0] command_addr;
    logic [63:0] command_wdata, command_rdata;
    logic [7:0] command_wstrb;
    logic sram_ready, sram_rvalid;
    logic [63:0] sram_rdata;
    logic core_ext_req, core_ext_wr, core_ext_ready, core_ext_rvalid;
    logic [23:0] core_ext_addr;
    logic [63:0] core_ext_wdata;
    logic [7:0] core_ext_wstrb;
    logic engine_done, dma_done;
    logic [23:0] dma_ext_base, dma_sram_base;
    logic [31:0] dma_length;
    logic dma_to_sram, dma_start, dma_abort_register;
    logic mmio_rvalid;
    logic [63:0] mmio_rdata;
    logic [1:0] read_owner; // 1 = DMA, 2 = host
    logic seq_busy, seq_start, seq_abort, seq_engine_start, seq_dma_start;
    logic seq_dma_to_sram, seq_host_ready, seq_host_rvalid, seq_reg_rvalid;
    logic [7:0] seq_error;
    logic [11:0] seq_index;
    logic [23:0] seq_pc, seq_dma_ext, seq_dma_sram;
    logic [31:0] seq_dma_length, seq_elapsed, seq_engine_cycles, seq_dma_cycles, seq_overlap;
    logic [63:0] seq_host_rdata, seq_reg_rdata;
    wire seq_program_region = command_addr>=24'h500000 && command_addr<24'h508000;
    wire seq_control_region = command_addr>=24'h410000 && command_addr<24'h410020;
    wire active_dma_start = seq_busy ? seq_dma_start : dma_start;

    wire sram_region = command_addr < 24'h008000;
    wire mmio_region = command_addr >= 24'h400000 &&
                       command_addr < 24'h400020;
    wire external_region = command_addr[23];
    wire host_ext_req = command_req && external_region;
    wire [23:0] host_ext_addr = {1'b0,command_addr[22:0]};
    wire host_ext_ready = ext_ready && !core_ext_req;
    wire host_ext_rvalid = ext_rvalid && read_owner == 2'd2;

    v2_command #(.TIMEOUT_CYCLES(TIMEOUT_CYCLES), .TILED_MEM_MAP(1'b1))
    command (
        .clk(clk), .rst_n(rst_n), .rx_valid(rx_valid), .rx_data(rx_data),
        .tx_valid(tx_valid), .tx_data(tx_data), .tx_ready(tx_ready),
        .start(start_engine), .abort_run(abort_command),
        .clear_counters(clear_counters), .start_pc(start_pc),
        .busy(seq_busy || engine_busy || dma_busy || dma_pending || !memory_initialized),
        .core_error(seq_error != 0 ? seq_error : engine_error != 0 ? engine_error :
                    dma_error != 0 ? dma_error : guard_error),
        .mmio_while_busy(!seq_busy && memory_initialized && engine_busy &&
                         !dma_busy && !dma_pending),
        .elapsed(engine_elapsed), .compute_cycles(compute_cycles),
        .wait_cycles(wait_cycles), .control_cycles(control_cycles),
        .useful_macs(useful_macs), .read_bytes(read_bytes),
        .write_bytes(write_bytes), .layer_count(layer_count),
        .mem_req(command_req), .mem_wr(command_wr),
        .mem_addr(command_addr), .mem_wdata(command_wdata),
        .mem_wstrb(command_wstrb), .mem_ready(command_ready),
        .mem_rvalid(command_rvalid), .mem_rdata(command_rdata),
        .protocol_errors(protocol_errors)
    );

    v2_tiled_core core (
        .clk(clk), .rst_n(rst_n),
        .engine_start(seq_busy ? seq_engine_start : start_engine),
        .engine_abort(abort_command || seq_abort),
        .engine_start_pc(seq_busy ? seq_pc : start_pc), .engine_busy(engine_busy),
        .engine_done(engine_done), .engine_error(engine_error),
        .engine_elapsed(engine_elapsed), .compute_cycles(compute_cycles),
        .wait_cycles(wait_cycles), .control_cycles(control_cycles),
        .useful_macs(useful_macs), .read_bytes(read_bytes),
        .write_bytes(write_bytes), .layer_count(layer_count),
        .dma_start(active_dma_start),
        .dma_abort(abort_command || seq_abort || dma_abort_register),
        .dma_to_sram(seq_busy ? seq_dma_to_sram : dma_to_sram),
        .dma_sram_base(seq_busy ? seq_dma_sram : dma_sram_base),
        .dma_ext_base(seq_busy ? seq_dma_ext : dma_ext_base),
        .dma_length(seq_busy ? seq_dma_length : dma_length),
        .dma_busy(dma_busy), .dma_done(dma_done), .dma_error(dma_error),
        .dma_bytes_copied(dma_bytes_copied),
        .host_req(command_req && sram_region), .host_wr(command_wr),
        .host_addr(command_addr), .host_wdata(command_wdata),
        .host_wstrb(command_wstrb), .host_ready(sram_ready),
        .host_rvalid(sram_rvalid), .host_rdata(sram_rdata),
        .ext_req(core_ext_req), .ext_wr(core_ext_wr),
        .ext_addr(core_ext_addr), .ext_wdata(core_ext_wdata),
        .ext_wstrb(core_ext_wstrb),
        .ext_ready(core_ext_ready), .ext_rvalid(core_ext_rvalid),
        .ext_rdata(ext_rdata)
    );

    v2_tile_sequencer sequencer (
        .clk(clk), .rst_n(rst_n), .start(seq_start), .abort_run(abort_command),
        .busy(seq_busy), .abort_units(seq_abort), .error_code(seq_error),
        .elapsed(seq_elapsed), .engine_cycles(seq_engine_cycles),
        .dma_cycles(seq_dma_cycles), .overlap_cycles(seq_overlap), .command_index(seq_index),
        .host_req(command_req && seq_program_region), .host_wr(command_wr),
        .host_addr(command_addr), .host_wdata(command_wdata), .host_wstrb(command_wstrb),
        .host_ready(seq_host_ready), .host_rvalid(seq_host_rvalid), .host_rdata(seq_host_rdata),
        .engine_start(seq_engine_start), .engine_pc(seq_pc),
        .engine_busy(engine_busy), .engine_error(engine_error),
        .dma_start(seq_dma_start), .dma_to_sram(seq_dma_to_sram),
        .dma_ext(seq_dma_ext), .dma_sram(seq_dma_sram), .dma_length(seq_dma_length),
        .dma_busy(dma_busy || dma_pending), .dma_error(dma_error)
    );
    wire [255:0] seq_registers = {32'd0, 20'd0,seq_index, seq_overlap,
        seq_dma_cycles, seq_engine_cycles, seq_elapsed, 32'h34514553,
        16'd0, seq_error, 7'd0,seq_busy};
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin seq_start<=0;seq_reg_rvalid<=0;seq_reg_rdata<=0;end
        else begin
            seq_start<=0;
            seq_reg_rvalid<=command_req && seq_control_region && !command_wr;
            if (command_req && seq_control_region) begin
                if (!command_wr) seq_reg_rdata<=seq_registers >> (command_addr[4:0]*8);
                else if (command_addr[4:0]==0 && command_wstrb[0] && command_wdata[0])
                    seq_start<=1;
            end
        end
    end

    always_comb begin
        ext_req = core_ext_req || host_ext_req;
        ext_wr = core_ext_req ? core_ext_wr : command_wr;
        ext_addr = core_ext_req ? core_ext_addr : host_ext_addr;
        ext_wdata = core_ext_req ? core_ext_wdata : command_wdata;
        ext_wstrb = core_ext_req ? core_ext_wstrb : command_wstrb;
        core_ext_ready = ext_ready && !host_ext_req;
        core_ext_rvalid = ext_rvalid && read_owner == 2'd1;
        command_ready = (sram_region && sram_ready) ||
                        (mmio_region && command_req) ||
                        (seq_program_region && seq_host_ready) ||
                        (seq_control_region && command_req) ||
                        (external_region && host_ext_ready);
        command_rvalid = sram_rvalid || mmio_rvalid || host_ext_rvalid || seq_host_rvalid || seq_reg_rvalid;
        command_rdata = seq_host_rvalid ? seq_host_rdata : seq_reg_rvalid ? seq_reg_rdata : sram_rvalid ? sram_rdata :
                        mmio_rvalid ? mmio_rdata : ext_rdata;
    end

    function automatic logic [7:0] control_byte(input logic [4:0] offset);
        case (offset)
            0: control_byte = dma_ext_base[7:0];
            1: control_byte = dma_ext_base[15:8];
            2: control_byte = dma_ext_base[23:16];
            4: control_byte = dma_sram_base[7:0];
            5: control_byte = dma_sram_base[15:8];
            6: control_byte = dma_sram_base[23:16];
            8: control_byte = dma_length[7:0];
            9: control_byte = dma_length[15:8];
            10: control_byte = dma_length[23:16];
            11: control_byte = dma_length[31:24];
            12: control_byte = {7'b0,dma_to_sram};
            16: control_byte = {6'b0,engine_busy,dma_busy};
            14: control_byte = engine_live_limit[7:0];
            15: control_byte = engine_live_limit[15:8];
            17: control_byte = engine_error != 0 ? engine_error :
                               dma_error != 0 ? dma_error : guard_error;
            18: control_byte = dma_bytes_copied[7:0];
            19: control_byte = dma_bytes_copied[15:8];
            20: control_byte = dma_bytes_copied[23:16];
            21: control_byte = dma_bytes_copied[31:24];
            22: control_byte = engine_elapsed[7:0];
            23: control_byte = engine_elapsed[15:8];
            24: control_byte = engine_elapsed[23:16];
            25: control_byte = engine_elapsed[31:24];
            26: control_byte = dma_elapsed_cycles[7:0];
            27: control_byte = dma_elapsed_cycles[15:8];
            28: control_byte = dma_elapsed_cycles[23:16];
            29: control_byte = dma_elapsed_cycles[31:24];
            30: control_byte = overlap_cycles[7:0];
            31: control_byte = overlap_cycles[15:8];
            default: control_byte = 0;
        endcase
    endfunction

    logic [63:0] control_word;
    integer lane;
    always_comb begin
        control_word = 0;
        for (integer k=0; k<8; k=k+1)
            control_word[k*8+:8] = control_byte(command_addr[4:0]+5'(k));
        lane = 0;
        for (integer k=0; k<8; k=k+1)
            if (command_wstrb[k]) lane = k;
    end
    wire [4:0] control_offset = command_addr[4:0]+5'(lane);
    wire control_write = command_req && command_ready &&
                         command_wr && mmio_region;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dma_ext_base <= 0; dma_sram_base <= 0; dma_length <= 0;
            dma_to_sram <= 0; dma_start <= 0; dma_abort_register <= 0;
            dma_elapsed_cycles <= 0; dma_pending <= 0;
            engine_live_limit <= 16'd32768; overlap_cycles <= 0;
            guard_error <= 0;
            mmio_rvalid <= 0; mmio_rdata <= 0; read_owner <= 0;
        end else begin
            dma_start <= 0;
            dma_abort_register <= 0;
            if (clear_counters) begin
                overlap_cycles <= 0;
                guard_error <= 0;
            end else if (engine_busy && dma_busy && overlap_cycles != 16'hffff)
                overlap_cycles <= overlap_cycles + 1'b1;
            if (active_dma_start) begin
                dma_elapsed_cycles <= 0;
                dma_pending <= 1;
            end else if (dma_pending) begin
                if (!dma_busy && !memory_port_busy) dma_pending <= 0;
                else dma_elapsed_cycles <= dma_elapsed_cycles + 1'b1;
            end
            mmio_rvalid <= command_req && command_ready &&
                            !command_wr && mmio_region;
            if (command_req && command_ready && !command_wr && mmio_region)
                mmio_rdata <= control_word;
            if (ext_req && ext_ready && !ext_wr)
                read_owner <= core_ext_req ? 2'd1 : 2'd2;
            if (control_write) begin
                case (control_offset)
                    0: dma_ext_base[7:0] <= command_wdata[7:0];
                    1: dma_ext_base[15:8] <= command_wdata[7:0];
                    2: dma_ext_base[23:16] <= command_wdata[7:0];
                    4: dma_sram_base[7:0] <= command_wdata[7:0];
                    5: dma_sram_base[15:8] <= command_wdata[7:0];
                    6: dma_sram_base[23:16] <= command_wdata[7:0];
                    8: dma_length[7:0] <= command_wdata[7:0];
                    9: dma_length[15:8] <= command_wdata[7:0];
                    10: dma_length[23:16] <= command_wdata[7:0];
                    11: dma_length[31:24] <= command_wdata[7:0];
                    12: dma_to_sram <= command_wdata[0];
                    13: if (command_wdata[0]) begin
                        if (engine_busy && (dma_sram_base < {8'b0,engine_live_limit} ||
                            {9'b0,dma_sram_base}+{1'b0,dma_length} > 33'd32768))
                            guard_error <= 8'd9;
                        else begin dma_start <= 1; guard_error <= 0; end
                    end
                    14: engine_live_limit[7:0] <= command_wdata[7:0];
                    15: engine_live_limit[15:8] <= command_wdata[7:0];
                    default: ;
                endcase
            end
        end
    end
endmodule
