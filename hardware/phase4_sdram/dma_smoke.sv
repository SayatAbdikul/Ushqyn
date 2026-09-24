// Board-level Phase 4 DMA path: v2_tiled_core scratchpad and tile DMA ->
// v2_hs_sdram_port -> embedded SDRAM. The engine is present but idle.
// Case 0 moves a full 32-KiB tile ending at the highest SDRAM byte. Case 1 moves
// 13 bytes across the 2-MiB bank boundary and checks the partial tail.
module phase4_dma_smoke (
    input logic sys_clk, reset_button,
    output logic uart_tx_pin,
    output logic [5:0] led,
    output logic O_sdram_clk, O_sdram_cke, O_sdram_cs_n,
    output logic O_sdram_cas_n, O_sdram_ras_n, O_sdram_wen_n,
    output logic [3:0] O_sdram_dqm,
    output logic [10:0] O_sdram_addr,
    output logic [1:0] O_sdram_ba,
    inout wire [31:0] IO_sdram_dq
);
    logic clk, clk_sdram, lock;
    phase4_dma_pll pll(.clkin(sys_clk), .clk(clk),
                       .clk_sdram(clk_sdram), .lock(lock));
    logic [2:0] reset_sync = 0;
    always_ff @(posedge clk or posedge reset_button) begin
        if (reset_button) reset_sync <= 0;
        else reset_sync <= {reset_sync[1:0], 1'b1};
    end
    wire rst_n = lock && reset_sync[2];

    typedef enum logic [3:0] {
        WAIT_INIT, FILL, DMA_OUT_START, DMA_OUT_WAIT, CLEAR,
        DMA_IN_START, DMA_IN_WAIT, VERIFY_REQ, VERIFY_WAIT,
        REPORT, REPORT_PULSE, REPORT_TXWAIT
    } state_t;
    state_t state;
    logic case_id;
    logic [11:0] word_index;
    logic passed, failed;
    logic [7:0] failure_reason;
    logic [31:0] watchdog, report_count;
    logic [4:0] report_index;
    logic sending;
    logic [31:0] dma_cycle_count;
    logic [31:0] out_cycles_0, in_cycles_0, out_cycles_1, in_cycles_1;
    logic tx_valid, tx_ready;
    logic [7:0] tx_data;
    wire [11:0] last_word = case_id ? 12'd1 : 12'd4095;
    wire [31:0] transfer_length = case_id ? 32'd13 : 32'd32768;
    wire [23:0] transfer_ext_base = case_id ? 24'h1ffff8 : 24'h7f8000;

    function automatic logic [63:0] pattern(input logic which,
                                             input logic [11:0] index);
        pattern = 64'h0123456789abcdef ^ {8{index[7:0]}} ^
                  {52'b0,index} ^
                  (which ? 64'hf0e1d2c3b4a59687 : 64'h0000000000000000);
    endfunction
    function automatic logic [63:0] expected_word(input logic which,
                                                   input logic [11:0] index);
        expected_word = pattern(which, index);
        if (which && index == 1)
            expected_word = expected_word & 64'h000000ffffffffff;
    endfunction

    logic host_req, host_wr, host_ready, host_rvalid;
    logic [23:0] host_addr;
    logic [63:0] host_wdata, host_rdata;
    logic [7:0] host_wstrb;
    logic dma_busy, dma_done;
    logic [7:0] dma_error;
    logic [31:0] dma_bytes_copied;
    logic ext_req, ext_wr, ext_ready, ext_rvalid;
    logic [23:0] ext_addr;
    logic [63:0] ext_wdata, ext_rdata;
    logic [7:0] ext_wstrb;
    logic engine_busy, engine_done;
    logic [7:0] engine_error;
    logic [31:0] engine_elapsed;
    logic init_done, refresh_missed;

    assign host_req = state == FILL || state == CLEAR || state == VERIFY_REQ;
    assign host_wr = state == FILL || state == CLEAR;
    assign host_addr = {9'b0,word_index,3'b0};
    assign host_wdata = state == CLEAR ? 64'b0 : pattern(case_id, word_index);
    assign host_wstrb = 8'hff;
    assign led = ~{failed | refresh_missed, passed, case_id,
                   word_index[2:0]};
    v2_tiled_core core (
        .clk(clk), .rst_n(rst_n),
        .engine_start(1'b0), .engine_abort(1'b0),
        .engine_start_pc(24'b0), .engine_busy(engine_busy),
        .engine_done(engine_done), .engine_error(engine_error),
        .engine_elapsed(engine_elapsed),
        .dma_start(state == DMA_OUT_START || state == DMA_IN_START),
        .dma_abort(1'b0), .dma_to_sram(state == DMA_IN_START ||
                                      state == DMA_IN_WAIT),
        .dma_sram_base(24'b0), .dma_ext_base(transfer_ext_base),
        .dma_length(transfer_length), .dma_busy(dma_busy),
        .dma_done(dma_done), .dma_error(dma_error),
        .dma_bytes_copied(dma_bytes_copied),
        .host_req(host_req), .host_wr(host_wr), .host_addr(host_addr),
        .host_wdata(host_wdata), .host_wstrb(host_wstrb),
        .host_ready(host_ready), .host_rvalid(host_rvalid),
        .host_rdata(host_rdata),
        .ext_req(ext_req), .ext_wr(ext_wr), .ext_addr(ext_addr),
        .ext_wdata(ext_wdata), .ext_wstrb(ext_wstrb),
        .ext_ready(ext_ready), .ext_rvalid(ext_rvalid),
        .ext_rdata(ext_rdata)
    );
    v2_hs_sdram_port #(.CLOCK_HZ(20250000)) memory (
        .clk(clk), .clk_sdram(clk_sdram), .rst_n(rst_n),
        .ext_req(ext_req), .ext_wr(ext_wr), .ext_addr(ext_addr),
        .ext_wdata(ext_wdata), .ext_wstrb(ext_wstrb),
        .ext_ready(ext_ready), .ext_rvalid(ext_rvalid),
        .ext_rdata(ext_rdata), .init_done(init_done),
        .refresh_deadline_missed(refresh_missed), .debug_status(),
        .O_sdram_clk(O_sdram_clk), .O_sdram_cke(O_sdram_cke),
        .O_sdram_cs_n(O_sdram_cs_n), .O_sdram_cas_n(O_sdram_cas_n),
        .O_sdram_ras_n(O_sdram_ras_n), .O_sdram_wen_n(O_sdram_wen_n),
        .O_sdram_dqm(O_sdram_dqm), .O_sdram_addr(O_sdram_addr),
        .O_sdram_ba(O_sdram_ba), .IO_sdram_dq(IO_sdram_dq)
    );
    always_comb begin
        tx_data = 0;
        if (failed) begin
            if (report_index == 0) tx_data = 8'hed;
            else tx_data = failure_reason;
        end else case (report_index)
            0: tx_data = 8'hbd;
            1: tx_data = 8'd2;
            2,3,4,5: tx_data = out_cycles_0[(report_index-2)*8 +: 8];
            6,7,8,9: tx_data = in_cycles_0[(report_index-6)*8 +: 8];
            10,11,12,13: tx_data = out_cycles_1[(report_index-10)*8 +: 8];
            default: tx_data = in_cycles_1[(report_index-14)*8 +: 8];
        endcase
    end
    uart_tx #(.CLK_FREQ(20250000), .BAUD_RATE(115200)) tx (
        .clk(clk), .rst_n(rst_n), .tx_data(tx_data),
        .tx_valid(tx_valid), .tx_ready(tx_ready), .tx_o(uart_tx_pin)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= WAIT_INIT; case_id <= 0; word_index <= 0;
            passed <= 0; failed <= 0; failure_reason <= 0;
            watchdog <= 0; report_count <= 0; report_index <= 0;
            sending <= 0; tx_valid <= 0;
            dma_cycle_count <= 0;
            out_cycles_0 <= 0; in_cycles_0 <= 0;
            out_cycles_1 <= 0; in_cycles_1 <= 0;
        end else begin
            tx_valid <= 0;
            if (state != REPORT && state != REPORT_PULSE &&
                state != REPORT_TXWAIT) watchdog <= watchdog + 1'b1;
            case (state)
                WAIT_INIT: if (init_done) state <= FILL;
                FILL: if (host_ready) begin
                    if (word_index == last_word) begin
                        word_index <= 0; state <= DMA_OUT_START;
                    end else word_index <= word_index + 1'b1;
                end
                DMA_OUT_START: begin
                    dma_cycle_count <= 0; state <= DMA_OUT_WAIT;
                end
                DMA_OUT_WAIT: begin
                    dma_cycle_count <= dma_cycle_count + 1'b1;
                    if (dma_done) begin
                    if (case_id) out_cycles_1 <= dma_cycle_count + 1'b1;
                    else out_cycles_0 <= dma_cycle_count + 1'b1;
                    if (dma_error != 0 || dma_bytes_copied != transfer_length) begin
                        failed <= 1; failure_reason <= 1; state <= REPORT;
                    end else state <= CLEAR;
                    end
                end
                CLEAR: if (host_ready) begin
                    if (word_index == last_word) begin
                        word_index <= 0; state <= DMA_IN_START;
                    end else word_index <= word_index + 1'b1;
                end
                DMA_IN_START: begin
                    dma_cycle_count <= 0; state <= DMA_IN_WAIT;
                end
                DMA_IN_WAIT: begin
                    dma_cycle_count <= dma_cycle_count + 1'b1;
                    if (dma_done) begin
                    if (case_id) in_cycles_1 <= dma_cycle_count + 1'b1;
                    else in_cycles_0 <= dma_cycle_count + 1'b1;
                    if (dma_error != 0 || dma_bytes_copied != transfer_length) begin
                        failed <= 1; failure_reason <= 2; state <= REPORT;
                    end else state <= VERIFY_REQ;
                    end
                end
                VERIFY_REQ: if (host_ready) state <= VERIFY_WAIT;
                VERIFY_WAIT: if (host_rvalid) begin
                    if (host_rdata != expected_word(case_id, word_index)) begin
                        failed <= 1; failure_reason <= 3; state <= REPORT;
                    end else if (word_index == last_word) begin
                        if (case_id) begin passed <= 1; state <= REPORT; end
                        else begin
                            case_id <= 1; word_index <= 0; state <= FILL;
                        end
                    end else begin
                        word_index <= word_index + 1'b1; state <= VERIFY_REQ;
                    end
                end
                REPORT: if (sending || report_count == 32'd20249999) begin
                    if (tx_ready) begin
                        tx_valid <= 1; sending <= 1; state <= REPORT_PULSE;
                    end
                end else report_count <= report_count + 1'b1;
                REPORT_PULSE: state <= REPORT_TXWAIT;
                REPORT_TXWAIT: if (tx_ready) begin
                    if ((failed && report_index == 1) ||
                        (!failed && report_index == 17)) begin
                        report_index <= 0; sending <= 0; report_count <= 0;
                    end else report_index <= report_index + 1'b1;
                    state <= REPORT;
                end
                default: state <= WAIT_INIT;
            endcase
            if ((watchdog >= 32'd20250000 || refresh_missed) && !failed &&
                !passed) begin
                failed <= 1;
                failure_reason <= refresh_missed ? 8'd5 : 8'd4;
                state <= REPORT;
            end
        end
    end
endmodule
