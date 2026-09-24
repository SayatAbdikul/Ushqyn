// Physical two-word bursts with a distinct-half seed, 64 walking-one data
// patterns, then row, bank and top-of-memory boundary cases.
// UART repeats B9,73,64 on success, or EA,index,reason,actual[63:0],expected[63:0] on
// mismatch, once per second at 115200 baud.
module phase4_burst_smoke (
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
    phase4_sdram_pll pll(.clkin(sys_clk), .clk(clk),
                         .clk_sdram(clk_sdram), .lock(lock));
    logic [2:0] reset_sync = 0;
    always_ff @(posedge clk or posedge reset_button) begin
        if (reset_button) reset_sync <= 0;
        else reset_sync <= {reset_sync[1:0],1'b1};
    end
    wire rst_n = lock && reset_sync[2];
    logic ext_req, ext_wr, ext_ready, ext_rvalid, init_done, refresh_missed;
    logic [7:0] memory_debug;
    logic [23:0] ext_addr;
    logic [63:0] ext_wdata, ext_rdata;
    v2_hs_sdram_port memory (
        .clk(clk), .clk_sdram(clk_sdram), .rst_n(rst_n),
        .ext_req(ext_req), .ext_wr(ext_wr), .ext_addr(ext_addr),
        .ext_wdata(ext_wdata), .ext_wstrb(8'hff),
        .ext_ready(ext_ready), .ext_rvalid(ext_rvalid),
        .ext_rdata(ext_rdata), .init_done(init_done),
        .refresh_deadline_missed(refresh_missed),
        .debug_status(memory_debug),
        .O_sdram_clk(O_sdram_clk), .O_sdram_cke(O_sdram_cke),
        .O_sdram_cs_n(O_sdram_cs_n), .O_sdram_cas_n(O_sdram_cas_n),
        .O_sdram_ras_n(O_sdram_ras_n), .O_sdram_wen_n(O_sdram_wen_n),
        .O_sdram_dqm(O_sdram_dqm), .O_sdram_addr(O_sdram_addr),
        .O_sdram_ba(O_sdram_ba), .IO_sdram_dq(IO_sdram_dq)
    );
    function automatic logic [23:0] test_address(input logic [6:0] index);
        if (index <= 64) test_address = 24'h000000;
        else case (index)
            65: test_address = 24'h000000;
            66: test_address = 24'h0003f8; // final two words of row 0
            67: test_address = 24'h000400; // first two words of row 1
            68: test_address = 24'h1ffff8; // bank 0 end
            69: test_address = 24'h200000; // bank 1 start
            70: test_address = 24'h3ffff8; // bank 1 end
            71: test_address = 24'h400000; // bank 2 start
            default: test_address = 24'h7ffff8; // bank 3 end
        endcase
    endfunction
    function automatic logic [63:0] test_data(input logic [6:0] index);
        if (index == 0) test_data = 64'h9abcdef012345678;
        else if (index <= 64) test_data = 64'h1 << (index - 1'b1);
        else test_data = 64'h0123456789abcdef ^ {8{5'b0,index[2:0]}};
    endfunction
    typedef enum logic [2:0] {WAIT_INIT, WRITE_REQ, WRITE_WAIT,
                              READ_REQ, READ_WAIT, REPORT,
                              REPORT_PULSE, REPORT_TXWAIT} state_t;
    state_t state;
    logic [6:0] index;
    logic seen_busy, passed, failed;
    logic [6:0] failed_index;
    logic [7:0] failure_reason;
    logic [63:0] failed_actual, failed_expected;
    logic [31:0] report_count;
    logic [4:0] report_index;
    logic sending;
    logic tx_ready, tx_valid;
    logic [7:0] tx_data;
    always_comb begin
        tx_data = 8'h00;
        if (!failed) begin
            case (report_index)
                0: tx_data = 8'hb9;
                1: tx_data = 8'd73;
                default: tx_data = 8'd64;
            endcase
        end else if (report_index == 0) tx_data = 8'hea;
        else case (report_index)
            1: tx_data = {1'b0,failed_index};
            2: tx_data = failure_reason;
            3: tx_data = failed_actual[7:0];
            4: tx_data = failed_actual[15:8];
            5: tx_data = failed_actual[23:16];
            6: tx_data = failed_actual[31:24];
            7: tx_data = failed_actual[39:32];
            8: tx_data = failed_actual[47:40];
            9: tx_data = failed_actual[55:48];
            10: tx_data = failed_actual[63:56];
            11: tx_data = failed_expected[7:0];
            12: tx_data = failed_expected[15:8];
            13: tx_data = failed_expected[23:16];
            14: tx_data = failed_expected[31:24];
            15: tx_data = failed_expected[39:32];
            16: tx_data = failed_expected[47:40];
            17: tx_data = failed_expected[55:48];
            default: tx_data = failed_expected[63:56];
        endcase
    end
    uart_tx #(.CLK_FREQ(27000000), .BAUD_RATE(115200)) tx (
        .clk(clk), .rst_n(rst_n), .tx_data(tx_data),
        .tx_valid(tx_valid), .tx_ready(tx_ready), .tx_o(uart_tx_pin)
    );
    assign ext_req = state == WRITE_REQ || state == READ_REQ;
    assign ext_wr = state == WRITE_REQ;
    assign ext_addr = test_address(index);
    assign ext_wdata = test_data(index);
    assign led = ~{failed | refresh_missed,passed,index[2:0],!passed};
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= WAIT_INIT; index <= 0; seen_busy <= 0;
            passed <= 0; failed <= 0; report_count <= 0; tx_valid <= 0;
            failed_index <= 0; failed_actual <= 0; failed_expected <= 0;
            failure_reason <= 0;
            report_index <= 0; sending <= 0;
        end else begin
            tx_valid <= 0;
            case (state)
                WAIT_INIT: if (init_done) state <= WRITE_REQ;
                WRITE_REQ: if (ext_ready) begin
                    seen_busy <= 0; state <= WRITE_WAIT;
                end
                WRITE_WAIT: begin
                    if (!ext_ready) seen_busy <= 1;
                    if (seen_busy && ext_ready) state <= READ_REQ;
                end
                READ_REQ: if (ext_ready) state <= READ_WAIT;
                READ_WAIT: if (ext_rvalid) begin
                    if (ext_rdata != test_data(index)) begin
                        failed <= 1;
                        failure_reason <= 1;
                        failed_index <= index;
                        failed_actual <= ext_rdata;
                        failed_expected <= test_data(index);
                        state <= REPORT;
                    end else if (index == 72) begin
                        passed <= 1; state <= REPORT;
                    end else begin
                        index <= index + 1'b1; state <= WRITE_REQ;
                    end
                end
                REPORT: if (sending || report_count == 32'd26999999) begin
                    if (tx_ready) begin
                        tx_valid <= 1;
                        sending <= 1;
                        state <= REPORT_PULSE;
                    end
                end else report_count <= report_count + 1'b1;
                REPORT_PULSE: state <= REPORT_TXWAIT;
                REPORT_TXWAIT: if (tx_ready) begin
                    if ((!failed && report_index == 2) ||
                        (failed && report_index == 18)) begin
                        report_index <= 0; sending <= 0; report_count <= 0;
                    end else report_index <= report_index + 1'b1;
                    state <= REPORT;
                end
                default: state <= WAIT_INIT;
            endcase
            if (refresh_missed && !failed) begin
                failed <= 1; passed <= 0; state <= REPORT;
                failure_reason <= 2;
                failed_index <= index;
                failed_actual <= {47'b0,memory_debug,5'b0,state,seen_busy};
                failed_expected <= 64'hfeed0000dead0001;
            end
        end
    end
endmodule
