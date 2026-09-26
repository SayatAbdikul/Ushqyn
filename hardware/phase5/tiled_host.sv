// Full host-addressable tile datapath for the Tang Nano 20K.
// Post-route candidate only until physically programmed and tested.
module phase5_tiled_host (
    input logic sys_clk, reset_button, uart_rx_pin,
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
    logic [7:0] rx_data, tx_data;
    logic rx_valid, tx_valid, tx_ready;
    logic ext_req, ext_wr, ext_ready, ext_rvalid;
    logic [23:0] ext_addr;
    logic [63:0] ext_wdata, ext_rdata;
    logic [7:0] ext_wstrb;
    logic init_done, refresh_missed, engine_busy, dma_busy, memory_port_busy;
    uart_rx #(.CLK_FREQ(20250000), .BAUD_RATE(750000)) receiver (
        .clk(clk), .rst_n(rst_n), .rx_i(uart_rx_pin),
        .rx_data(rx_data), .rx_valid(rx_valid)
    );
    uart_tx #(.CLK_FREQ(20250000), .BAUD_RATE(750000)) transmitter (
        .clk(clk), .rst_n(rst_n), .tx_data(tx_data),
        .tx_valid(tx_valid), .tx_ready(tx_ready), .tx_o(uart_tx_pin)
    );
    v2_tiled_host_bridge host (
        .clk(clk), .rst_n(rst_n),
        .rx_valid(rx_valid), .rx_data(rx_data),
        .tx_valid(tx_valid), .tx_data(tx_data), .tx_ready(tx_ready),
        .ext_req(ext_req), .ext_wr(ext_wr), .ext_addr(ext_addr),
        .ext_wdata(ext_wdata), .ext_wstrb(ext_wstrb),
        .ext_ready(ext_ready), .ext_rvalid(ext_rvalid),
        .ext_rdata(ext_rdata), .memory_initialized(init_done),
        .memory_port_busy(memory_port_busy),
        .engine_busy(engine_busy), .dma_busy(dma_busy)
    );
    v2_hs_sdram_burst_port #(.CLOCK_HZ(20250000)) memory (
        .clk(clk), .clk_sdram(clk_sdram), .rst_n(rst_n),
        .ext_req(ext_req), .ext_wr(ext_wr), .ext_addr(ext_addr),
        .ext_wdata(ext_wdata), .ext_wstrb(ext_wstrb),
        .ext_ready(ext_ready), .ext_rvalid(ext_rvalid),
        .ext_rdata(ext_rdata), .port_busy(memory_port_busy),
        .init_done(init_done),
        .refresh_deadline_missed(refresh_missed), .debug_status(),
        .O_sdram_clk(O_sdram_clk), .O_sdram_cke(O_sdram_cke),
        .O_sdram_cs_n(O_sdram_cs_n), .O_sdram_cas_n(O_sdram_cas_n),
        .O_sdram_ras_n(O_sdram_ras_n), .O_sdram_wen_n(O_sdram_wen_n),
        .O_sdram_dqm(O_sdram_dqm), .O_sdram_addr(O_sdram_addr),
        .O_sdram_ba(O_sdram_ba), .IO_sdram_dq(IO_sdram_dq)
    );
    assign led = ~{refresh_missed, init_done, engine_busy, dma_busy,
                   ext_req, ext_rvalid};
endmodule
