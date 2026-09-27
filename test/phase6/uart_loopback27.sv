// Exact 27 MHz / 750 kbaud UART PHY check for the routed Phase 6 host.
module uart_loopback27 (
    input logic clk, rst_n, rx_i,
    output logic tx_o,
    output logic [7:0] received,
    output logic received_valid
);
    logic [7:0] rx_data;
    logic rx_valid, tx_ready;
    logic [7:0] tx_data;
    logic tx_valid;
    uart_rx #(.CLK_FREQ(27000000), .BAUD_RATE(750000)) rx (
        .clk(clk), .rst_n(rst_n), .rx_i(rx_i),
        .rx_data(rx_data), .rx_valid(rx_valid));
    uart_tx #(.CLK_FREQ(27000000), .BAUD_RATE(750000)) tx (
        .clk(clk), .rst_n(rst_n), .tx_data(tx_data),
        .tx_valid(tx_valid), .tx_ready(tx_ready), .tx_o(tx_o));
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            received <= 0;
            received_valid <= 0;
            tx_data <= 0;
            tx_valid <= 0;
        end else begin
            received_valid <= rx_valid;
            tx_valid <= rx_valid && tx_ready;
            if (rx_valid) begin
                received <= rx_data;
                tx_data <= rx_data;
            end
        end
    end
endmodule
