// Minimal phase-0 lab image. Stop-and-wait host protocol: send byte, read echo.
module uart_loopback (
    input logic sys_clk, reset_button, uart_rx,
    output logic uart_tx,
    output logic [5:0] led
);
    logic [7:0] rx_data;
    logic rx_valid, tx_ready;
    // KEY1 is active HIGH. Synchronize reset release, including power-up.
    logic [2:0] reset_sync = 3'b000;
    always_ff @(posedge sys_clk or posedge reset_button) begin
        if (reset_button) reset_sync <= 0;
        else reset_sync <= {reset_sync[1:0], 1'b1};
    end
    wire rst_n = reset_sync[2];
    uart_rx receiver(.clk(sys_clk), .rst_n(rst_n), .rx_i(uart_rx),
                     .rx_data(rx_data), .rx_valid(rx_valid));
    uart_tx transmitter(.clk(sys_clk), .rst_n(rst_n), .tx_data(rx_data),
                        .tx_valid(rx_valid && tx_ready), .tx_ready(tx_ready), .tx_o(uart_tx));
    assign led = ~rx_data[5:0];
endmodule
