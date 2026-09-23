// Board-only clock/reset/pin/UART wrapper. Compute/control live only in rtl/v2.
module tang_nano_v2 #(
    parameter integer CLOCK_HZ=target_pkg::CLOCK_HZ,
    parameter integer BAUD=target_pkg::BAUD
)(
    input logic sys_clk,reset_button,uart_rx,
    output logic uart_tx,
    output logic [5:0] led
);
    // Tang Nano 20K KEY1 (pin 88) is HIGH while pressed. Initialize for
    // configuration startup; assert reset asynchronously and release on clk.
    logic [2:0] reset_sync = 3'b000;
    always_ff @(posedge sys_clk or posedge reset_button) begin
        if(reset_button)reset_sync<=0;else reset_sync<={reset_sync[1:0],1'b1};
    end
    wire rst_n=reset_sync[2];
    logic rv,tv,tr,busy,marker;
    logic [7:0] rd,td,error_code;
    uart_rx #(.CLK_FREQ(CLOCK_HZ),.BAUD_RATE(BAUD)) receiver(.clk(sys_clk),.rst_n(rst_n),.rx_i(uart_rx),.rx_data(rd),.rx_valid(rv));
    uart_tx #(.CLK_FREQ(CLOCK_HZ),.BAUD_RATE(BAUD)) transmitter(.clk(sys_clk),.rst_n(rst_n),.tx_data(td),.tx_valid(tv),.tx_ready(tr),.tx_o(uart_tx));
    v2_system system(.clk(sys_clk),.rst_n(rst_n),.rx_valid(rv),.rx_data(rd),.tx_valid(tv),.tx_data(td),.tx_ready(tr),.busy(busy),.layer_marker(marker),.error_code(error_code));
    assign led=~{error_code!=0,marker,3'b0,busy};
endmodule
