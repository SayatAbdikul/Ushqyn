// UART Transmitter Module
// Matches uart_rx.sv interface for bidirectional UART communication
// 8N1 format: 8 data bits, no parity, 1 stop bit
//
// Interface:
//   tx_data:  Byte to transmit
//   tx_valid: Pulse high for one cycle to start transmission
//   tx_ready: High when transmitter is idle and ready for new data
//   tx_o:     Serial output line

module uart_tx #(
    parameter CLK_FREQ = 27_000_000,
    parameter BAUD_RATE = 115200
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [7:0] tx_data,
    input  logic tx_valid,
    output logic tx_ready,
    output logic tx_o
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    localparam CNT_WIDTH    = $clog2(CLKS_PER_BIT);

    typedef enum logic [1:0] {IDLE, START, DATA, STOP} state_t;
    state_t state;

    logic [CNT_WIDTH-1:0] clk_cnt;
    logic [2:0] bit_cnt;
    logic [7:0] shift_reg;

    // TX output and ready signals
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            clk_cnt <= 0;
            bit_cnt <= 0;
            shift_reg <= 8'hFF;
            tx_o <= 1'b1;       // Idle high
            tx_ready <= 1'b1;   // Ready to transmit
        end else begin
            case (state)
                IDLE: begin
                    tx_o <= 1'b1;       // Idle high
                    tx_ready <= 1'b1;   // Ready for new data
                    clk_cnt <= 0;
                    bit_cnt <= 0;

                    if (tx_valid && tx_ready) begin
                        shift_reg <= tx_data;
                        tx_ready <= 1'b0;
                        state <= START;
                    end
                end

                START: begin
                    // Send start bit (low)
                    tx_o <= 1'b0;
                    tx_ready <= 1'b0;

                    if (clk_cnt == CNT_WIDTH'(CLKS_PER_BIT - 1)) begin
                        clk_cnt <= 0;
                        state <= DATA;
                    end else begin
                        clk_cnt <= clk_cnt + CNT_WIDTH'(1);
                    end
                end

                DATA: begin
                    // Send data bits LSB first
                    tx_o <= shift_reg[0];
                    tx_ready <= 1'b0;

                    if (clk_cnt == CNT_WIDTH'(CLKS_PER_BIT - 1)) begin
                        clk_cnt <= 0;
                        shift_reg <= {1'b1, shift_reg[7:1]};  // Shift right, fill with 1

                        if (bit_cnt == 7) begin
                            bit_cnt <= 0;
                            state <= STOP;
                        end else begin
                            bit_cnt <= bit_cnt + 3'd1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + CNT_WIDTH'(1);
                    end
                end

                STOP: begin
                    // Send stop bit (high)
                    tx_o <= 1'b1;
                    tx_ready <= 1'b0;

                    if (clk_cnt == CNT_WIDTH'(CLKS_PER_BIT - 1)) begin
                        clk_cnt <= 0;
                        state <= IDLE;
                    end else begin
                        clk_cnt <= clk_cnt + CNT_WIDTH'(1);
                    end
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
