module uart_rx #(
    parameter CLK_FREQ = 27_000_000,
    parameter BAUD_RATE = 115200
)(
    input  logic clk,
    input  logic rst_n,
    input  logic rx_i,
    output logic [7:0] rx_data,
    output logic rx_valid
);

    localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    localparam CNT_WIDTH    = $clog2(CLKS_PER_BIT);

    typedef enum logic [1:0] {IDLE, WAIT_HALF, READ, STOP} state_t;
    state_t state;

    logic [CNT_WIDTH-1:0] clk_cnt;
    logic [2:0] bit_cnt;
    logic [7:0] shift_reg;

    // Sync RX input to avoid metastability
    logic rx_sync_1, rx_sync_2;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rx_sync_1 <= 1'b1;
            rx_sync_2 <= 1'b1;
        end else begin
            rx_sync_1 <= rx_i;
            rx_sync_2 <= rx_sync_1;
        end
    end
    wire rx = rx_sync_2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            clk_cnt <= 0;
            bit_cnt <= 0;
            shift_reg <= 0;
            rx_valid <= 0;
            rx_data <= 0;
        end else begin
            rx_valid <= 0;

            case (state)
                IDLE: begin
                    if (!rx) begin // Start bit detection (falling edge)
                        state <= WAIT_HALF;
                        clk_cnt <= 0;
                    end
                end

                WAIT_HALF: begin
                    if (clk_cnt == CNT_WIDTH'((CLKS_PER_BIT / 2) - 1)) begin // Wait for middle of start bit
                        if (!rx) begin // Confirm it's still low
                            state <= READ;
                            clk_cnt <= 0;
                            bit_cnt <= 0;
                        end else begin
                            state <= IDLE; // False alarm
                        end
                    end else begin
                        clk_cnt <= clk_cnt + CNT_WIDTH'(1);
                    end
                end

                READ: begin
                    if (clk_cnt == CNT_WIDTH'(CLKS_PER_BIT - 1)) begin
                        clk_cnt <= 0;
                        shift_reg <= {rx, shift_reg[7:1]}; // LSB first
                        if (bit_cnt == 7) begin
                            state <= STOP;
                        end else begin
                            bit_cnt <= bit_cnt + 3'd1;
                        end
                    end else begin
                        clk_cnt <= clk_cnt + CNT_WIDTH'(1);
                    end
                end

                STOP: begin
                    if (clk_cnt == CNT_WIDTH'(CLKS_PER_BIT - 1)) begin
                        state <= IDLE;
                        rx_data <= shift_reg;
                        rx_valid <= rx; // Reject a low stop bit (framing error)
                    end else begin
                        clk_cnt <= clk_cnt + CNT_WIDTH'(1);
                    end
                end
            endcase
        end
    end

endmodule
