// One-byte diagnostic bridge for Gowin SDRAM Controller HS.
// This first validates controller initialization, ACT, read/write and refresh
// on the board. It is intentionally not a burst DMA adapter.
module phase4_hs_byte_adapter (
    input logic clk, clk_sdram, resetn,
    input logic rd, wr, refresh,
    input logic [22:0] addr,
    input logic [7:0] din,
    output logic [7:0] dout,
    output logic [31:0] dout32,
    output logic data_ready, busy,
    output logic SDRAM_CLK, SDRAM_CKE, SDRAM_nCS,
    output logic SDRAM_nCAS, SDRAM_nRAS, SDRAM_nWE,
    output logic [3:0] SDRAM_DQM,
    output logic [10:0] SDRAM_A,
    output logic [1:0] SDRAM_BA,
    inout wire [31:0] SDRAM_DQ
);
    localparam logic [2:0] CMD_ACT = 3'b011;
    localparam logic [2:0] CMD_READ = 3'b101;
    localparam logic [2:0] CMD_WRITE = 3'b100;
    localparam logic [2:0] CMD_REFRESH = 3'b001;
    typedef enum logic [2:0] {
        IDLE, ACT_WAIT, OP_WAIT, READ_CAPTURE, REF_WAIT
    } state_t;
    state_t state;
    logic cmd_en, precharge;
    logic [2:0] cmd;
    logic [20:0] word_addr;
    logic [3:0] dqm;
    logic [31:0] write_data, read_data;
    logic [7:0] data_len;
    logic init_done, cmd_ack, operation_read;
    logic [1:0] byte_offset;
    logic [3:0] wait_count;
    assign dout32 = read_data;
    assign busy = !init_done || state != IDLE;

    SDRAM_Controller_HS_Top controller (
        .O_sdram_clk(SDRAM_CLK), .O_sdram_cke(SDRAM_CKE),
        .O_sdram_cs_n(SDRAM_nCS), .O_sdram_cas_n(SDRAM_nCAS),
        .O_sdram_ras_n(SDRAM_nRAS), .O_sdram_wen_n(SDRAM_nWE),
        .O_sdram_dqm(SDRAM_DQM), .O_sdram_addr(SDRAM_A),
        .O_sdram_ba(SDRAM_BA), .IO_sdram_dq(SDRAM_DQ),
        .I_sdrc_rst_n(resetn), .I_sdrc_clk(clk), .I_sdram_clk(clk_sdram),
        .I_sdrc_cmd_en(cmd_en), .I_sdrc_cmd(cmd),
        .I_sdrc_precharge_ctrl(precharge),
        .I_sdram_power_down(1'b0), .I_sdram_selfrefresh(1'b0),
        .I_sdrc_addr(word_addr), .I_sdrc_dqm(dqm),
        .I_sdrc_data(write_data), .I_sdrc_data_len(data_len),
        .O_sdrc_data(read_data), .O_sdrc_init_done(init_done),
        .O_sdrc_cmd_ack(cmd_ack)
    );

    always_ff @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            state <= IDLE; cmd_en <= 0; cmd <= 3'b111;
            word_addr <= 0; dqm <= 0; write_data <= 0; data_len <= 0;
            precharge <= 1; byte_offset <= 0; operation_read <= 0;
            wait_count <= 0; dout <= 0; data_ready <= 0;
        end else begin
            cmd_en <= 0;
            data_ready <= 0;
            case (state)
                IDLE: if (init_done) begin
                    if (refresh) begin
                        cmd <= CMD_REFRESH; cmd_en <= 1;
                        state <= REF_WAIT;
                    end else if (rd || wr) begin
                        word_addr <= addr[22:2];
                        byte_offset <= addr[1:0];
                        write_data <= {4{din}};
                        dqm <= wr ? ~(4'b0001 << addr[1:0]) : 4'b0000;
                        operation_read <= rd;
                        precharge <= 1;
                        data_len <= 0;
                        cmd <= CMD_ACT; cmd_en <= 1;
                        state <= ACT_WAIT;
                    end
                end
                ACT_WAIT: if (cmd_ack) begin
                    cmd <= operation_read ? CMD_READ : CMD_WRITE;
                    cmd_en <= 1;
                    wait_count <= 0;
                    state <= OP_WAIT;
                end
                OP_WAIT: begin
                    if (wait_count != 15) wait_count <= wait_count + 1'b1;
                    if (operation_read) begin
                        if (wait_count == 5) state <= READ_CAPTURE;
                    end else if (cmd_ack && wait_count >= 3) state <= IDLE;
                end
                READ_CAPTURE: begin
                    case (byte_offset)
                        0: dout <= read_data[7:0];
                        1: dout <= read_data[15:8];
                        2: dout <= read_data[23:16];
                        default: dout <= read_data[31:24];
                    endcase
                    data_ready <= 1;
                    state <= IDLE;
                end
                REF_WAIT: if (cmd_ack) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
