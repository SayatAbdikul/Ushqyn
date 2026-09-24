// Single-outstanding 64-bit external-memory port backed by Gowin SDRAM HS.
// Two 32-bit SDRAM words form one DMA beat. A two-word command is a real SDRAM
// burst; refresh is prioritized before accepting another beat. The controller
// IP is generated locally from hardware/phase4_sdram/sdram_controller_hs.ipc.
module v2_hs_sdram_port (
    input logic clk, clk_sdram, rst_n,
    input logic ext_req, ext_wr,
    input logic [23:0] ext_addr,
    input logic [63:0] ext_wdata,
    input logic [7:0] ext_wstrb,
    output logic ext_ready, ext_rvalid,
    output logic [63:0] ext_rdata,
    output logic init_done, refresh_deadline_missed,
    output logic [7:0] debug_status,
    output logic O_sdram_clk, O_sdram_cke, O_sdram_cs_n,
    output logic O_sdram_cas_n, O_sdram_ras_n, O_sdram_wen_n,
    output logic [3:0] O_sdram_dqm,
    output logic [10:0] O_sdram_addr,
    output logic [1:0] O_sdram_ba,
    inout wire [31:0] IO_sdram_dq
);
    localparam logic [2:0] CMD_ACT = 3'b011;
    localparam logic [2:0] CMD_READ = 3'b101;
    localparam logic [2:0] CMD_WRITE = 3'b100;
    localparam logic [2:0] CMD_REFRESH = 3'b001;
    typedef enum logic [2:0] {IDLE, ACT_WAIT, RW_WAIT, REF_WAIT,
                              READ_FINISH, SETTLE} state_t;
    state_t state;
    logic cmd_en, cmd_ack, precharge;
    logic [2:0] cmd;
    logic [20:0] word_addr;
    logic [7:0] data_len;
    logic [31:0] read_data32;
    logic [63:0] write_data64;
    logic [7:0] write_strobes;
    logic write_operation;
    logic [31:0] first_read_word;
    logic [3:0] cycle_count;
    logic ack_seen;
    logic ack_low;
    logic [3:0] settle_count;
    logic refresh_req, refresh_ack;
    logic [12:0] pending_refreshes;
    wire [31:0] write_data32 = cycle_count == 0 ? write_data64[31:0] :
                                 write_data64[63:32];
    wire [3:0] write_dqm = cycle_count == 0 ? ~write_strobes[3:0] :
                            ~write_strobes[7:4];
    assign ext_ready = init_done && state == IDLE && !refresh_req;
    assign debug_status = {init_done,cmd_ack,cmd_en,write_operation,
                           state,refresh_req};
    v2_sdram_refresh scheduler (
        .clk(clk), .rst_n(rst_n), .init_done(init_done),
        .controller_idle(state == IDLE), .refresh_ack(refresh_ack),
        .refresh_req(refresh_req), .pending_refreshes(pending_refreshes),
        .refresh_deadline_missed(refresh_deadline_missed)
    );
    SDRAM_Controller_HS_Top controller (
        .O_sdram_clk(O_sdram_clk), .O_sdram_cke(O_sdram_cke),
        .O_sdram_cs_n(O_sdram_cs_n), .O_sdram_cas_n(O_sdram_cas_n),
        .O_sdram_ras_n(O_sdram_ras_n), .O_sdram_wen_n(O_sdram_wen_n),
        .O_sdram_dqm(O_sdram_dqm), .O_sdram_addr(O_sdram_addr),
        .O_sdram_ba(O_sdram_ba), .IO_sdram_dq(IO_sdram_dq),
        .I_sdrc_rst_n(rst_n), .I_sdrc_clk(clk), .I_sdram_clk(clk_sdram),
        .I_sdrc_cmd_en(cmd_en), .I_sdrc_cmd(cmd),
        .I_sdrc_precharge_ctrl(precharge),
        .I_sdram_power_down(1'b0), .I_sdram_selfrefresh(1'b0),
        .I_sdrc_addr(word_addr), .I_sdrc_dqm(write_operation ? write_dqm : 4'b0),
        .I_sdrc_data(write_data32), .I_sdrc_data_len(data_len),
        .O_sdrc_data(read_data32), .O_sdrc_init_done(init_done),
        .O_sdrc_cmd_ack(cmd_ack)
    );
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            cmd_en <= 0; cmd <= 3'b111; precharge <= 1;
            word_addr <= 0; data_len <= 1;
            write_data64 <= 0; write_strobes <= 0;
            write_operation <= 0; cycle_count <= 0; ack_seen <= 0;
            ack_low <= 0;
            settle_count <= 0;
            first_read_word <= 0; ext_rdata <= 0; ext_rvalid <= 0;
            refresh_ack <= 0;
        end else begin
            cmd_en <= 0;
            ext_rvalid <= 0;
            refresh_ack <= 0;
            case (state)
                IDLE: if (init_done) begin
                    if (refresh_req) begin
                        cmd <= CMD_REFRESH; cmd_en <= 1;
                        refresh_ack <= 1; state <= REF_WAIT;
                    end else if (ext_req && ext_ready) begin
                        word_addr <= ext_addr[22:2];
                        write_data64 <= ext_wdata;
                        write_strobes <= ext_wstrb;
                        write_operation <= ext_wr;
                        cmd <= CMD_ACT; cmd_en <= 1;
                        state <= ACT_WAIT;
                    end
                end
                ACT_WAIT: if (cmd_ack) begin
                    cmd <= write_operation ? CMD_WRITE : CMD_READ;
                    cmd_en <= 1; cycle_count <= 0; ack_seen <= 0;
                    ack_low <= 0;
                    state <= RW_WAIT;
                end
                RW_WAIT: begin
                    if (cycle_count != 15) cycle_count <= cycle_count + 1'b1;
                    if (!cmd_ack) ack_low <= 1;
                    if (cmd_ack && ack_low) ack_seen <= 1;
                    if (write_operation) begin
                        if (cmd_ack && cycle_count >= 3) begin
                            settle_count <= 0;
                            state <= SETTLE;
                        end
                    end else begin
                        if (cycle_count == 4) first_read_word <= read_data32;
                        if (cycle_count == 5) begin
                            ext_rdata <= {read_data32, first_read_word};
                            ext_rvalid <= 1;
                            state <= READ_FINISH;
                        end
                    end
                end
                READ_FINISH: if (ack_seen || (cmd_ack && ack_low)) begin
                    settle_count <= 0;
                    state <= SETTLE;
                end
                SETTLE: if (settle_count == 15) state <= IDLE;
                        else settle_count <= settle_count + 1'b1;
                REF_WAIT: if (cmd_ack) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
