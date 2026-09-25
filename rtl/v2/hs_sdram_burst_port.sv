// 64-byte line bursts over the documented Gowin HS command interface.
// Read-ahead and byte-masked write combining preserve the single-beat host ABI.
// A dirty write line is drained before reads, address changes, refresh or idle
// timeout. port_busy includes buffered writes, so DMA completion is ordered.
module v2_hs_sdram_burst_port #(
    parameter integer CLOCK_HZ=20250000
) (
    input logic clk, clk_sdram, rst_n,
    input logic ext_req, ext_wr,
    input logic [23:0] ext_addr,
    input logic [63:0] ext_wdata,
    input logic [7:0] ext_wstrb,
    output logic ext_ready, ext_rvalid,
    output logic [63:0] ext_rdata,
    output logic port_busy, init_done, refresh_deadline_missed,
    output logic [7:0] debug_status,
    output logic O_sdram_clk, O_sdram_cke, O_sdram_cs_n,
    output logic O_sdram_cas_n, O_sdram_ras_n, O_sdram_wen_n,
    output logic [3:0] O_sdram_dqm,
    output logic [10:0] O_sdram_addr,
    output logic [1:0] O_sdram_ba,
    inout wire [31:0] IO_sdram_dq
);
    typedef enum logic [2:0] {IDLE, ACT_WAIT, RW_WAIT, READ_RETURN,
                              READ_FINISH, SETTLE, REF_WAIT} state_t;
    state_t state;
    logic cmd_en, cmd_ack, write_operation, ack_low, ack_seen;
    logic [2:0] cmd;
    logic [20:0] word_addr;
    logic [5:0] cycle_count;
    logic [3:0] settle_count;
    logic [31:0] read_word;
    logic [31:0] read_line[0:15], write_line[0:15];
    logic [63:0] write_mask;
    logic [16:0] read_tag, write_tag;
    logic read_valid, write_dirty, flush_pending;
    logic [4:0] idle_age;
    logic [2:0] requested_beat;
    logic refresh_req, refresh_ack, refresh_deferred;
    logic [12:0] pending_refreshes;
    wire same_write_line = ext_addr[22:6]==write_tag;
    assign ext_ready = init_done && state==IDLE && !refresh_req && !refresh_deferred &&
        (!write_dirty || (ext_wr && same_write_line && !flush_pending && idle_age<15));
    assign port_busy = state!=IDLE || write_dirty || refresh_deferred;
    assign debug_status = {init_done,cmd_ack,cmd_en,write_operation,state,refresh_req};
    wire [3:0] write_index = cycle_count[3:0];
    wire [31:0] write_word = write_line[write_index];
    wire [3:0] write_dqm = ~write_mask[{write_index,2'b0}+:4];
    v2_sdram_refresh #(.CLOCK_HZ(CLOCK_HZ)) scheduler (
        .clk(clk), .rst_n(rst_n), .init_done(init_done),
        .controller_idle(state==IDLE), .refresh_ack(refresh_ack),
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
        .I_sdrc_cmd_en(cmd_en), .I_sdrc_cmd(cmd), .I_sdrc_precharge_ctrl(1'b1),
        .I_sdram_power_down(1'b0), .I_sdram_selfrefresh(1'b0),
        .I_sdrc_addr(word_addr), .I_sdrc_dqm(write_operation ? write_dqm : 4'b0),
        .I_sdrc_data(write_word), .I_sdrc_data_len(8'd15),
        .O_sdrc_data(read_word), .O_sdrc_init_done(init_done), .O_sdrc_cmd_ack(cmd_ack)
    );
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=IDLE;cmd_en<=0;cmd<=7;word_addr<=0;cycle_count<=0;
            settle_count<=0;write_operation<=0;ack_low<=0;ack_seen<=0;
            write_mask<=0;read_tag<=0;write_tag<=0;read_valid<=0;
            write_dirty<=0;flush_pending<=0;idle_age<=0;requested_beat<=0;
            ext_rvalid<=0;ext_rdata<=0;refresh_ack<=0;refresh_deferred<=0;
        end else begin
            cmd_en<=0;ext_rvalid<=0;refresh_ack<=0;
            if (refresh_req) refresh_deferred<=1;
            if (write_dirty && state==IDLE && idle_age!=31) idle_age<=idle_age+1;
            case (state)
                IDLE: if (init_done) begin
                    if (write_dirty && (refresh_req || refresh_deferred || flush_pending || idle_age>=15 ||
                        (ext_req && (!ext_wr || !same_write_line)))) begin
                        word_addr<={write_tag,4'b0};write_operation<=1;
                        cmd<=3'b011;cmd_en<=1;state<=ACT_WAIT;
                    end else if (refresh_req || refresh_deferred) begin
                        cmd<=3'b001;cmd_en<=1;refresh_ack<=1;state<=REF_WAIT;
                        refresh_deferred<=0;
                    end else if (ext_req && ext_ready) begin
                        if (ext_wr) begin
                            if (read_valid && read_tag==ext_addr[22:6]) read_valid<=0;
                            write_tag<=ext_addr[22:6];write_dirty<=1;idle_age<=0;
                            if (ext_addr[5:3]==7) flush_pending<=1;
                            for (integer k=0;k<8;k=k+1) if (ext_wstrb[k]) begin
                                write_line[{ext_addr[5:3],1'b0}+4'(k/4)][(k%4)*8+:8]
                                    <=ext_wdata[k*8+:8];
                                write_mask[{ext_addr[5:3],3'b0}+6'(k)]<=1;
                            end
                        end else if (read_valid && read_tag==ext_addr[22:6]) begin
                            ext_rdata<={read_line[{ext_addr[5:3],1'b1}],
                                        read_line[{ext_addr[5:3],1'b0}]};
                            ext_rvalid<=1;
                        end else begin
                            word_addr<={ext_addr[22:6],4'b0};read_tag<=ext_addr[22:6];
                            requested_beat<=ext_addr[5:3];read_valid<=0;write_operation<=0;
                            cmd<=3'b011;cmd_en<=1;state<=ACT_WAIT;
                        end
                    end
                end
                ACT_WAIT: if (cmd_ack) begin
                    cmd<=write_operation ? 3'b100 : 3'b101;cmd_en<=1;
                    cycle_count<=0;ack_low<=0;ack_seen<=0;state<=RW_WAIT;
                end
                RW_WAIT: begin
                    if (cycle_count!=63) cycle_count<=cycle_count+1;
                    if (!cmd_ack) ack_low<=1;
                    if (cmd_ack && ack_low) ack_seen<=1;
                    if (write_operation) begin
                        if (cmd_ack && ack_low && cycle_count>=16) begin
                            settle_count<=0;state<=SETTLE;
                        end
                    end else begin
                        if (cycle_count>=4 && cycle_count<20)
                            read_line[cycle_count-4]<=read_word;
                        if (cycle_count==19) state<=READ_RETURN;
                    end
                end
                READ_RETURN: begin
                    ext_rdata<={read_line[{requested_beat,1'b1}],
                                read_line[{requested_beat,1'b0}]};
                    ext_rvalid<=1;read_valid<=1;state<=READ_FINISH;
                    if (cmd_ack && ack_low) ack_seen<=1;
                end
                READ_FINISH: if (ack_seen || (cmd_ack && ack_low)) begin
                    settle_count<=0;state<=SETTLE;
                end
                SETTLE: if (settle_count==15) begin
                    state<=IDLE;
                    if (write_operation) begin
                        write_dirty<=0;write_mask<=0;flush_pending<=0;idle_age<=0;
                    end
                end else settle_count<=settle_count+1;
                REF_WAIT: if (cmd_ack) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule
