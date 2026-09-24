// Refresh request scheduler for the GW2AR embedded SDR SDRAM.
// Gowin SDRAM Controller HS requires user-issued AUTO REFRESH commands.
// The caller must prioritize refresh_req over new read/write work and map it
// to I_sdrc_cmd_en=1, I_sdrc_cmd=3'b001, then return O_sdrc_cmd_ack.
module v2_sdram_refresh #(
    parameter integer CLOCK_HZ = 27000000,
    parameter integer REFRESH_PERIOD_CYCLES = CLOCK_HZ / 64000
)(
    input logic clk, rst_n,
    input logic init_done,
    input logic controller_idle,
    input logic refresh_ack,
    output logic refresh_req,
    output logic [12:0] pending_refreshes,
    output logic refresh_deadline_missed
);
    localparam integer COUNT_BITS = $clog2(REFRESH_PERIOD_CYCLES + 1);
    logic [COUNT_BITS-1:0] period_counter;
    logic inflight;
    wire tick = period_counter == COUNT_BITS'(REFRESH_PERIOD_CYCLES - 1);
    assign refresh_req = init_done && controller_idle && !inflight &&
                         pending_refreshes != 0;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            period_counter <= 0;
            pending_refreshes <= 0;
            inflight <= 0;
            refresh_deadline_missed <= 0;
        end else if (!init_done) begin
            period_counter <= 0;
            pending_refreshes <= 0;
            inflight <= 0;
            refresh_deadline_missed <= 0;
        end else begin
            period_counter <= tick ? '0 : period_counter + 1'b1;
            if (refresh_ack && inflight) inflight <= 0;
            else if (refresh_req) inflight <= 1;
            case ({tick, refresh_ack && inflight})
                2'b10: if (pending_refreshes != 13'h1fff)
                    pending_refreshes <= pending_refreshes + 1'b1;
                2'b01: pending_refreshes <= pending_refreshes - 1'b1;
                default: ;
            endcase
            if (pending_refreshes >= 13'd4096)
                refresh_deadline_missed <= 1;
        end
    end
// synthesis translate_off
`ifndef SYNTHESIS
    initial if (REFRESH_PERIOD_CYCLES < 2)
        $fatal(1, "refresh period must be at least two clocks");
`endif
// synthesis translate_on
endmodule
