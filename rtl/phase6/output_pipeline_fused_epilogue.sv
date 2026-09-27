// Feasibility prototype: exact mapping of an already requantized INT8 value.
// This retains the producer's INT8 rounding/saturation boundary. The 256-byte
// table is written while idle and has one synchronous lookup port.
// This module is not yet wired into the full engine or deployed on the board.
module phase6_fused_epilogue (
    input logic clk, rst_n, flush,
    input logic table_we,
    input logic [7:0] table_index, table_value,
    input logic in_valid,
    output logic in_ready,
    input logic [7:0] in_value,
    output logic out_valid,
    input logic out_ready,
    output logic [7:0] result
);
    (* syn_ramstyle = "block_ram" *) logic [7:0] mapping[0:255];
    assign in_ready = !out_valid || out_ready;
    always_ff @(posedge clk) begin
        if(table_we)mapping[table_index] <= table_value;
        if(in_ready && in_valid && !flush)
            result <= mapping[in_value];
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n)out_valid <= 0;
        else if(flush)out_valid <= 0;
        else if(in_ready)out_valid <= in_valid;
    end
// synthesis translate_off
    always @(posedge clk)if(rst_n && table_we && (in_valid || out_valid))
        $fatal(1,"activation table must remain unchanged while stream is live");
// synthesis translate_on
endmodule
