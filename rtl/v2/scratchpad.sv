// Eight byte banks, a single synchronous 64-bit port, byte write enables.
// No reset/clear loops: inferred BSRAM contents persist across RUN/ABORT.
module v2_scratchpad #(
    parameter integer MEM_BYTES=target_pkg::MEM_BYTES
)(
    input logic clk, rst_n,
    input logic req, wr,
    input logic [23:0] addr,
    input logic [63:0] wdata,
    input logic [7:0] wstrb,
    output logic ready, rvalid,
    output logic [63:0] rdata
);
    localparam integer WORD_BITS=$clog2(MEM_BYTES/8);
    assign ready=1'b1;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) rvalid<=0;
        else rvalid<=req && !wr;
    end
    for (genvar lane=0;lane<8;lane=lane+1) begin: banks
        (* syn_ramstyle = "block_ram" *) logic [7:0] ram[0:MEM_BYTES/8-1];
        always_ff @(posedge clk) begin
            if (req) begin
                if (wr && wstrb[lane]) ram[addr[WORD_BITS+2:3]]<=wdata[lane*8+:8];
                rdata[lane*8+:8]<=ram[addr[WORD_BITS+2:3]];
            end
        end
    end
// synthesis translate_off
`ifndef SYNTHESIS
    initial begin
        if (MEM_BYTES<1024 || ((MEM_BYTES&(MEM_BYTES-1))!=0) || MEM_BYTES>(1<<24)) $fatal(1,"invalid SRAM geometry");
    end
    always @(posedge clk) if (rst_n && req && ({8'b0,addr}>=MEM_BYTES || addr[2:0]!=0)) $fatal(1,"invalid SRAM request");
`endif
// synthesis translate_on
endmodule
