// The same system hierarchy is used by board synthesis and cycle simulation.
module v2_system #(
    parameter integer TIMEOUT_CYCLES=target_pkg::RX_TIMEOUT
)(
    input logic clk,rst_n,
    input logic rx_valid,input logic [7:0] rx_data,
    output logic tx_valid,output logic [7:0] tx_data,input logic tx_ready,
    output logic busy,layer_marker,
    output logic [7:0] error_code
);
    logic start,abort_run,clear_counters,done;
    logic [23:0] pc;
    logic [31:0] elapsed,compute_cycles,wait_cycles,control_cycles,useful_macs,read_bytes,write_bytes,layer_count,protocol_errors;
    logic hreq,hwr,ereq,ewr,mready,mvalid;
    logic [23:0] haddr,eaddr;
    logic [63:0] hwdata,ewdata,mdata;
    logic [7:0] hstrb,estrb;
    v2_engine engine(.clk(clk),.rst_n(rst_n),.start(start),.abort_run(abort_run),.clear_counters(clear_counters),.start_pc(pc),.busy(busy),.done(done),.error_code(error_code),
        .mem_req(ereq),.mem_wr(ewr),.mem_addr(eaddr),.mem_wdata(ewdata),.mem_wstrb(estrb),.mem_ready(mready&&busy),.mem_rvalid(mvalid&&busy),.mem_rdata(mdata),
        .elapsed(elapsed),.compute_cycles(compute_cycles),.wait_cycles(wait_cycles),.control_cycles(control_cycles),.useful_macs(useful_macs),.read_bytes(read_bytes),.write_bytes(write_bytes),.layer_count(layer_count),.layer_marker(layer_marker));
    v2_command #(.TIMEOUT_CYCLES(TIMEOUT_CYCLES)) command(.clk(clk),.rst_n(rst_n),.rx_valid(rx_valid),.rx_data(rx_data),.tx_valid(tx_valid),.tx_data(tx_data),.tx_ready(tx_ready),.mmio_while_busy(1'b0),
        .start(start),.abort_run(abort_run),.clear_counters(clear_counters),.start_pc(pc),.busy(busy),.core_error(error_code),
        .elapsed(elapsed),.compute_cycles(compute_cycles),.wait_cycles(wait_cycles),.control_cycles(control_cycles),.useful_macs(useful_macs),.read_bytes(read_bytes),.write_bytes(write_bytes),.layer_count(layer_count),
        .mem_req(hreq),.mem_wr(hwr),.mem_addr(haddr),.mem_wdata(hwdata),.mem_wstrb(hstrb),.mem_ready(mready&&!busy),.mem_rvalid(mvalid&&!busy),.mem_rdata(mdata),.protocol_errors(protocol_errors));
    v2_scratchpad memory(.clk(clk),.rst_n(rst_n),.req(busy?ereq:hreq),.wr(busy?ewr:hwr),.addr(busy?eaddr:haddr),.wdata(busy?ewdata:hwdata),.wstrb(busy?estrb:hstrb),.ready(mready),.rvalid(mvalid),.rdata(mdata));
endmodule
