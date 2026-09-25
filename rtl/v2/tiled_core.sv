// Experimental shared-SRAM hierarchy for boardless Phase 4 validation.
// The external port remains abstract: no SDRAM initialization or refresh here.
module v2_tiled_core (
    input logic clk, rst_n,
    input logic engine_start, engine_abort,
    input logic [23:0] engine_start_pc,
    output logic engine_busy, engine_done,
    output logic [7:0] engine_error,
    output logic [31:0] engine_elapsed,
    output logic [31:0] compute_cycles, wait_cycles, control_cycles,
    output logic [31:0] useful_macs, read_bytes, write_bytes, layer_count,
    input logic dma_start, dma_abort, dma_to_sram,
    input logic [23:0] dma_sram_base, dma_ext_base,
    input logic [31:0] dma_length,
    output logic dma_busy, dma_done,
    output logic [7:0] dma_error,
    output logic [31:0] dma_bytes_copied,
    input logic host_req, host_wr,
    input logic [23:0] host_addr,
    input logic [63:0] host_wdata,
    input logic [7:0] host_wstrb,
    output logic host_ready, host_rvalid,
    output logic [63:0] host_rdata,
    output logic ext_req, ext_wr,
    output logic [23:0] ext_addr,
    output logic [63:0] ext_wdata,
    output logic [7:0] ext_wstrb,
    input logic ext_ready, ext_rvalid,
    input logic [63:0] ext_rdata
);
    logic ereq, ewr, dreq, dwr, sready, srvalid;
    logic [23:0] eaddr, daddr;
    logic [63:0] ewdata, dwdata, srdata;
    logic [7:0] estrb, dstrb;
    logic [31:0] dma_physical_read_bytes;
    logic layer_marker;

    v2_engine engine (
        .clk(clk), .rst_n(rst_n), .start(engine_start), .abort_run(engine_abort),
        .clear_counters(1'b0), .start_pc(engine_start_pc),
        .busy(engine_busy), .done(engine_done), .error_code(engine_error),
        .mem_req(ereq), .mem_wr(ewr), .mem_addr(eaddr),
        .mem_wdata(ewdata), .mem_wstrb(estrb),
        .mem_ready(eready), .mem_rvalid(ervalid), .mem_rdata(srdata),
        .elapsed(engine_elapsed), .compute_cycles(compute_cycles),
        .wait_cycles(wait_cycles), .control_cycles(control_cycles),
        .useful_macs(useful_macs), .read_bytes(read_bytes),
        .write_bytes(write_bytes), .layer_count(layer_count),
        .layer_marker(layer_marker)
    );
    v2_tile_dma dma (
        .clk(clk), .rst_n(rst_n), .start(dma_start), .abort_run(dma_abort),
        .to_sram(dma_to_sram), .sram_base(dma_sram_base),
        .ext_base(dma_ext_base), .length_bytes(dma_length),
        .busy(dma_busy), .done(dma_done), .error_code(dma_error),
        .bytes_copied(dma_bytes_copied),
        .physical_read_bytes(dma_physical_read_bytes),
        .sram_req(dreq), .sram_wr(dwr), .sram_addr(daddr),
        .sram_wdata(dwdata), .sram_wstrb(dstrb),
        .sram_ready(dready), .sram_rvalid(drvalid), .sram_rdata(srdata),
        .ext_req(ext_req), .ext_wr(ext_wr), .ext_addr(ext_addr),
        .ext_wdata(ext_wdata), .ext_wstrb(ext_wstrb),
        .ext_ready(ext_ready), .ext_rvalid(ext_rvalid), .ext_rdata(ext_rdata)
    );

    logic eready, ervalid, dready, drvalid;
    logic grant_engine, grant_dma, grant_host, last_grant_dma;
    logic [1:0] read_owner;
    logic sreq, swr;
    logic [23:0] saddr;
    logic [63:0] swdata;
    logic [7:0] swstrb;
    always_comb begin
        grant_engine = 0;
        grant_dma = 0;
        grant_host = 0;
        if (ereq && dreq) begin
            grant_engine = last_grant_dma;
            grant_dma = !last_grant_dma;
        end else if (ereq) grant_engine = 1;
        else if (dreq) grant_dma = 1;
        else if (host_req && !engine_busy && !dma_busy &&
                 !engine_start && !dma_start) grant_host = 1;
        sreq = grant_engine || grant_dma || grant_host;
        swr = grant_engine ? ewr : grant_dma ? dwr : host_wr;
        saddr = grant_engine ? eaddr : grant_dma ? daddr : host_addr;
        swdata = grant_engine ? ewdata : grant_dma ? dwdata : host_wdata;
        swstrb = grant_engine ? estrb : grant_dma ? dstrb : host_wstrb;
        eready = grant_engine && sready;
        dready = grant_dma && sready;
        host_ready = grant_host && sready;
        ervalid = srvalid && read_owner == 2'd1;
        drvalid = srvalid && read_owner == 2'd2;
        host_rvalid = srvalid && read_owner == 2'd3;
        host_rdata = srdata;
    end
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            last_grant_dma <= 0;
            read_owner <= 0;
        end else begin
            if (sreq && sready) begin
                if (grant_engine) last_grant_dma <= 0;
                else if (grant_dma) last_grant_dma <= 1;
                if (!swr)
                    read_owner <= grant_engine ? 2'd1 : grant_dma ? 2'd2 : 2'd3;
            end
        end
    end
    v2_scratchpad memory (
        .clk(clk), .rst_n(rst_n), .req(sreq), .wr(swr), .addr(saddr),
        .wdata(swdata), .wstrb(swstrb), .ready(sready),
        .rvalid(srvalid), .rdata(srdata)
    );
endmodule
