// Single-outstanding tile transfer between the existing 64-bit SRAM port and
// an abstract 64-bit off-chip memory port. The latter needs an SDRAM adapter;
// this module does not claim refresh, burst scheduling or physical bring-up.
module v2_tile_dma #(
    parameter integer SRAM_BYTES = 32768,
    parameter integer EXT_BYTES = 8388608
)(
    input logic clk, rst_n, start, abort_run,
    input logic to_sram,
    input logic [23:0] sram_base, ext_base,
    input logic [31:0] length_bytes,
    output logic busy, done,
    output logic [7:0] error_code,
    output logic [31:0] bytes_copied, physical_read_bytes,
    output logic sram_req, sram_wr,
    output logic [23:0] sram_addr,
    output logic [63:0] sram_wdata,
    output logic [7:0] sram_wstrb,
    input logic sram_ready, sram_rvalid,
    input logic [63:0] sram_rdata,
    output logic ext_req, ext_wr,
    output logic [23:0] ext_addr,
    output logic [63:0] ext_wdata,
    output logic [7:0] ext_wstrb,
    input logic ext_ready, ext_rvalid,
    input logic [63:0] ext_rdata
);
    typedef enum logic [2:0] {IDLE,SOURCE_REQ,SOURCE_WAIT,DEST_REQ,DRAIN} state_t;
    state_t state;
    logic direction;
    logic [23:0] sram_origin,ext_origin;
    logic [31:0] transfer_length,offset;
    logic [63:0] payload;
    wire [31:0] remaining=transfer_length-offset;
    wire [7:0] byte_mask=(remaining>=8)?8'hff:(8'hff>>(8-remaining[2:0]));
    wire [31:0] copied_now=(remaining>=8)?32'd8:remaining;
    wire source_ready=direction?ext_ready:sram_ready;
    wire source_rvalid=direction?ext_rvalid:sram_rvalid;
    wire [63:0] source_rdata=direction?ext_rdata:sram_rdata;
    wire dest_ready=direction?sram_ready:ext_ready;

    always_comb begin
        sram_req=0;sram_wr=0;sram_addr=sram_origin+offset[23:0];
        sram_wdata=payload;sram_wstrb=0;
        ext_req=0;ext_wr=0;ext_addr=ext_origin+offset[23:0];
        ext_wdata=payload;ext_wstrb=0;
        if(!abort_run)begin
            if(state==SOURCE_REQ)begin
                if(direction)ext_req=1;
                else sram_req=1;
            end else if(state==DEST_REQ)begin
                if(direction)begin sram_req=1;sram_wr=1;sram_wstrb=byte_mask;end
                else begin ext_req=1;ext_wr=1;ext_wstrb=byte_mask;end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            state<=IDLE;busy<=0;done<=0;error_code<=0;
            direction<=0;sram_origin<=0;ext_origin<=0;
            transfer_length<=0;offset<=0;payload<=0;
            bytes_copied<=0;physical_read_bytes<=0;
        end else begin
            done<=0;
            if(abort_run&&state!=IDLE&&state!=DRAIN)begin
                if(state==SOURCE_WAIT&&!source_rvalid)state<=DRAIN;
                else begin state<=IDLE;busy<=0;done<=1;error_code<=8'd2;end
            end else case(state)
                IDLE:if(start)begin
                    bytes_copied<=0;physical_read_bytes<=0;error_code<=0;
                    if(length_bytes==0||length_bytes>SRAM_BYTES||
                       sram_base[2:0]!=0||ext_base[2:0]!=0||
                       {40'b0,sram_base}+{32'b0,length_bytes}>64'(SRAM_BYTES)||
                       {40'b0,ext_base}+{32'b0,length_bytes}>64'(EXT_BYTES))begin
                        done<=1;error_code<=8'd1;
                    end else begin
                        direction<=to_sram;sram_origin<=sram_base;ext_origin<=ext_base;
                        transfer_length<=length_bytes;offset<=0;busy<=1;state<=SOURCE_REQ;
                    end
                end
                SOURCE_REQ:if(source_ready)state<=SOURCE_WAIT;
                SOURCE_WAIT:if(source_rvalid)begin
                    payload<=source_rdata;physical_read_bytes<=physical_read_bytes+32'd8;
                    state<=DEST_REQ;
                end
                DEST_REQ:if(dest_ready)begin
                    bytes_copied<=bytes_copied+copied_now;
                    if(remaining<=8)begin state<=IDLE;busy<=0;done<=1;end
                    else begin offset<=offset+32'd8;state<=SOURCE_REQ;end
                end
                DRAIN:if(source_rvalid)begin
                    state<=IDLE;busy<=0;done<=1;error_code<=8'd2;
                end
                default:begin state<=IDLE;busy<=0;done<=1;error_code<=8'd3;end
            endcase
        end
    end
endmodule
