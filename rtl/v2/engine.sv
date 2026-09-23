// Shared phase-2 engine: eight signed INT8 MAC lanes, static per-channel v2
// requantization, synchronous memory requests, explicit errors and abort.
module v2_engine (
    input logic clk, rst_n, start, abort_run, clear_counters,
    input logic [23:0] start_pc,
    output logic busy, done,
    output logic [7:0] error_code,
    output logic mem_req, mem_wr,
    output logic [23:0] mem_addr,
    output logic [63:0] mem_wdata,
    output logic [7:0] mem_wstrb,
    input logic mem_ready, mem_rvalid,
    input logic [63:0] mem_rdata,
    output logic [31:0] elapsed, compute_cycles, wait_cycles, control_cycles,
    output logic [31:0] useful_macs, read_bytes, write_bytes, layer_count,
    output logic layer_marker
);
    import target_pkg::*;
    typedef enum logic [4:0] {IDLE,D_REQ,D_WAIT,D_CHECK,P_REQ,P_WAIT,P_CHECK,X_REQ,X_WAIT,W_REQ,W_WAIT,MAC,Q_SEND,Q_WAIT,WRITE_OUT,NEXT_OUT,NEXT_LAYER} state_t;
    state_t state;
    logic [63:0] d[0:7];
    logic [2:0] di;
    logic pi;
    logic [23:0] pc;
    logic [31:0] row,col;
    logic [23:0] weight_row;
    logic [63:0] xword,whex,p0,p1;
    wire [7:0] op=d[0][47:40];
    wire [31:0] xb=d[1][31:0], yb=d[1][63:32], wb=d[2][31:0], pb=d[2][63:32];
    wire [31:0] count=d[3][31:0], outputs=d[3][63:32], stride=d[4][31:0], next_pc=d[4][63:32];
    logic signed [31:0] acc, mult;
    logic [5:0] shift;
    logic signed [7:0] zy,zx;
    logic signed [7:0] result_byte;
    logic signed [15:0] products[0:7];
    logic signed [35:0] sum,acc_next;
    logic [3:0] valid_lanes;
    logic qvalid,qready,qout;
    logic signed [7:0] qresult;
    logic signed [7:0] scalar_x;
    logic signed [31:0] scalar_acc;
    logic [63:0] address_full;
    integer j;
    always_comb begin
        sum=0;valid_lanes=0;
        for (integer k=0;k<8;k=k+1) begin
            products[k]=$signed(xword[k*8+:8])*$signed(whex[k*8+:8]);
            if (col+k<count) begin sum=sum+{{20{products[k][15]}},products[k]}; valid_lanes=valid_lanes+4'd1; end
        end
        acc_next={{4{acc[31]}},acc}+sum;
        scalar_x=$signed(xword[row[2:0]*8+:8]);
        scalar_acc=(scalar_x<zx) ? 0 : ({{24{scalar_x[7]}},scalar_x}-{{24{zx[7]}},zx});
        mem_req=0;mem_wr=0;address_full=0;mem_wdata=0;mem_wstrb=0;
        case(state)
            D_REQ:begin mem_req=1;address_full={40'b0,pc}+{58'b0,di,3'b0};end
            P_REQ:begin mem_req=1;address_full={32'b0,pb}+((op==OP_GEMM)?({32'b0,row}<<4):0)+(pi?8:0);end
            X_REQ:begin mem_req=1;address_full={32'b0,xb}+((op==OP_GEMM)?{32'b0,col}:({32'b0,row}&64'hfffffffffffffff8));end
            W_REQ:begin mem_req=1;address_full={40'b0,weight_row}+{32'b0,col};end
            WRITE_OUT:begin
                mem_req=1;mem_wr=1;address_full=({32'b0,yb}+{32'b0,row})&64'hfffffffffffffff8;
                mem_wstrb=8'b1<<((yb+row)&7);mem_wdata={8{result_byte}};
            end
            default:begin end
        endcase
        mem_addr=address_full[23:0];
        qvalid=(state==Q_SEND);
    end
    v2_requantizer requant(.clk(clk),.rst_n(rst_n),.flush(abort_run||clear_counters),.in_valid(qvalid),.in_ready(qready),.accumulator(acc),.multiplier(mult),.shift(shift),.zero_point(zy),.out_valid(qout),.out_ready(state==Q_WAIT),.result(qresult));
    task automatic fail(input logic [7:0] code);
        begin error_code<=code;busy<=0;done<=1;state<=IDLE;end
    endtask
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=IDLE;busy<=0;done<=0;error_code<=0;pc<=0;di<=0;pi<=0;row<=0;col<=0;weight_row<=0;
            acc<=0;mult<=0;shift<=0;zy<=0;zx<=0;xword<=0;whex<=0;p0<=0;p1<=0;result_byte<=0;
            elapsed<=0;compute_cycles<=0;wait_cycles<=0;control_cycles<=0;useful_macs<=0;read_bytes<=0;write_bytes<=0;layer_count<=0;layer_marker<=0;
            for(j=0;j<8;j=j+1)d[j]<=0;
        end else begin
            done<=0;layer_marker<=0;
            if(busy)begin
                elapsed<=elapsed+1;
                if(state==MAC)compute_cycles<=compute_cycles+1;
                else if(state==D_REQ||state==D_WAIT||state==P_REQ||state==P_WAIT||state==X_REQ||state==X_WAIT||state==W_REQ||state==W_WAIT||state==WRITE_OUT)wait_cycles<=wait_cycles+1;
                else control_cycles<=control_cycles+1;
                if(mem_req&&mem_ready)begin
                    if(mem_wr)write_bytes<=write_bytes+1;else read_bytes<=read_bytes+8;
                end
            end
            if(clear_counters)begin
                state<=IDLE;busy<=0;done<=0;error_code<=0;elapsed<=0;compute_cycles<=0;wait_cycles<=0;control_cycles<=0;useful_macs<=0;read_bytes<=0;write_bytes<=0;layer_count<=0;
            end else if(abort_run)begin state<=IDLE;busy<=0;done<=busy;error_code<=busy?8'd6:8'd0;end
            else if(busy&&elapsed>=WATCHDOG-1)fail(8'd7);
            else case(state)
                IDLE:if(start)begin
                    elapsed<=0;compute_cycles<=0;wait_cycles<=0;control_cycles<=0;useful_macs<=0;read_bytes<=0;write_bytes<=0;layer_count<=0;error_code<=0;
                    if(start_pc[5:0]!=0||{8'b0,start_pc}>MEM_BYTES-64)fail(8'd1);
                    else begin pc<=start_pc;di<=0;busy<=1;state<=D_REQ;end
                end
                D_REQ:if(mem_ready)state<=D_WAIT;
                D_WAIT:if(mem_rvalid)begin d[di]<=mem_rdata;if(di==7)state<=D_CHECK;else begin di<=di+3'd1;state<=D_REQ;end end
                D_CHECK:begin
                    if(d[0][31:0]!=32'h32445355||d[0][39:32]!=8'(DESC_VERSION)||d[0][63:48]!=0)fail(8'd2);
                    else if(op==OP_HALT)begin busy<=0;done<=1;state<=IDLE;end
                    else if(op!=OP_GEMM&&op!=OP_RELU&&op!=OP_COPY)fail(8'd3);
                    else if(count==0||outputs==0||count>MEM_BYTES||outputs>MEM_BYTES||xb[2:0]!=0||{32'b0,xb}+(({32'b0,count}+7)&64'hfffffffffffffff8)>64'(MEM_BYTES)||{32'b0,yb}+{32'b0,outputs}>64'(MEM_BYTES)||next_pc[5:0]!=0||next_pc>MEM_BYTES-64||d[5]!=64'h0001000100010001||d[6]!=0||d[7]!=64'h0001000100010001)fail(8'd1);
                    else if(op==OP_GEMM&&(wb[2:0]!=0||pb[2:0]!=0||stride[2:0]!=0||stride<count||stride>MEM_BYTES||{32'b0,wb}+{32'b0,outputs}*{32'b0,stride}>64'(MEM_BYTES)||{32'b0,pb}+({32'b0,outputs}<<4)>64'(MEM_BYTES)))fail(8'd1);
                    else if(op!=OP_GEMM&&(count!=outputs||(op==OP_RELU&&(pb[2:0]!=0||{32'b0,pb}+16>64'(MEM_BYTES)))))fail(8'd1);
                    else if(layer_count>=MAX_DESCRIPTORS)fail(8'd7);
                    else begin row<=0;col<=0;weight_row<=wb[23:0];pi<=0;state<=(op==OP_COPY)?X_REQ:P_REQ;end
                end
                P_REQ:if(mem_ready)state<=P_WAIT;
                P_WAIT:if(mem_rvalid)begin
                    if(!pi)begin p0<=mem_rdata;pi<=1;state<=P_REQ;end
                    else begin p1<=mem_rdata;state<=P_CHECK;end
                end
                P_CHECK:begin
                    if(p0[63]||p0[63:32]==0||p1[7:0]>62||p1[63:40]!=0||p1[39:32]!=8'd127||(op==OP_GEMM&&p1[31:24]!=8'h80)||(op==OP_RELU&&(p1[31:24]!=p1[23:16]||p0[31:0]!=0)))fail(8'd4);
                    else begin acc<=$signed(p0[31:0]);mult<=$signed(p0[63:32]);shift<=p1[5:0];zy<=$signed(p1[15:8]);zx<=$signed(p1[23:16]);col<=0;state<=X_REQ;end
                end
                X_REQ:if(mem_ready)state<=X_WAIT;
                X_WAIT:if(mem_rvalid)begin xword<=mem_rdata;if(op==OP_GEMM)state<=W_REQ;else state<=MAC;end
                W_REQ:if(mem_ready)state<=W_WAIT;
                W_WAIT:if(mem_rvalid)begin whex<=mem_rdata;state<=MAC;end
                MAC:begin
                    if(op==OP_GEMM)begin
                        if(acc_next>36'sd2147483647||acc_next< -36'sd2147483648)fail(8'd5);
                        else begin acc<=acc_next[31:0];useful_macs<=useful_macs+{28'b0,valid_lanes};if(col+8>=count)state<=Q_SEND;else begin col<=col+8;state<=X_REQ;end end
                    end else if(op==OP_RELU)begin acc<=scalar_acc;state<=Q_SEND;end
                    else begin result_byte<=scalar_x;state<=WRITE_OUT;end
                end
                Q_SEND:if(qready)state<=Q_WAIT;
                Q_WAIT:if(qout)begin result_byte<=qresult;state<=WRITE_OUT;end
                WRITE_OUT:if(mem_ready)state<=NEXT_OUT;
                NEXT_OUT:begin
                    if(row+1>=outputs)state<=NEXT_LAYER;
                    else begin row<=row+1;weight_row<=weight_row+stride[23:0];col<=0;pi<=0;state<=(op==OP_GEMM)?P_REQ:X_REQ;end
                end
                NEXT_LAYER:begin pc<=next_pc[23:0];di<=0;layer_count<=layer_count+1;layer_marker<=1;state<=D_REQ;end
                default:fail(8'd8);
            endcase
        end
    end
endmodule
