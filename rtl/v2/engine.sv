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
    typedef enum logic [4:0] {IDLE,D_REQ,D_WAIT,D_CHECK,P_REQ,P_WAIT,P_CHECK,X_REQ,X_WAIT,W_REQ,W_WAIT,MAC,Q_SEND,Q_WAIT,WRITE_OUT,NEXT_OUT,NEXT_LAYER,WIN_PREP,WIN_REQ,WIN_WAIT,POOL_PREP,POOL_REQ,POOL_WAIT,POOL_DONE,GEOM0,GEOM1,GEOM_COUNT,GEOM_CHECK,OTHER_CHECK} state_t;
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
    wire [15:0] kh=d[5][15:0],kw=d[5][31:16],sh=d[5][47:32],sw=d[5][63:48];
    wire [15:0] pt=d[6][15:0],pbm=d[6][31:16],pl=d[6][47:32],pr=d[6][63:48];
    wire [15:0] ih=d[7][15:0],iw=d[7][31:16],ic=d[7][47:32],oc_total=d[7][63:48];
    wire spatial=(op==OP_CONV||op==OP_MAXPOOL);
    wire [31:0] plane_calc=32'(ih)*32'(iw);
    wire [31:0] numerator_h=32'(ih)+32'(pt)+32'(pbm)-32'(kh);
    wire [31:0] numerator_w=32'(iw)+32'(pl)+32'(pr)-32'(kw);
    wire [31:0] oh_calc=(sh==2)?((numerator_h>>1)+1):(numerator_h+1);
    wire [31:0] ow_calc=(sw==2)?((numerator_w>>1)+1):(numerator_w+1);
    logic [15:0] oh,ow,ox,oy,oc,win_kx,win_ky,win_ic;
    logic [2:0] win_lane;
    logic [31:0] plane,row_step,pad_offset,kernel_area,output_area;
    logic [31:0] input_bytes_reg,reduction_reg,outputs_reg;
    logic signed [31:0] channel_origin,row_origin,pixel_origin,window_channel_origin,window_line_origin,win_addr;
    logic signed [17:0] origin_x,origin_y;
    wire signed [18:0] window_x=$signed(origin_x)+$signed({3'b0,win_kx});
    wire signed [18:0] window_y=$signed(origin_y)+$signed({3'b0,win_ky});
    wire window_inside=(window_x>=0 && window_y>=0 && window_x<$signed({3'b0,iw}) && window_y<$signed({3'b0,ih}));
    logic signed [7:0] pool_max;
    logic pool_any;
    // One filter tile (up to 64 weights) is reused over its spatial outputs.
    // Larger reductions use the same MAC array and stream weights from SRAM.
    logic [63:0] weight_cache[0:7];
    logic weight_cache_valid;
    wire cached_weight=(op==OP_CONV&&count<=64&&weight_cache_valid);
    // The physical SRAM returns eight bytes even for one window pixel. Keep
    // eight tagged words across neighboring windows to avoid duplicate reads.
    logic [63:0] activation_cache_data[0:7];
    logic [17:0] activation_cache_tag[0:7];
    logic [7:0] activation_cache_valid;
    wire [2:0] activation_cache_index=win_addr[5:3];
    wire activation_cache_hit=activation_cache_valid[activation_cache_index]&&
        activation_cache_tag[activation_cache_index]==win_addr[23:6];
    wire signed [7:0] activation_cache_byte=$signed(activation_cache_data[activation_cache_index][win_addr[2:0]*8+:8]);
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
            P_REQ:begin mem_req=1;address_full={32'b0,pb}+((op==OP_GEMM)?({32'b0,row}<<4):((op==OP_CONV)?({48'b0,oc}<<4):0))+(pi?8:0);end
            X_REQ:begin mem_req=1;address_full={32'b0,xb}+((op==OP_GEMM)?{32'b0,col}:({32'b0,row}&64'hfffffffffffffff8));end
            WIN_REQ,POOL_REQ:begin mem_req=1;address_full={40'b0,win_addr[23:3],3'b0};end
            W_REQ:begin mem_req=!cached_weight;address_full={40'b0,weight_row}+{32'b0,col};end
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
    task automatic advance_window;
        begin
            if(win_kx+16'd1<kw)begin win_kx<=win_kx+16'd1;win_addr<=win_addr+1;end
            else if(win_ky+1<kh)begin
                win_kx<=0;win_ky<=win_ky+16'd1;window_line_origin<=window_line_origin+$signed({16'b0,iw});
                win_addr<=window_line_origin+$signed({16'b0,iw});
            end else begin
                win_kx<=0;win_ky<=0;win_ic<=win_ic+16'd1;
                window_channel_origin<=window_channel_origin+$signed(plane);
                window_line_origin<=window_channel_origin+$signed(plane);
                win_addr<=window_channel_origin+$signed(plane);
            end
        end
    endtask
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=IDLE;busy<=0;done<=0;error_code<=0;pc<=0;di<=0;pi<=0;row<=0;col<=0;weight_row<=0;
            oh<=0;ow<=0;ox<=0;oy<=0;oc<=0;win_kx<=0;win_ky<=0;win_ic<=0;win_lane<=0;weight_cache_valid<=0;activation_cache_valid<=0;
            plane<=0;row_step<=0;pad_offset<=0;kernel_area<=0;output_area<=0;
            input_bytes_reg<=0;reduction_reg<=0;outputs_reg<=0;
            channel_origin<=0;row_origin<=0;pixel_origin<=0;
            window_channel_origin<=0;window_line_origin<=0;win_addr<=0;origin_x<=0;origin_y<=0;pool_max<=0;pool_any<=0;
            acc<=0;mult<=0;shift<=0;zy<=0;zx<=0;xword<=0;whex<=0;p0<=0;p1<=0;result_byte<=0;
            elapsed<=0;compute_cycles<=0;wait_cycles<=0;control_cycles<=0;useful_macs<=0;read_bytes<=0;write_bytes<=0;layer_count<=0;layer_marker<=0;
            for(j=0;j<8;j=j+1)d[j]<=0;
        end else begin
            done<=0;layer_marker<=0;
            if(busy)begin
                elapsed<=elapsed+1;
                if(state==MAC)compute_cycles<=compute_cycles+1;
                else if(state==D_REQ||state==D_WAIT||state==P_REQ||state==P_WAIT||state==X_REQ||state==X_WAIT||(state==W_REQ&&mem_req)||state==W_WAIT||state==WRITE_OUT||state==WIN_REQ||state==WIN_WAIT||state==POOL_REQ||state==POOL_WAIT)wait_cycles<=wait_cycles+1;
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
                    else if(op!=OP_GEMM&&op!=OP_RELU&&op!=OP_COPY&&op!=OP_CONV&&op!=OP_MAXPOOL)fail(8'd3);
                    else if(count==0||outputs==0||count>MEM_BYTES||outputs>MEM_BYTES||xb[2:0]!=0||{32'b0,yb}+{32'b0,outputs}>64'(MEM_BYTES)||next_pc[5:0]!=0||next_pc>MEM_BYTES-64)fail(8'd1);
                    else state<=spatial?GEOM0:OTHER_CHECK;
                end
                OTHER_CHECK:begin
                    if(d[5]!=64'h0001000100010001||d[6]!=0||d[7]!=64'h0001000100010001||
                        {32'b0,xb}+(({32'b0,count}+7)&64'hfffffffffffffff8)>64'(MEM_BYTES))fail(8'd1);
                    else if(op==OP_GEMM&&(wb[2:0]!=0||pb[2:0]!=0||stride[2:0]!=0||stride<count||stride>MEM_BYTES||
                        {32'b0,wb}+{32'b0,outputs}*{32'b0,stride}>64'(MEM_BYTES)||
                        {32'b0,pb}+({32'b0,outputs}<<4)>64'(MEM_BYTES)))fail(8'd1);
                    else if(op!=OP_GEMM&&count!=outputs)fail(8'd1);
                    else if(op==OP_RELU&&(pb[2:0]!=0||{32'b0,pb}+16>64'(MEM_BYTES)))fail(8'd1);
                    else if(layer_count>=MAX_DESCRIPTORS)fail(8'd7);
                    else begin
                        row<=0;col<=0;weight_row<=wb[23:0];pi<=0;
                        weight_cache_valid<=0;activation_cache_valid<=0;
                        state<=(op==OP_COPY)?X_REQ:P_REQ;
                    end
                end
                GEOM0:begin
                    plane<=plane_calc;row_step<=32'(sh)*32'(iw);pad_offset<=32'(pt)*32'(iw);
                    kernel_area<=32'(kh)*32'(kw);output_area<=oh_calc*ow_calc;
                    oh<=oh_calc[15:0];ow<=ow_calc[15:0];
                    state<=GEOM_COUNT;
                end
                GEOM_COUNT:begin
                    input_bytes_reg<=plane*32'(ic);
                    reduction_reg<=kernel_area*32'(ic);
                    outputs_reg<=output_area*32'(oc_total);
                    state<=GEOM_CHECK;
                end
                GEOM_CHECK:begin
                    if(
                        kh==0||kw==0||kh>7||kw>7||ih==0||iw==0||ic==0||oc_total==0||
                        ih>255||iw>255||ic>255||oc_total>255||
                        (sh!=1&&sh!=2)||(sw!=1&&sw!=2)||pt>=kh||pbm>=kh||pl>=kw||pr>=kw||
                        32'(ih)+32'(pt)+32'(pbm)<32'(kh)||32'(iw)+32'(pl)+32'(pr)<32'(kw)||
                        {32'b0,xb}+{32'b0,input_bytes_reg}>64'(MEM_BYTES)||
                        outputs!=outputs_reg||
                        (op==OP_CONV&&count!=reduction_reg)||
                        (op==OP_MAXPOOL&&(count!=input_bytes_reg||oc_total!=ic))
                    )fail(8'd1);
                    else if(op==OP_CONV&&(wb[2:0]!=0||pb[2:0]!=0||stride[2:0]!=0||stride<count||stride>MEM_BYTES||
                        {32'b0,wb}+{48'b0,oc_total}*{32'b0,stride}>64'(MEM_BYTES)||
                        {32'b0,pb}+({48'b0,oc_total}<<4)>64'(MEM_BYTES)))fail(8'd1);
                    else if(op==OP_MAXPOOL&&(pb[2:0]!=0||{32'b0,pb}+16>64'(MEM_BYTES)))fail(8'd1);
                    else if(layer_count>=MAX_DESCRIPTORS)fail(8'd7);
                    else begin
                        row<=0;col<=0;weight_row<=wb[23:0];pi<=0;weight_cache_valid<=0;activation_cache_valid<=0;
                        state<=GEOM1;
                    end
                end
                GEOM1:begin
                    ox<=0;oy<=0;oc<=0;
                    channel_origin<=$signed(xb)-$signed(pad_offset)-$signed({16'b0,pl});
                    row_origin<=$signed(xb)-$signed(pad_offset)-$signed({16'b0,pl});
                    pixel_origin<=$signed(xb)-$signed(pad_offset)-$signed({16'b0,pl});
                    origin_y<=-$signed({2'b0,pt});origin_x<=-$signed({2'b0,pl});
                    state<=P_REQ;
                end
                P_REQ:if(mem_ready)state<=P_WAIT;
                P_WAIT:if(mem_rvalid)begin
                    if(!pi)begin p0<=mem_rdata;pi<=1;state<=P_REQ;end
                    else begin p1<=mem_rdata;state<=P_CHECK;end
                end
                P_CHECK:begin
                    if(p0[63]||p0[63:32]==0||p1[7:0]>62||p1[63:40]!=0||p1[39:32]!=8'd127||
                        ((op==OP_GEMM||op==OP_CONV)&&p1[31:24]!=8'h80)||
                        ((op==OP_RELU||op==OP_MAXPOOL)&&(p1[31:24]!=p1[23:16]||p0[31:0]!=0)))fail(8'd4);
                    else begin
                        acc<=$signed(p0[31:0]);mult<=$signed(p0[63:32]);shift<=p1[5:0];
                        zy<=$signed(p1[15:8]);zx<=$signed(p1[23:16]);col<=0;
                        win_kx<=0;win_ky<=0;win_ic<=0;win_lane<=0;
                        window_channel_origin<=pixel_origin;window_line_origin<=pixel_origin;win_addr<=pixel_origin;
                        pool_max<=-8'sd128;pool_any<=0;
                        state<=(op==OP_CONV)?WIN_PREP:((op==OP_MAXPOOL)?POOL_PREP:X_REQ);
                    end
                end
                X_REQ:if(mem_ready)state<=X_WAIT;
                X_WAIT:if(mem_rvalid)begin xword<=mem_rdata;if(op==OP_GEMM)state<=W_REQ;else state<=MAC;end
                WIN_PREP:begin
                    if(col+{29'b0,win_lane}>=count)begin
                        xword[win_lane*8+:8]<=8'b0;
                        if(win_lane==7)state<=W_REQ;else win_lane<=win_lane+3'd1;
                    end else if(!window_inside)begin
                        xword[win_lane*8+:8]<=zx;
                        advance_window();
                        if(win_lane==7||col+{29'b0,win_lane}+1>=count)state<=W_REQ;
                        else win_lane<=win_lane+3'd1;
                    end else if(activation_cache_hit)begin
                        xword[win_lane*8+:8]<=activation_cache_byte;
                        advance_window();
                        if(win_lane==7||col+{29'b0,win_lane}+1>=count)state<=W_REQ;
                        else win_lane<=win_lane+3'd1;
                    end else state<=WIN_REQ;
                end
                WIN_REQ:if(mem_ready)state<=WIN_WAIT;
                WIN_WAIT:if(mem_rvalid)begin
                    xword[win_lane*8+:8]<=mem_rdata[win_addr[2:0]*8+:8];
                    activation_cache_data[activation_cache_index]<=mem_rdata;
                    activation_cache_tag[activation_cache_index]<=win_addr[23:6];
                    activation_cache_valid[activation_cache_index]<=1;
                    advance_window();
                    if(win_lane==7||col+{29'b0,win_lane}+1>=count)state<=W_REQ;
                    else begin win_lane<=win_lane+3'd1;state<=WIN_PREP;end
                end
                POOL_PREP:begin
                    if(!window_inside)begin
                        if(win_ky+1==kh&&win_kx+1==kw)state<=POOL_DONE;
                        else begin advance_window();state<=POOL_PREP;end
                    end else if(activation_cache_hit)begin
                        if(!pool_any||activation_cache_byte>pool_max)pool_max<=activation_cache_byte;
                        pool_any<=1;
                        if(win_ky+1==kh&&win_kx+1==kw)state<=POOL_DONE;
                        else begin advance_window();state<=POOL_PREP;end
                    end else state<=POOL_REQ;
                end
                POOL_REQ:if(mem_ready)state<=POOL_WAIT;
                POOL_WAIT:if(mem_rvalid)begin
                    if(!pool_any||$signed(mem_rdata[win_addr[2:0]*8+:8])>pool_max)
                        pool_max<=$signed(mem_rdata[win_addr[2:0]*8+:8]);
                    pool_any<=1;
                    activation_cache_data[activation_cache_index]<=mem_rdata;
                    activation_cache_tag[activation_cache_index]<=win_addr[23:6];
                    activation_cache_valid[activation_cache_index]<=1;
                    if(win_ky+1==kh&&win_kx+1==kw)state<=POOL_DONE;
                    else begin advance_window();state<=POOL_PREP;end
                end
                POOL_DONE:begin
                    if(!pool_any)fail(8'd1);
                    else begin
                        acc<={{24{pool_max[7]}},pool_max}-{{24{zx[7]}},zx};
                        state<=Q_SEND;
                    end
                end
                W_REQ:begin
                    if(cached_weight)begin whex<=weight_cache[col[5:3]];state<=MAC;end
                    else if(mem_ready)state<=W_WAIT;
                end
                W_WAIT:if(mem_rvalid)begin
                    whex<=mem_rdata;
                    if(op==OP_CONV&&count<=64)weight_cache[col[5:3]]<=mem_rdata;
                    state<=MAC;
                end
                MAC:begin
                    if(op==OP_GEMM||op==OP_CONV)begin
                        if(acc_next>36'sd2147483647||acc_next< -36'sd2147483648)fail(8'd5);
                        else begin
                            acc<=acc_next[31:0];useful_macs<=useful_macs+{28'b0,valid_lanes};
                            if(col+8>=count)begin
                                if(op==OP_CONV&&count<=64)weight_cache_valid<=1;
                                state<=Q_SEND;
                            end
                            else begin col<=col+8;win_lane<=0;state<=(op==OP_CONV)?WIN_PREP:X_REQ;end
                        end
                    end else if(op==OP_RELU)begin acc<=scalar_acc;state<=Q_SEND;end
                    else begin result_byte<=scalar_x;state<=WRITE_OUT;end
                end
                Q_SEND:if(qready)state<=Q_WAIT;
                Q_WAIT:if(qout)begin result_byte<=qresult;state<=WRITE_OUT;end
                WRITE_OUT:if(mem_ready)state<=NEXT_OUT;
                NEXT_OUT:begin
                    if(row+1>=outputs)state<=NEXT_LAYER;
                    else begin
                        row<=row+1;col<=0;pi<=0;
                        if(spatial)begin
                            if(ox+1<ow)begin
                                ox<=ox+16'd1;origin_x<=origin_x+$signed({2'b0,sw});
                                pixel_origin<=pixel_origin+$signed({16'b0,sw});
                            end else if(oy+1<oh)begin
                                ox<=0;oy<=oy+16'd1;origin_x<=-$signed({2'b0,pl});origin_y<=origin_y+$signed({2'b0,sh});
                                row_origin<=row_origin+$signed(row_step);
                                pixel_origin<=row_origin+$signed(row_step);
                            end else begin
                                ox<=0;oy<=0;oc<=oc+16'd1;origin_x<=-$signed({2'b0,pl});origin_y<=-$signed({2'b0,pt});
                                if(op==OP_CONV)begin
                                    row_origin<=channel_origin;pixel_origin<=channel_origin;
                                    weight_row<=weight_row+stride[23:0];weight_cache_valid<=0;
                                end else begin
                                    channel_origin<=channel_origin+$signed(plane);
                                    row_origin<=channel_origin+$signed(plane);
                                    pixel_origin<=channel_origin+$signed(plane);
                                end
                            end
                            state<=P_REQ;
                        end else begin
                            weight_row<=weight_row+stride[23:0];
                            state<=(op==OP_GEMM)?P_REQ:X_REQ;
                        end
                    end
                end
                NEXT_LAYER:begin pc<=next_pc[23:0];di<=0;layer_count<=layer_count+1;layer_marker<=1;state<=D_REQ;end
                default:fail(8'd8);
            endcase
        end
    end
endmodule
