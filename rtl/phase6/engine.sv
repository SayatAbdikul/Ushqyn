// Isolated Phase 6 experiment. The production Phase 5 engine is unchanged.
// These caches are common hardware for every compared scheduling policy.
module v2_engine #(
    parameter integer PW_CACHE_ENTRIES = 256,
    parameter bit PARAM_CACHE = 1
) (
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
    wire spatial=(op==OP_CONV||op==OP_DWCONV||op==OP_MAXPOOL||op==OP_AVGPOOL);
    wire [31:0] plane_calc=32'(ih)*32'(iw);
    wire [31:0] numerator_h=32'(ih)+32'(pt)+32'(pbm)-32'(kh);
    wire [31:0] numerator_w=32'(iw)+32'(pl)+32'(pr)-32'(kw);
    // Large-stride average pools are restricted to a single full window.
    wire [31:0] oh_calc=(sh==2)?((numerator_h>>1)+1):((sh==1)?(numerator_h+1):1);
    wire [31:0] ow_calc=(sw==2)?((numerator_w>>1)+1):((sw==1)?(numerator_w+1):1);
    logic [15:0] oh,ow,ox,oy,oc,win_kx,win_ky,win_ic;
    logic [2:0] win_lane;
    logic [31:0] plane,row_step,pad_offset,kernel_area,output_area;
    logic [31:0] input_bytes_reg,reduction_reg,outputs_reg;
    logic [31:0] pixel_index,output_channel_base;
    logic signed [31:0] channel_origin,row_origin,pixel_origin,window_channel_origin,window_line_origin,win_addr;
    logic signed [17:0] origin_x,origin_y;
    wire signed [18:0] window_x=$signed(origin_x)+$signed({3'b0,win_kx});
    wire signed [18:0] window_y=$signed(origin_y)+$signed({3'b0,win_ky});
    wire window_inside=(window_x>=0 && window_y>=0 && window_x<$signed({3'b0,iw}) && window_y<$signed({3'b0,ih}));
    logic signed [7:0] pool_max;
    logic pool_any;
    logic signed [31:0] pool_sum;
    logic signed [7:0] clip_min,clip_max;
    // A small ordinary Conv visits output channels in pairs for each pixel.
    // One gathered activation tile is delivered to both channel reductions;
    // their filter tiles remain in separate banks across spatial positions.
    // Longer reductions retain the original output-stationary fallback.
    wire broadcast_mode=(op==OP_CONV&&oc_total>1&&count<=128);
    (* syn_ramstyle = "distributed_ram" *) logic [63:0] activation_tile[0:15];
    (* syn_ramstyle = "distributed_ram" *) logic [63:0] weight_cache[0:31];
    logic [31:0] weight_cache_valid;
    wire [4:0] weight_cache_index={(broadcast_mode&&oc[0]),col[6:3]};
    wire cached_weight=((op==OP_CONV||op==OP_DWCONV)&&count<=128&&weight_cache_valid[weight_cache_index]);
    // Four independently tagged input rows, eight SRAM words per row. The
    // data array has one synchronous access per cycle; a miss fills the word
    // returned by the sole physical scratchpad port. Address tags preserve
    // correctness for unaligned rows, wide rows and channel transitions.
    localparam integer LINE_ENTRIES = PW_CACHE_ENTRIES > 32 ? PW_CACHE_ENTRIES : 32;
    localparam integer LINE_BITS = $clog2(LINE_ENTRIES);
    wire pointwise_cache = PW_CACHE_ENTRIES != 0 && op==OP_CONV &&
        kh==1 && kw==1 && sh==1 && sw==1 && pt==0 && pbm==0 && pl==0 && pr==0;
    // Tag and data are read synchronously together. A full word-address tag
    // is required because the pointwise index contains no address bits.
    (* syn_ramstyle = "block_ram" *) logic [84:0] line_words[0:LINE_ENTRIES-1];
    logic [LINE_ENTRIES-1:0] line_valid;
    logic [84:0] line_read_word;
    wire [63:0] line_read_data = line_read_word[63:0];
    wire [LINE_BITS-1:0] line_index=pointwise_cache ? LINE_BITS'(win_ic) :
        LINE_BITS'({window_y[1:0],win_addr[5:3]});
    wire line_hit=line_valid[line_index]&&line_read_word[84:64]==win_addr[23:3];
    // Two tagged records preserve the existing paired-output traversal.
    logic [127:0] parameter_data[0:1];
    logic [15:0] parameter_tag[0:1];
    logic [1:0] parameter_valid;
    wire [15:0] parameter_channel=(op==OP_CONV||op==OP_DWCONV)?oc:16'd0;
    wire parameter_index=parameter_channel[0];
    wire parameter_hit=PARAM_CACHE && spatial && parameter_valid[parameter_index] &&
        parameter_tag[parameter_index]==parameter_channel;
    // Consume every available byte of a horizontal run from one 64-bit
    // buffer word in one cycle. A run never crosses the kernel row, memory
    // word, output MAC tile, reduction tail or right image boundary.
    logic [31:0] gather_limit;
    logic [3:0] win_take;
    logic [31:0] pool_limit;
    logic [3:0] pool_take;
    logic [63:0] pool_word;
    logic signed [31:0] pool_chunk_sum;
    logic signed [7:0] pool_chunk_max;
    logic signed [7:0] pool_sample;
    always_comb begin
        gather_limit=32'(kw)-32'(win_kx);
        if(gather_limit>32'd8-{29'b0,win_addr[2:0]})gather_limit=32'd8-{29'b0,win_addr[2:0]};
        if(gather_limit>32'd8-{29'b0,win_lane})gather_limit=32'd8-{29'b0,win_lane};
        if(gather_limit>count-col-{29'b0,win_lane})gather_limit=count-col-{29'b0,win_lane};
        if(window_inside&&gather_limit>32'(iw)-32'($unsigned(window_x)))
            gather_limit=32'(iw)-32'($unsigned(window_x));
        win_take=gather_limit[3:0];
        pool_limit=32'(kw)-32'(win_kx);
        if(pool_limit>32'd4)pool_limit=32'd4;
        if(pool_limit>32'd8-{29'b0,win_addr[2:0]})pool_limit=32'd8-{29'b0,win_addr[2:0]};
        if(window_inside&&pool_limit>32'(iw)-32'($unsigned(window_x)))
            pool_limit=32'(iw)-32'($unsigned(window_x));
        pool_take=pool_limit[3:0];
        pool_word=(state==POOL_WAIT)?mem_rdata:line_read_data;
        pool_chunk_sum=0;pool_chunk_max=-8'sd128;pool_sample=0;
        for(integer k=0;k<4;k=k+1)
            if(k<pool_take)begin
                pool_sample=$signed(pool_word[(32'(win_addr[2:0])+k)*8+:8]);
                pool_chunk_sum=pool_chunk_sum+{{24{pool_sample[7]}},pool_sample}-{{24{zx[7]}},zx};
                if(pool_sample>pool_chunk_max)pool_chunk_max=pool_sample;
            end
    end
    wire [3:0] advance_count=((state==WIN_PREP&&window_inside&&line_hit)||
        (state==WIN_WAIT&&mem_rvalid))?win_take:
        (((state==POOL_PREP&&window_inside&&line_hit)||
          (state==POOL_WAIT&&mem_rvalid))?pool_take:4'd1);
    wire signed [31:0] next_win_addr=(win_kx+{12'b0,advance_count}<kw)?win_addr+$signed({28'b0,advance_count}):
        ((win_ky+16'd1<kh)?window_line_origin+$signed({16'b0,iw}):window_channel_origin+$signed(plane));
    wire signed [18:0] next_window_y=(win_kx+{12'b0,advance_count}<kw)?window_y:
        ((win_ky+16'd1<kh)?window_y+19'sd1:$signed({origin_y[17],origin_y}));
    wire [LINE_BITS-1:0] next_line_index=pointwise_cache ? LINE_BITS'(win_ic+16'd1) :
        LINE_BITS'({next_window_y[1:0],next_win_addr[5:3]});
    wire line_advance=(state==WIN_PREP||state==POOL_PREP)&&
        (!window_inside||line_hit);
    wire [LINE_BITS-1:0] prefetch_index=(state==P_CHECK)?
        (pointwise_cache ? {LINE_BITS{1'b0}} : LINE_BITS'({origin_y[1:0],pixel_origin[5:3]})):
        ((line_advance||((state==WIN_WAIT||state==POOL_WAIT)&&mem_rvalid))?next_line_index:line_index);
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
    always_ff @(posedge clk) begin
        if ((state==WIN_WAIT||state==POOL_WAIT)&&mem_rvalid)
            line_words[line_index]<={win_addr[23:3],mem_rdata};
        if ((state==WIN_WAIT||state==POOL_WAIT)&&mem_rvalid&&next_line_index==line_index)
            line_read_word<={win_addr[23:3],mem_rdata};
        else line_read_word<=line_words[prefetch_index];
    end
    always_comb begin
        sum=0;valid_lanes=0;
        for (integer k=0;k<8;k=k+1) begin
            products[k]=$signed(xword[k*8+:8])*$signed(whex[k*8+:8]);
            if (col+k<count) begin sum=sum+{{20{products[k][15]}},products[k]}; valid_lanes=valid_lanes+4'd1; end
        end
        acc_next={{4{acc[31]}},acc}+sum;
        scalar_x=$signed(xword[row[2:0]*8+:8]);
        scalar_acc=(op==OP_CLIP ?
            (($signed(scalar_x)<$signed(clip_min)) ? {{24{clip_min[7]}},clip_min} :
             (($signed(scalar_x)>$signed(clip_max)) ? {{24{clip_max[7]}},clip_max} : {{24{scalar_x[7]}},scalar_x})) :
            (($signed(scalar_x)<$signed(zx)) ? {{24{zx[7]}},zx} : {{24{scalar_x[7]}},scalar_x})) - {{24{zx[7]}},zx};
        mem_req=0;mem_wr=0;address_full=0;mem_wdata=0;mem_wstrb=0;
        case(state)
            D_REQ:begin mem_req=1;address_full={40'b0,pc}+{58'b0,di,3'b0};end
            P_REQ:begin mem_req=!parameter_hit;address_full={32'b0,pb}+((op==OP_GEMM)?({32'b0,row}<<4):((op==OP_CONV||op==OP_DWCONV)?({48'b0,oc}<<4):0))+(pi?8:0);end
            X_REQ:begin mem_req=1;address_full={32'b0,xb}+((op==OP_GEMM)?{32'b0,col}:({32'b0,row}&64'hfffffffffffffff8));end
            WIN_REQ,POOL_REQ:begin mem_req=1;address_full={40'b0,win_addr[23:3],3'b0};end
            W_REQ:begin mem_req=!cached_weight;address_full={40'b0,weight_row}+{32'b0,col};end
            WRITE_OUT:begin
                mem_req=1;mem_wr=1;
                address_full=spatial?({32'b0,yb}+{32'b0,output_channel_base}+{32'b0,pixel_index}):
                    ({32'b0,yb}+{32'b0,row});
                mem_wstrb=8'b1<<address_full[2:0];mem_wdata={8{result_byte}};
                address_full=address_full&64'hfffffffffffffff8;
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
    task automatic advance_window(input logic [3:0] steps);
        begin
            if(win_kx+{12'b0,steps}<kw)begin
                win_kx<=win_kx+{12'b0,steps};win_addr<=win_addr+$signed({28'b0,steps});
            end
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
            oh<=0;ow<=0;ox<=0;oy<=0;oc<=0;win_kx<=0;win_ky<=0;win_ic<=0;win_lane<=0;weight_cache_valid<=0;line_valid<=0;parameter_valid<=0;
            plane<=0;row_step<=0;pad_offset<=0;kernel_area<=0;output_area<=0;
            input_bytes_reg<=0;reduction_reg<=0;outputs_reg<=0;pixel_index<=0;output_channel_base<=0;
            channel_origin<=0;row_origin<=0;pixel_origin<=0;
            window_channel_origin<=0;window_line_origin<=0;win_addr<=0;origin_x<=0;origin_y<=0;pool_max<=0;pool_any<=0;pool_sum<=0;clip_min<=-8'sd128;clip_max<=8'sd127;
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
                line_valid<=0;parameter_valid<=0;
                state<=IDLE;busy<=0;done<=0;error_code<=0;elapsed<=0;compute_cycles<=0;wait_cycles<=0;control_cycles<=0;useful_macs<=0;read_bytes<=0;write_bytes<=0;layer_count<=0;
            end else if(abort_run)begin state<=IDLE;busy<=0;done<=busy;error_code<=busy?8'd6:8'd0;line_valid<=0;parameter_valid<=0;end
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
                    else if(op!=OP_GEMM&&op!=OP_RELU&&op!=OP_COPY&&op!=OP_CONV&&op!=OP_MAXPOOL&&op!=OP_DWCONV&&op!=OP_AVGPOOL&&op!=OP_CLIP)fail(8'd3);
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
                    else if((op==OP_RELU||op==OP_CLIP)&&(pb[2:0]!=0||{32'b0,pb}+16>64'(MEM_BYTES)))fail(8'd1);
                    else if(layer_count>=MAX_DESCRIPTORS)fail(8'd7);
                    else begin
                        row<=0;col<=0;weight_row<=wb[23:0];pi<=0;
                        weight_cache_valid<=0;line_valid<=0;parameter_valid<=0;
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
                    reduction_reg<=(op==OP_DWCONV)?kernel_area:kernel_area*32'(ic);
                    outputs_reg<=output_area*32'(oc_total);
                    state<=GEOM_CHECK;
                end
                GEOM_CHECK:begin
                    if(
                        kh==0||kw==0||kh>31||kw>31||ih==0||iw==0||ic==0||oc_total==0||
                        ih>255||iw>255||ic>1024||oc_total>1024||
                        (((sh!=1&&sh!=2)||(sw!=1&&sw!=2))&&op!=OP_AVGPOOL)||
                        (op==OP_AVGPOOL&&((sh!=1&&sh!=2&&sh!=kh)||(sw!=1&&sw!=2&&sw!=kw)||
                            (sh>2&&numerator_h>=32'(sh))||(sw>2&&numerator_w>=32'(sw))||
                            pt!=0||pbm!=0||pl!=0||pr!=0))||
                        pt>=kh||pbm>=kh||pl>=kw||pr>=kw||
                        32'(ih)+32'(pt)+32'(pbm)<32'(kh)||32'(iw)+32'(pl)+32'(pr)<32'(kw)||
                        {32'b0,xb}+{32'b0,input_bytes_reg}>64'(MEM_BYTES)||
                        outputs!=outputs_reg||
                        ((op==OP_CONV||op==OP_DWCONV)&&count!=reduction_reg)||
                        (op==OP_DWCONV&&oc_total!=ic)||
                        ((op==OP_MAXPOOL||op==OP_AVGPOOL)&&(count!=input_bytes_reg||oc_total!=ic))
                    )fail(8'd1);
                    else if((op==OP_CONV||op==OP_DWCONV)&&(wb[2:0]!=0||pb[2:0]!=0||stride[2:0]!=0||stride<count||stride>MEM_BYTES||
                        {32'b0,wb}+{48'b0,oc_total}*{32'b0,stride}>64'(MEM_BYTES)||
                        {32'b0,pb}+({48'b0,oc_total}<<4)>64'(MEM_BYTES)))fail(8'd1);
                    else if((op==OP_MAXPOOL||op==OP_AVGPOOL)&&(pb[2:0]!=0||{32'b0,pb}+16>64'(MEM_BYTES)))fail(8'd1);
                    else if(layer_count>=MAX_DESCRIPTORS)fail(8'd7);
                    else begin
                        row<=0;col<=0;weight_row<=wb[23:0];pi<=0;weight_cache_valid<=0;line_valid<=0;parameter_valid<=0;
                        state<=GEOM1;
                    end
                end
                GEOM1:begin
                    ox<=0;oy<=0;oc<=0;pixel_index<=0;output_channel_base<=0;
                    channel_origin<=$signed(xb)-$signed(pad_offset)-$signed({16'b0,pl});
                    row_origin<=$signed(xb)-$signed(pad_offset)-$signed({16'b0,pl});
                    pixel_origin<=$signed(xb)-$signed(pad_offset)-$signed({16'b0,pl});
                    origin_y<=-$signed({2'b0,pt});origin_x<=-$signed({2'b0,pl});
                    state<=P_REQ;
                end
                P_REQ:begin
                    if(parameter_hit)begin
                        {p1,p0}<=parameter_data[parameter_index];state<=P_CHECK;
                    end else if(mem_ready)state<=P_WAIT;
                end
                P_WAIT:if(mem_rvalid)begin
                    if(!pi)begin p0<=mem_rdata;pi<=1;state<=P_REQ;end
                    else begin
                        p1<=mem_rdata;state<=P_CHECK;
                        if(PARAM_CACHE&&spatial)begin
                            parameter_data[parameter_index]<={mem_rdata,p0};
                            parameter_tag[parameter_index]<=parameter_channel;
                            parameter_valid[parameter_index]<=1;
                        end
                    end
                end
                P_CHECK:begin
                    if(p0[63]||p0[63:32]==0||p1[7:0]>62||p1[63:40]!=0||
                        ((op==OP_GEMM||op==OP_CONV||op==OP_DWCONV)&&
                            (p1[31:24]!=8'h80||p1[39:32]!=8'd127))||
                        (op==OP_RELU&&
                            (p1[31:24]!=p1[23:16]||p1[39:32]!=8'd127||p0[31:0]!=0))||
                        ((op==OP_CLIP||op==OP_AVGPOOL||op==OP_MAXPOOL)&&
                            (p0[31:0]!=0||$signed(p1[31:24])>$signed(p1[39:32]))))fail(8'd4);
                    else begin
                        acc<=$signed(p0[31:0]);mult<=$signed(p0[63:32]);shift<=p1[5:0];
                        zy<=$signed(p1[15:8]);zx<=$signed(p1[23:16]);col<=0;
                        clip_min<=$signed(p1[31:24]);clip_max<=$signed(p1[39:32]);
                        win_kx<=0;win_ky<=0;win_ic<=0;win_lane<=0;
                        window_channel_origin<=pixel_origin;window_line_origin<=pixel_origin;win_addr<=pixel_origin;
                        pool_max<=-8'sd128;pool_any<=0;pool_sum<=0;
                        state<=(op==OP_CONV||op==OP_DWCONV)?WIN_PREP:((op==OP_MAXPOOL||op==OP_AVGPOOL)?POOL_PREP:X_REQ);
                    end
                end
                X_REQ:if(mem_ready)state<=X_WAIT;
                X_WAIT:if(mem_rvalid)begin xword<=mem_rdata;if(op==OP_GEMM)state<=W_REQ;else state<=MAC;end
                WIN_PREP:begin
                    if(broadcast_mode&&oc[0])begin
                        xword<=activation_tile[col[6:3]];
                        state<=W_REQ;
                    end else if(col+{29'b0,win_lane}>=count)begin
                        for(integer k=0;k<8;k=k+1)
                            if(k>=win_lane)xword[k*8+:8]<=8'b0;
                        state<=W_REQ;
                    end else if(!window_inside)begin
                        xword[win_lane*8+:8]<=zx;
                        advance_window(4'd1);
                        if(win_lane==7||col+{29'b0,win_lane}+1>=count)state<=W_REQ;
                        else win_lane<=win_lane+3'd1;
                    end else if(line_hit)begin
                        for(integer k=0;k<8;k=k+1)
                            if(k<win_take)
                                xword[(32'(win_lane)+k)*8+:8]<=line_read_data[(32'(win_addr[2:0])+k)*8+:8];
                        advance_window(win_take);
                        if(win_lane+win_take>=8||col+{29'b0,win_lane}+{28'b0,win_take}>=count)state<=W_REQ;
                        else win_lane<=win_lane+win_take[2:0];
                    end
                    else state<=WIN_REQ;
                end
                WIN_REQ:if(mem_ready)state<=WIN_WAIT;
                WIN_WAIT:if(mem_rvalid)begin
                    for(integer k=0;k<8;k=k+1)
                        if(k<win_take)
                            xword[(32'(win_lane)+k)*8+:8]<=mem_rdata[(32'(win_addr[2:0])+k)*8+:8];
                    line_valid[line_index]<=1;
                    advance_window(win_take);
                    if(win_lane+win_take>=8||col+{29'b0,win_lane}+{28'b0,win_take}>=count)state<=W_REQ;
                    else begin win_lane<=win_lane+win_take[2:0];state<=WIN_PREP;end
                end
                POOL_PREP:begin
                    if(!window_inside)begin
                        if(win_ky+1==kh&&win_kx+1==kw)state<=POOL_DONE;
                        else begin advance_window(4'd1);state<=POOL_PREP;end
                    end else if(line_hit)begin
                        if(op==OP_AVGPOOL)pool_sum<=pool_sum+pool_chunk_sum;
                        else if(!pool_any||pool_chunk_max>pool_max)pool_max<=pool_chunk_max;
                        pool_any<=1;
                        if(win_ky+1==kh&&win_kx+{12'b0,pool_take}>=kw)state<=POOL_DONE;
                        else begin advance_window(pool_take);state<=POOL_PREP;end
                    end
                    else state<=POOL_REQ;
                end
                POOL_REQ:if(mem_ready)state<=POOL_WAIT;
                POOL_WAIT:if(mem_rvalid)begin
                    if(op==OP_AVGPOOL)pool_sum<=pool_sum+pool_chunk_sum;
                    else if(!pool_any||pool_chunk_max>pool_max)pool_max<=pool_chunk_max;
                    pool_any<=1;
                    line_valid[line_index]<=1;
                    if(win_ky+1==kh&&win_kx+{12'b0,pool_take}>=kw)state<=POOL_DONE;
                    else begin advance_window(pool_take);state<=POOL_PREP;end
                end
                POOL_DONE:begin
                    if(!pool_any)fail(8'd1);
                    else begin
                        acc<=(op==OP_AVGPOOL)?pool_sum:({{24{pool_max[7]}},pool_max}-{{24{zx[7]}},zx});
                        state<=Q_SEND;
                    end
                end
                W_REQ:begin
                    if(cached_weight)begin whex<=weight_cache[weight_cache_index];state<=MAC;end
                    else if(mem_ready)state<=W_WAIT;
                end
                W_WAIT:if(mem_rvalid)begin
                    whex<=mem_rdata;
                    if((op==OP_CONV||op==OP_DWCONV)&&count<=128)begin
                        weight_cache[weight_cache_index]<=mem_rdata;
                        weight_cache_valid[weight_cache_index]<=1;
                    end
                    state<=MAC;
                end
                MAC:begin
                    if(op==OP_GEMM||op==OP_CONV||op==OP_DWCONV)begin
                        if(acc_next>36'sd2147483647||acc_next< -36'sd2147483648)fail(8'd5);
                        else begin
                            if(broadcast_mode&&!oc[0])activation_tile[col[6:3]]<=xword;
                            acc<=acc_next[31:0];useful_macs<=useful_macs+{28'b0,valid_lanes};
                            if(col+8>=count)begin
                                state<=Q_SEND;
                            end
                            else begin col<=col+8;win_lane<=0;state<=(op==OP_CONV||op==OP_DWCONV)?WIN_PREP:X_REQ;end
                        end
                    end else if(op==OP_RELU||op==OP_CLIP)begin acc<=scalar_acc;state<=Q_SEND;end
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
                            if(broadcast_mode&&!oc[0]&&oc+16'd1<oc_total)begin
                                // Same input pixel, second output channel.
                                oc<=oc+16'd1;
                                output_channel_base<=output_channel_base+output_area;
                                weight_row<=weight_row+stride[23:0];
                            end else if(pixel_index+1<output_area)begin
                                // Both channels have consumed this activation
                                // window (or this is an unpaired last channel).
                                if(broadcast_mode&&oc[0])begin
                                    oc<=oc-16'd1;
                                    output_channel_base<=output_channel_base-output_area;
                                    weight_row<=weight_row-stride[23:0];
                                end
                                pixel_index<=pixel_index+1;
                                if(ox+1<ow)begin
                                    ox<=ox+16'd1;origin_x<=origin_x+$signed({2'b0,sw});
                                    pixel_origin<=pixel_origin+$signed({16'b0,sw});
                                end else begin
                                    ox<=0;oy<=oy+16'd1;origin_x<=-$signed({2'b0,pl});origin_y<=origin_y+$signed({2'b0,sh});
                                    row_origin<=row_origin+$signed(row_step);
                                    pixel_origin<=row_origin+$signed(row_step);
                                end
                            end else begin
                                // Complete this channel group and return to
                                // the first spatial position of the next one.
                                ox<=0;oy<=0;pixel_index<=0;oc<=oc+16'd1;
                                output_channel_base<=output_channel_base+output_area;
                                origin_x<=-$signed({2'b0,pl});origin_y<=-$signed({2'b0,pt});
                                if(op==OP_CONV||op==OP_DWCONV)begin
                                    row_origin<=channel_origin;pixel_origin<=channel_origin;
                                    weight_row<=weight_row+stride[23:0];weight_cache_valid<=0;
                                    if(op==OP_DWCONV)begin
                                        channel_origin<=channel_origin+$signed(plane);
                                        row_origin<=channel_origin+$signed(plane);
                                        pixel_origin<=channel_origin+$signed(plane);
                                    end
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
