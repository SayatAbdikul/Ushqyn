// Isolated balanced, bounded average-pool reduction.
// Exact INT8 epilogue with mandatory per-descriptor LUT reload.
// Isolated one-result-per-cycle SIMD output drain; parent hash recorded by driver.
// Isolated parameterized optimization experiments; frozen releases are unchanged.
// Phase 6 direct-writeback candidate, derived from the frozen spatial engine.
// Consume the requantizer output and advance on the accepted SRAM write.
// Experimental geometry width proof. Descriptor arithmetic and output values
// are unchanged for every valid ABI descriptor; existing full-width checks
// reject malformed fields before any shortened product can be used.
// These caches are common hardware for every compared scheduling policy.
module v2_engine #(
    parameter integer PW_CACHE_ENTRIES = 256,
    parameter bit PARAM_CACHE = 1,
    parameter bit SPATIAL_PW = 1,
    parameter bit PW_REGISTERED_WEIGHT = 1,
    parameter bit PW_FETCH_AHEAD = 1,
    parameter bit SCALAR_WORD_REUSE = 1,
    parameter bit PW_WORD_CACHE = 1,
    parameter bit BALANCED_SUM = 1,
    parameter integer ACC_WORK_BITS = 33,
    parameter bit SCALAR_LUT = 1,
    parameter bit PARAM_FAST = 0,
    parameter bit SPATIAL_DW = 1,
    parameter bit VALID_IN_BRAM = 1,
    parameter bit PW_HIT_BYPASS = 0,
    parameter bit DW_WORD_REUSE = 1,
    parameter bit RQ_FAST = 1,
    parameter bit VECTOR_LUT = 0
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
    typedef enum logic [5:0] {IDLE,D_REQ,D_WAIT,D_CHECK,P_REQ,P_WAIT,P_CHECK,X_REQ,X_WAIT,W_REQ,W_WAIT,MAC,Q_SEND,Q_WAIT,WRITE_OUT,NEXT_OUT,NEXT_LAYER,WIN_PREP,WIN_REQ,WIN_WAIT,POOL_PREP,POOL_REQ,POOL_WAIT,POOL_DONE,GEOM0,GEOM1,GEOM_COUNT,GEOM_CHECK,OTHER_CHECK,PW_X_REQ,PW_X_WAIT,PW_X2_REQ,PW_X2_WAIT,PW_W_REQ,PW_W_WAIT,PW_MAC,LUT_BUILD,LUT_SEND,LUT_WAIT,LUT_WRITE,DW_PREP,LINE_CLEAR,LUT_WORD_WRITE,Q_STREAM,F_L_REQ,F_L_WAIT,F_L_BYTES} state_t;
    state_t state;
    logic [63:0] d[0:7];
    logic [2:0] di;
    logic pi;
    logic [23:0] pc;
    logic [31:0] row,col;
    logic [7:0] mac_lane_mask;
    // Operand fetch always precedes MAC.  Register tail validity there so
    // the col/count comparison does not feed the multiplier reduction tree.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) mac_lane_mask <= 8'b0;
        else if (state != MAC) begin
            for (integer lane = 0; lane < 8; lane = lane + 1)
                mac_lane_mask[lane] <= (col + 32'(lane) < count);
        end
    end
    logic [23:0] weight_row;
    logic [63:0] xword,whex,p0,p1;
    wire [7:0] op=d[0][47:40];
    wire [31:0] xb=d[1][31:0], yb=d[1][63:32], wb=d[2][31:0], pb=d[2][63:32];
    wire [31:0] count=d[3][31:0], outputs=d[3][63:32], stride=d[4][31:0], next_pc=d[4][63:32];
    wire [15:0] kh=d[5][15:0],kw=d[5][31:16],sh=d[5][47:32],sw=d[5][63:48];
    wire [15:0] pt=d[6][15:0],pbm=d[6][31:16],pl=d[6][47:32],pr=d[6][63:48];
    wire [15:0] ih=d[7][15:0],iw=d[7][31:16],ic=d[7][47:32],oc_total=d[7][63:48];
    wire spatial=(op==OP_CONV||op==OP_DWCONV||op==OP_MAXPOOL||op==OP_AVGPOOL);
    // Legal dimensions: ih,iw<=255; kh,kw<=31; ic,oc<=1024;
    // stride<=32768; and each output axis is at most 285. GEOM_CHECK keeps
    // the original full-field rejection predicates for larger encodings.
    wire [15:0] plane_calc=ih[7:0]*iw[7:0];
    wire [31:0] numerator_h=32'(ih)+32'(pt)+32'(pbm)-32'(kh);
    wire [31:0] numerator_w=32'(iw)+32'(pl)+32'(pr)-32'(kw);
    // Large-stride average pools are restricted to a single full window.
    wire [31:0] oh_calc=(sh==2)?((numerator_h>>1)+1):((sh==1)?(numerator_h+1):1);
    wire [31:0] ow_calc=(sw==2)?((numerator_w>>1)+1):((sw==1)?(numerator_w+1):1);
    wire [12:0] row_step_calc=sh[4:0]*iw[7:0];
    wire [12:0] pad_offset_calc=pt[4:0]*iw[7:0];
    wire [9:0] kernel_area_calc=kh[4:0]*kw[4:0];
    wire [17:0] output_area_calc=oh_calc[8:0]*ow_calc[8:0];
    wire [26:0] input_bytes_calc=plane[15:0]*ic[10:0];
    wire [20:0] reduction_calc=kernel_area[9:0]*ic[10:0];
    wire [28:0] outputs_calc=output_area[17:0]*oc_total[10:0];
    wire [31:0] gemm_weight_span=outputs[15:0]*stride[15:0];
    wire [26:0] spatial_weight_span=oc_total[10:0]*stride[15:0];
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
    // Each accepted pool contains at most 31*31 samples. One centered
    // INT8 sample is in [-255,255], hence the total is in [-245055,245055]
    // and cannot overflow a signed 19-bit accumulator.
    logic signed [18:0] pool_sum;
    logic signed [7:0] clip_min,clip_max;
    // A small ordinary Conv visits output channels in pairs for each pixel.
    // One gathered activation tile is delivered to both channel reductions;
    // their filter tiles remain in separate banks across spatial positions.
    // Longer reductions retain the original output-stationary fallback.
    // Decode once after fetching a descriptor. Keeping this wide predicate
    // out of the shared multiplier path preserves the production core clock.
    logic spatial_pw;
    logic spatial_dw;
    logic [7:0] dw_mask;
    logic [3:0] dw_skip,dw_take;
    logic [3:0] dw_skip_calc,dw_take_calc;
    logic [7:0] dw_mask_calc;
    always_comb begin
        dw_skip_calc=0;dw_take_calc=0;dw_mask_calc=0;
        for(integer k=0;k<8;k=k+1)begin
            if(k<pw_take&&window_y>=0&&window_y<$signed({3'b0,ih})&&
                window_x+k>=0&&window_x+k<$signed({3'b0,iw}))begin
                dw_mask_calc[k]=1;dw_take_calc=dw_take_calc+1;
            end
            if(k<pw_take&&window_x+k<0)dw_skip_calc=dw_skip_calc+1;
        end
    end
    logic scalar_word_reuse;
    logic scalar_lut;
    logic vector_lut;
    logic [7:0] lut_index,lut_value;
    wire fused_activation = d[0][48];
    wire [23:0] fused_table_base = {6'b0,d[0][63:49],3'b0};
    wire fusion_abi_valid = d[0][63:48]==0 ||
        (fused_activation&&(op==OP_CONV||op==OP_DWCONV)&&fused_table_base<=24'd32512);
    wire output_state = state==Q_WAIT||state==Q_STREAM;
    logic fused_result_valid;
    wire fused_lookup_ready = !fused_result_valid||mem_ready;
    wire output_valid = fused_activation ? fused_result_valid : qout;
    wire [7:0] output_result = fused_activation ? lut_value : qresult;
    wire requant_output_ready = output_state&&
        (fused_activation?fused_lookup_ready:mem_ready);
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)fused_result_valid<=0;
        else if(abort_run||clear_counters||!output_state)fused_result_valid<=0;
        else if(fused_activation&&fused_lookup_ready)fused_result_valid<=qout;
    end
    wire [63:0] lut_word;
    logic [7:0] lut_word_strb;
    (* syn_ramstyle = "block_ram" *) logic [7:0] activation_lut[0:255];
    wire lut_building=state==LUT_BUILD||state==LUT_SEND||state==LUT_WAIT;
    wire [2:0] lut_byte_index=row[2:0]+((state==LUT_WRITE&&mem_ready)?3'd1:3'd0);
    wire [7:0] lut_read_index=(state==X_WAIT)?mem_rdata[row[2:0]*8+:8]:xword[lut_byte_index*8+:8];
    always_ff @(posedge clk) begin
        if(state==F_L_BYTES)activation_lut[lut_index]<=xword[lut_index[2:0]*8+:8];
        else if(SCALAR_LUT&&state==LUT_WAIT&&qout)activation_lut[lut_index]<=qresult;
        if(fused_activation&&output_state)begin
            if(qout&&fused_lookup_ready)lut_value<=activation_lut[qresult];
        end else lut_value<=activation_lut[lut_read_index];
    end
    generate for(genvar lut_lane=0;lut_lane<8;lut_lane=lut_lane+1)begin:parallel_lut
        (* syn_ramstyle = "block_ram" *) logic [7:0] table_words[0:255];
        logic [7:0] result_byte_lane;
        always_ff @(posedge clk)begin
            if(VECTOR_LUT&&state==LUT_WAIT&&qout)table_words[lut_index]<=qresult;
            if(VECTOR_LUT&&state==X_WAIT&&mem_rvalid)
                result_byte_lane<=table_words[mem_rdata[lut_lane*8+:8]];
        end
        assign lut_word[lut_lane*8+:8]=result_byte_lane;
    end endgenerate
    always_comb begin
        for(integer lut_byte=0;lut_byte<8;lut_byte=lut_byte+1)
            lut_word_strb[lut_byte]=(row+32'(lut_byte)<outputs);
    end
    logic signed [7:0] pw_weight;
    logic pw_prefetch_pending;
    wire spatial_pw_decode=SPATIAL_PW&&op==OP_CONV&&kh==1&&kw==1&&sh==1&&sw==1&&
        pt==0&&pbm==0&&pl==0&&pr==0&&count<=256;
    wire broadcast_mode=(!spatial_pw&&op==OP_CONV&&oc_total>1&&count<=128);
    // Eight pixels share a weight, with one accumulator per pixel. Keep 36
    // bits between input channels and check INT32 range at the original
    // eight-channel reduction boundaries, preserving overflow semantics.
    logic signed [ACC_WORK_BITS-1:0] pw_acc[0:7],pw_next[0:7];
    logic [23:0] pw_input_addr;
    logic [3:0] pw_take;
    logic [2:0] pw_lane;
    logic [3:0] q_issue_lane;
    logic pw_overflow;
    (* syn_ramstyle = "distributed_ram" *) logic [63:0] activation_tile[0:15];
    (* syn_ramstyle = "distributed_ram" *) logic [63:0] weight_cache[0:31];
    logic [31:0] weight_cache_valid;
    wire [4:0] weight_cache_index=spatial_pw?col[7:3]:{(broadcast_mode&&oc[0]),col[6:3]};
    wire cached_weight=((op==OP_CONV||op==OP_DWCONV)&&count<=(spatial_pw?256:128)&&weight_cache_valid[weight_cache_index]);
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
    (* syn_ramstyle = "block_ram" *) logic [85:0] line_words[0:LINE_ENTRIES-1];
    logic [LINE_ENTRIES-1:0] line_valid;
    logic [85:0] line_read_word;
    logic [LINE_BITS-1:0] line_clear_index;
    wire [63:0] line_read_data = line_read_word[63:0];
    wire [LINE_BITS-1:0] line_index=pointwise_cache ? LINE_BITS'(win_ic) :
        LINE_BITS'({window_y[1:0],win_addr[5:3]});
    wire line_hit=(VALID_IN_BRAM?line_read_word[85]:line_valid[line_index])&&line_read_word[84:64]==win_addr[23:3];
    wire [23:0] pw_next_input_addr=pw_input_addr+plane[23:0];
    wire [23:0] dw_next_input_addr=win_addr[23:0]+{20'b0,dw_skip_calc};
    wire [LINE_BITS-1:0] pw_cache_index=(DW_WORD_REUSE&&spatial_dw)?
        LINE_BITS'(pw_input_addr[3+:LINE_BITS]):LINE_BITS'(col);
    wire [LINE_BITS-1:0] pw_next_cache_index=LINE_BITS'(col+1);
    wire pw_cache_hit=PW_WORD_CACHE&&(spatial_pw||(DW_WORD_REUSE&&spatial_dw))&&
        (VALID_IN_BRAM?line_read_word[85]:line_valid[pw_cache_index])&&
        line_read_word[84:64]==pw_input_addr[23:3];
    wire pw_next_cache_hit=PW_WORD_CACHE&&spatial_pw&&(VALID_IN_BRAM?line_read_word[85]:line_valid[pw_next_cache_index])&&
        line_read_word[84:64]==pw_next_input_addr[23:3];
    wire pw_next_crosses=32'(pw_next_input_addr[2:0])+32'(pw_take)>8;
    // Keep the current lookup stable while a first-word request is stalled.
    // Other fetch states pre-read the next channel for MAC/request overlap.
    wire [LINE_BITS-1:0] pw_prefetch_index=(DW_WORD_REUSE&&spatial_dw)?
        ((state==DW_PREP)?LINE_BITS'(dw_next_input_addr[3+:LINE_BITS]):pw_cache_index):
        (state==P_CHECK)?LINE_BITS'(0):
        ((state==PW_X_REQ&&!(pw_cache_hit||mem_ready))?pw_cache_index:
         ((state==PW_X_REQ||state==PW_X_WAIT||state==PW_X2_REQ||state==PW_X2_WAIT||
           state==PW_W_REQ||state==PW_W_WAIT||state==PW_MAC)?pw_next_cache_index:pw_cache_index));
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
    logic signed [10:0] pool_chunk_sum;
    logic signed [8:0] pool_delta [0:3];
    logic signed [9:0] pool_pair [0:1];
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
        pool_chunk_max=-8'sd128;pool_sample=0;
        for(integer k=0;k<4;k=k+1)begin
            pool_delta[k]=0;
            if(k<pool_take)begin
                pool_sample=$signed(pool_word[(32'(win_addr[2:0])+k)*8+:8]);
                pool_delta[k]=$signed({pool_sample[7],pool_sample})-
                              $signed({zx[7],zx});
                if(pool_sample>pool_chunk_max)pool_chunk_max=pool_sample;
            end
        end
        pool_pair[0]=$signed({pool_delta[0][8],pool_delta[0]})+
                     $signed({pool_delta[1][8],pool_delta[1]});
        pool_pair[1]=$signed({pool_delta[2][8],pool_delta[2]})+
                     $signed({pool_delta[3][8],pool_delta[3]});
        pool_chunk_sum=$signed({pool_pair[0][9],pool_pair[0]})+
                       $signed({pool_pair[1][9],pool_pair[1]});
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
    logic signed [ACC_WORK_BITS-1:0] sum,acc_next;
    logic signed [15:0] masked_products[0:7];
    logic signed [16:0] pair_sum[0:3];
    logic signed [17:0] quad_sum[0:1];
    logic signed [18:0] tree_sum;
    logic [3:0] valid_lanes;
    logic qvalid,qready,qout;
    logic signed [7:0] qresult;
    logic signed [7:0] scalar_x;
    logic signed [31:0] scalar_acc;
    logic [63:0] address_full;
    integer j;
    // One explicit RAM write port and one read address preserve BSRAM inference.
    wire line_fill_pw=PW_WORD_CACHE&&(state==PW_X_WAIT||state==PW_X2_WAIT);
    wire line_fill=mem_rvalid&&(line_fill_pw||state==WIN_WAIT||state==POOL_WAIT);
    wire [LINE_BITS-1:0] line_fill_index=line_fill_pw?
        ((DW_WORD_REUSE&&spatial_dw&&state==PW_X2_WAIT)?
            LINE_BITS'((pw_input_addr[23:3]+21'd1)):pw_cache_index):line_index;
    wire [20:0] line_fill_tag=line_fill_pw?
        (pw_input_addr[23:3]+(state==PW_X2_WAIT?21'd1:21'd0)):win_addr[23:3];
    wire [LINE_BITS-1:0] line_read_index=(PW_WORD_CACHE&&
        (spatial_pw||(DW_WORD_REUSE&&spatial_dw)))?pw_prefetch_index:prefetch_index;
    wire clearing_line=VALID_IN_BRAM&&state==LINE_CLEAR;
    wire [LINE_BITS-1:0] line_write_index=clearing_line?line_clear_index:line_fill_index;
    wire [85:0] line_write_data=clearing_line?86'd0:{1'b1,line_fill_tag,mem_rdata};
    always_ff @(posedge clk) begin
        if(clearing_line||line_fill)line_words[line_write_index]<=line_write_data;
        if (!(PW_WORD_CACHE&&spatial_pw)&&(state==WIN_WAIT||state==POOL_WAIT)&&mem_rvalid&&next_line_index==line_index)
            line_read_word<={1'b1,win_addr[23:3],mem_rdata};
        else line_read_word<=line_words[line_read_index];
    end
    always_comb begin
        sum=0;valid_lanes=0;pw_overflow=0;
        for (integer k=0;k<8;k=k+1) begin
            products[k]=$signed((spatial_dw&&!dw_mask[k])?zx:xword[k*8+:8])*$signed((spatial_pw||spatial_dw)?(PW_REGISTERED_WEIGHT?pw_weight:whex[col[2:0]*8+:8]):whex[k*8+:8]);
            pw_next[k]=pw_acc[k]+{{(ACC_WORK_BITS-16){products[k][15]}},products[k]};
            if(k<pw_take&&(pw_next[k]>36'sd2147483647||pw_next[k]< -36'sd2147483648))pw_overflow=1;
            masked_products[k]=mac_lane_mask[k]?products[k]:16'sd0;
            if (mac_lane_mask[k]) begin sum=sum+{{(ACC_WORK_BITS-16){products[k][15]}},products[k]}; valid_lanes=valid_lanes+4'd1; end
        end
        for(integer k=0;k<4;k=k+1)
            pair_sum[k]=$signed({masked_products[2*k][15],masked_products[2*k]})+
                $signed({masked_products[2*k+1][15],masked_products[2*k+1]});
        for(integer k=0;k<2;k=k+1)
            quad_sum[k]=$signed({pair_sum[2*k][16],pair_sum[2*k]})+
                $signed({pair_sum[2*k+1][16],pair_sum[2*k+1]});
        tree_sum=$signed({quad_sum[0][17],quad_sum[0]})+$signed({quad_sum[1][17],quad_sum[1]});
        if(BALANCED_SUM)sum={{(ACC_WORK_BITS-19){tree_sum[18]}},tree_sum};
        acc_next={{(ACC_WORK_BITS-32){acc[31]}},acc}+sum;
        scalar_x=$signed((SCALAR_LUT&&lut_building)?lut_index:xword[row[2:0]*8+:8]);
        scalar_acc=(op==OP_CLIP ?
            (($signed(scalar_x)<$signed(clip_min)) ? {{24{clip_min[7]}},clip_min} :
             (($signed(scalar_x)>$signed(clip_max)) ? {{24{clip_max[7]}},clip_max} : {{24{scalar_x[7]}},scalar_x})) :
            (($signed(scalar_x)<$signed(zx)) ? {{24{zx[7]}},zx} : {{24{scalar_x[7]}},scalar_x})) - {{24{zx[7]}},zx};
        mem_req=0;mem_wr=0;address_full=0;mem_wdata=0;mem_wstrb=0;
        case(state)
            D_REQ:begin mem_req=1;address_full={40'b0,pc}+{58'b0,di,3'b0};end
            P_REQ:begin mem_req=!parameter_hit;address_full={32'b0,pb}+((op==OP_GEMM)?({32'b0,row}<<4):((op==OP_CONV||op==OP_DWCONV)?({48'b0,oc}<<4):0))+(pi?8:0);end
            F_L_REQ:begin
                mem_req=1;address_full={40'b0,fused_table_base}+{56'b0,lut_index[7:3],3'b0};
            end
            X_REQ:begin mem_req=1;address_full={32'b0,xb}+((op==OP_GEMM)?{32'b0,col}:({32'b0,row}&64'hfffffffffffffff8));end
            WIN_REQ,POOL_REQ:begin mem_req=1;address_full={40'b0,win_addr[23:3],3'b0};end
            // A prefetched input read is side-effect-free. On arithmetic
            // overflow the engine still reports the same failure; any
            // accepted speculative SRAM response is discarded in IDLE.
            PW_MAC:if(PW_FETCH_AHEAD && !spatial_dw && col+1<count)begin
                mem_req=!pw_next_cache_hit||pw_next_crosses;
                address_full={40'b0,pw_next_input_addr[23:3],3'b0}+
                    ((pw_next_cache_hit&&pw_next_crosses)?64'd8:64'd0);
            end
            PW_X_REQ:begin mem_req=pw_prefetch_pending||!pw_cache_hit;address_full={40'b0,pw_input_addr[23:3],3'b0};end
            PW_X2_REQ:begin mem_req=1;address_full={40'b0,pw_input_addr[23:3],3'b0}+64'd8;end
            PW_W_REQ:begin mem_req=!cached_weight;address_full={40'b0,weight_row}+({32'b0,col}&64'hfffffffffffffff8);end
            W_REQ:begin mem_req=!cached_weight;address_full={40'b0,weight_row}+{32'b0,col};end
            WRITE_OUT,Q_WAIT,Q_STREAM,LUT_WRITE,LUT_WORD_WRITE:begin
                mem_req=(state==WRITE_OUT)||(state==LUT_WRITE)||(state==LUT_WORD_WRITE)||output_valid;mem_wr=1;
                address_full=spatial?({32'b0,yb}+{32'b0,output_channel_base}+{32'b0,pixel_index}):
                    ({32'b0,yb}+{32'b0,row});
                mem_wstrb=(state==LUT_WORD_WRITE)?lut_word_strb:(8'b1<<address_full[2:0]);
                mem_wdata=(state==LUT_WORD_WRITE)?lut_word:
                    {8{(state==LUT_WRITE)?lut_value:((state==Q_WAIT||state==Q_STREAM)?output_result:result_byte)}};
                address_full=address_full&64'hfffffffffffffff8;
            end
            default:begin end
        endcase
        mem_addr=address_full[23:0];
        qvalid=(state==Q_SEND)||(state==Q_STREAM&&q_issue_lane<pw_take)||(SCALAR_LUT&&state==LUT_SEND);
    end
    generate if(RQ_FAST)begin:fast_requant
        v2_requantizer_fast requant(.clk(clk),.rst_n(rst_n),.flush(abort_run||clear_counters),.in_valid(qvalid),.in_ready(qready),.accumulator(acc),.multiplier(mult),.shift(shift),.zero_point(zy),.out_valid(qout),.out_ready(requant_output_ready||(SCALAR_LUT&&state==LUT_WAIT)),.result(qresult));
    end else begin:original_requant
        v2_requantizer requant(.clk(clk),.rst_n(rst_n),.flush(abort_run||clear_counters),.in_valid(qvalid),.in_ready(qready),.accumulator(acc),.multiplier(mult),.shift(shift),.zero_point(zy),.out_valid(qout),.out_ready(requant_output_ready||(SCALAR_LUT&&state==LUT_WAIT)),.result(qresult));
    end endgenerate
    task automatic fail(input logic [7:0] code);
        begin error_code<=code;busy<=0;done<=1;state<=IDLE;pw_prefetch_pending<=0;end
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
    // Advance only when the scratchpad accepts the output byte.
    // The requantizer holds its registered result under backpressure.
    task automatic select_parameters(input logic [15:0] channel);
        begin
            if(PARAM_FAST&&PARAM_CACHE&&parameter_valid[channel[0]]&&parameter_tag[channel[0]]==channel)begin
                {p1,p0}<=parameter_data[channel[0]];state<=P_CHECK;
            end else state<=P_REQ;
        end
    endtask
    // Operand and distributed weight-cache registers are independent.
    // Consume a valid cached word on the operand-completion edge, avoiding
    // the otherwise empty W_REQ/PW_W_REQ hit cycle. Misses use the original
    // stable request states; col already identifies the completed operand.
    task automatic operand_ready;
        begin
            if((spatial_pw||spatial_dw)&&col[2:0]!=0)state<=PW_MAC;
            else if(cached_weight)begin
                whex<=weight_cache[weight_cache_index];
                if(spatial_pw||spatial_dw)pw_weight<=$signed(weight_cache[weight_cache_index][7:0]);
                state<=(spatial_pw||spatial_dw)?PW_MAC:MAC;
            end else state<=(spatial_pw||spatial_dw)?PW_W_REQ:W_REQ;
        end
    endtask
    task automatic advance_output;
        begin
                    if(row+1>=outputs)state<=NEXT_LAYER;
                    else begin
                        row<=row+1;col<=0;pi<=0;
                        if(spatial_pw||spatial_dw)begin
                            if(pw_lane+1<pw_take)begin
                                pw_lane<=pw_lane+3'd1;pixel_index<=pixel_index+1;
                                acc<=pw_acc[pw_lane+3'd1][31:0];state<=Q_SEND;
                            end else if(spatial_dw)begin
                                if(pixel_index+1>=output_area)begin
                                    pixel_index<=0;ox<=0;oy<=0;oc<=oc+16'd1;
                                    output_channel_base<=output_channel_base+output_area;
                                    weight_row<=weight_row+stride[23:0];weight_cache_valid<=0;
                                    channel_origin<=channel_origin+$signed(plane);
                                    row_origin<=channel_origin+$signed(plane);pixel_origin<=channel_origin+$signed(plane);
                                    origin_x<=-$signed({2'b0,pl});origin_y<=-$signed({2'b0,pt});
                                    select_parameters(oc+16'd1);
                                end else begin
                                    pixel_index<=pixel_index+1;
                                    if(ox+{12'b0,pw_take}<ow)begin
                                        ox<=ox+{12'b0,pw_take};origin_x<=origin_x+$signed({14'b0,pw_take});
                                        pixel_origin<=pixel_origin+$signed({28'b0,pw_take});
                                    end else begin
                                        ox<=0;oy<=oy+16'd1;origin_x<=-$signed({2'b0,pl});origin_y<=origin_y+18'sd1;
                                        row_origin<=row_origin+$signed(row_step);pixel_origin<=row_origin+$signed(row_step);
                                    end
                                    select_parameters(oc);
                                end
                            end else if(pixel_index+1<output_area)begin
                                pixel_index<=pixel_index+1;select_parameters(oc);
                            end else begin
                                pixel_index<=0;oc<=oc+16'd1;output_channel_base<=output_channel_base+output_area;
                                weight_row<=weight_row+stride[23:0];weight_cache_valid<=0;select_parameters(oc+16'd1);
                            end
                        end else if(spatial)begin
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
                            select_parameters((op!=OP_CONV&&op!=OP_DWCONV)?16'd0:
                                ((broadcast_mode&&!oc[0]&&oc+16'd1<oc_total)?oc+16'd1:
                                 ((pixel_index+1<output_area)?((broadcast_mode&&oc[0])?oc-16'd1:oc):oc+16'd1)));
                        end else begin
                            weight_row<=weight_row+stride[23:0];
                            state<=(op==OP_GEMM)?P_REQ:((scalar_word_reuse&&row[2:0]!=7)?(scalar_lut?LUT_WRITE:MAC):X_REQ);
                        end
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
            pw_input_addr<=0;pw_take<=0;pw_lane<=0;q_issue_lane<=0;spatial_pw<=0;spatial_dw<=0;dw_mask<=0;dw_skip<=0;dw_take<=0;pw_weight<=0;pw_prefetch_pending<=0;scalar_word_reuse<=0;scalar_lut<=0;vector_lut<=0;lut_index<=0;line_clear_index<=0;
        end else begin
            done<=0;layer_marker<=0;
            if(busy)begin
                elapsed<=elapsed+1;
                if(state==MAC||state==PW_MAC)compute_cycles<=compute_cycles+1;
                else if(state==F_L_REQ||state==F_L_WAIT||state==D_REQ||state==D_WAIT||state==P_REQ||state==P_WAIT||state==X_REQ||state==X_WAIT||(state==W_REQ&&mem_req)||state==W_WAIT||state==WRITE_OUT||state==WIN_REQ||state==WIN_WAIT||state==POOL_REQ||state==POOL_WAIT||state==PW_X_REQ||state==PW_X_WAIT||state==PW_X2_REQ||state==PW_X2_WAIT||(state==PW_W_REQ&&mem_req)||state==PW_W_WAIT)wait_cycles<=wait_cycles+1;
                else control_cycles<=control_cycles+1;
                if(mem_req&&mem_ready)begin
                    if(mem_wr)write_bytes<=write_bytes+1;else read_bytes<=read_bytes+8;
                end
            end
            if(clear_counters)begin
                line_valid<=0;parameter_valid<=0;pw_prefetch_pending<=0;
                state<=IDLE;busy<=0;done<=0;error_code<=0;elapsed<=0;compute_cycles<=0;wait_cycles<=0;control_cycles<=0;useful_macs<=0;read_bytes<=0;write_bytes<=0;layer_count<=0;
            end else if(abort_run)begin state<=IDLE;busy<=0;done<=busy;error_code<=busy?8'd6:8'd0;line_valid<=0;parameter_valid<=0;pw_prefetch_pending<=0;end
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
                    spatial_pw<=spatial_pw_decode;
                    spatial_dw<=SPATIAL_DW&&op==OP_DWCONV&&kh==3&&kw==3&&sh==1&&sw==1;
                    scalar_word_reuse<=SCALAR_WORD_REUSE&&(op==OP_RELU||op==OP_CLIP||op==OP_COPY)&&
                        (xb==yb||{32'b0,xb}+{32'b0,outputs}<={32'b0,yb}||
                         {32'b0,yb}+{32'b0,outputs}<={32'b0,xb});
                    // Build only when two removed per-byte states amortize 768 setup cycles.
                    scalar_lut<=SCALAR_LUT&&(op==OP_RELU||op==OP_CLIP)&&outputs>=512;
                    vector_lut<=VECTOR_LUT&&SCALAR_LUT&&(op==OP_RELU||op==OP_CLIP)&&outputs>=512&&
                        yb[2:0]==0&&(xb==yb||{32'b0,xb}+{32'b0,outputs}<={32'b0,yb}||
                                      {32'b0,yb}+{32'b0,outputs}<={32'b0,xb});
                    if(d[0][31:0]!=32'h32445355||d[0][39:32]!=8'(DESC_VERSION)||!fusion_abi_valid)fail(8'd2);
                    else if(op==OP_HALT)begin busy<=0;done<=1;state<=IDLE;end
                    else if(op!=OP_GEMM&&op!=OP_RELU&&op!=OP_COPY&&op!=OP_CONV&&op!=OP_MAXPOOL&&op!=OP_DWCONV&&op!=OP_AVGPOOL&&op!=OP_CLIP)fail(8'd3);
                    else if(count==0||outputs==0||count>MEM_BYTES||outputs>MEM_BYTES||xb[2:0]!=0||{32'b0,yb}+{32'b0,outputs}>64'(MEM_BYTES)||next_pc[5:0]!=0||next_pc>MEM_BYTES-64)fail(8'd1);
                    else if(fused_activation)begin lut_index<=0;state<=F_L_REQ;end
                    else state<=spatial?GEOM0:OTHER_CHECK;
                end
                OTHER_CHECK:begin
                    if(d[5]!=64'h0001000100010001||d[6]!=0||d[7]!=64'h0001000100010001||
                        {32'b0,xb}+(({32'b0,count}+7)&64'hfffffffffffffff8)>64'(MEM_BYTES))fail(8'd1);
                    else if(op==OP_GEMM&&(wb[2:0]!=0||pb[2:0]!=0||stride[2:0]!=0||stride<count||stride>MEM_BYTES||
                        {32'b0,wb}+{32'b0,gemm_weight_span}>64'(MEM_BYTES)||
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
                    plane<=plane_calc;row_step<=row_step_calc;pad_offset<=pad_offset_calc;
                    kernel_area<=kernel_area_calc;output_area<=output_area_calc;
                    oh<=oh_calc[15:0];ow<=ow_calc[15:0];
                    state<=GEOM_COUNT;
                end
                GEOM_COUNT:begin
                    input_bytes_reg<=input_bytes_calc;
                    reduction_reg<=(op==OP_DWCONV)?kernel_area:reduction_calc;
                    outputs_reg<=outputs_calc;
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
                        {32'b0,wb}+{37'b0,spatial_weight_span}>64'(MEM_BYTES)||
                        {32'b0,pb}+({48'b0,oc_total}<<4)>64'(MEM_BYTES)))fail(8'd1);
                    else if((op==OP_MAXPOOL||op==OP_AVGPOOL)&&(pb[2:0]!=0||{32'b0,pb}+16>64'(MEM_BYTES)))fail(8'd1);
                    else if(layer_count>=MAX_DESCRIPTORS)fail(8'd7);
                    else begin
                        row<=0;col<=0;weight_row<=wb[23:0];pi<=0;weight_cache_valid<=0;line_valid<=0;parameter_valid<=0;
                        line_clear_index<=0;state<=VALID_IN_BRAM?LINE_CLEAR:GEOM1;
                    end
                end
                LINE_CLEAR:begin
                    if(line_clear_index==LINE_BITS'(LINE_ENTRIES-1))state<=GEOM1;
                    else line_clear_index<=line_clear_index+1'b1;
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
                        if(spatial_pw||spatial_dw)begin
                            for(integer k=0;k<8;k=k+1)pw_acc[k]<={{(ACC_WORK_BITS-32){p0[31]}},p0[31:0]};
                            pw_take<=spatial_dw?((ow-ox>=8)?4'd8:4'(ow-ox)):
                                ((output_area-pixel_index>=8)?4'd8:4'(output_area-pixel_index));
                            pw_lane<=0;pw_input_addr<=xb[23:0]+pixel_index[23:0];state<=spatial_dw?DW_PREP:PW_X_REQ;
                        end else if(scalar_lut)begin lut_index<=0;state<=LUT_BUILD;end
                        else state<=(op==OP_CONV||op==OP_DWCONV)?WIN_PREP:((op==OP_MAXPOOL||op==OP_AVGPOOL)?POOL_PREP:X_REQ);
                    end
                end
                F_L_REQ:if(mem_ready)state<=F_L_WAIT;
                F_L_WAIT:if(mem_rvalid)begin xword<=mem_rdata;state<=F_L_BYTES;end
                F_L_BYTES:begin
                    if(lut_index==255)state<=GEOM0;
                    else begin
                        lut_index<=lut_index+1;
                        if(lut_index[2:0]==7)state<=F_L_REQ;
                    end
                end
                X_REQ:if(mem_ready)state<=X_WAIT;
                X_WAIT:if(mem_rvalid)begin xword<=mem_rdata;if(op==OP_GEMM)state<=W_REQ;else state<=scalar_lut?(vector_lut?LUT_WORD_WRITE:LUT_WRITE):MAC;end
                LUT_BUILD:begin acc<=scalar_acc;state<=LUT_SEND;end
                LUT_SEND:if(qready)state<=LUT_WAIT;
                LUT_WAIT:if(qout)begin
                    if(lut_index==255)state<=X_REQ;
                    else begin lut_index<=lut_index+8'd1;state<=LUT_BUILD;end
                end
                LUT_WRITE:if(mem_ready)advance_output();
                LUT_WORD_WRITE:if(mem_ready)begin
                    if(row+32'd8>=outputs)state<=NEXT_LAYER;
                    else begin row<=row+32'd8;state<=X_REQ;end
                end
                DW_PREP:begin
                    dw_mask<=dw_mask_calc;dw_skip<=dw_skip_calc;dw_take<=dw_take_calc;
                    pw_input_addr<=win_addr[23:0]+{20'b0,dw_skip_calc};
                    if(col[2:0]!=0)pw_weight<=$signed(whex[col[2:0]*8+:8]);
                    if(dw_take_calc==0)begin xword<=0;operand_ready();end
                    else state<=PW_X_REQ;
                end
                PW_X_REQ:begin
                    if(pw_prefetch_pending)begin
                        if(mem_ready)begin pw_prefetch_pending<=0;state<=PW_X_WAIT;end
                    end else if(pw_cache_hit)begin
                        xword<=(line_read_data>>(32'(pw_input_addr[2:0])*8))<<
                            ((spatial_dw?32'(dw_skip):32'd0)*8);
                        if(col[2:0]!=0)pw_weight<=$signed(whex[col[2:0]*8+:8]);
                        if(32'(pw_input_addr[2:0])+
                            (spatial_dw?32'(dw_take):32'(pw_take))>8)state<=PW_X2_REQ;
                        else operand_ready();
                    end else if(mem_ready)state<=PW_X_WAIT;
                end
                PW_X_WAIT:if(mem_rvalid)begin
                    if(PW_WORD_CACHE)line_valid[pw_cache_index]<=1;
                    xword<=(mem_rdata>>(32'(pw_input_addr[2:0])*8))<<((spatial_dw?32'(dw_skip):32'd0)*8);
                    if(col[2:0]!=0)pw_weight<=$signed(whex[col[2:0]*8+:8]);
                    if(32'(pw_input_addr[2:0])+(spatial_dw?32'(dw_take):32'(pw_take))>8)state<=PW_X2_REQ;
                    else operand_ready();
                end
                PW_X2_REQ:if(mem_ready)state<=PW_X2_WAIT;
                PW_X2_WAIT:if(mem_rvalid)begin
                    if(PW_WORD_CACHE)line_valid[pw_cache_index]<=1;
                    xword<=xword|(mem_rdata<<((32'd8-32'(pw_input_addr[2:0])+(spatial_dw?32'(dw_skip):32'd0))*8));
                    operand_ready();
                end
                PW_W_REQ:begin
                    if(cached_weight)begin whex<=weight_cache[weight_cache_index];pw_weight<=$signed(weight_cache[weight_cache_index][7:0]);state<=PW_MAC;end
                    else if(mem_ready)state<=PW_W_WAIT;
                end
                PW_W_WAIT:if(mem_rvalid)begin
                    whex<=mem_rdata;pw_weight<=$signed(mem_rdata[7:0]);weight_cache[weight_cache_index]<=mem_rdata;
                    weight_cache_valid[weight_cache_index]<=1;state<=PW_MAC;
                end
                PW_MAC:begin
                    if((col[2:0]==7||col+1>=count)&&pw_overflow)fail(8'd5);
                    else begin
                        pw_prefetch_pending<=PW_FETCH_AHEAD&&!spatial_dw&&col+1<count&&
                            !pw_next_cache_hit&&!mem_ready;
                        for(integer k=0;k<8;k=k+1)if(k<pw_take)pw_acc[k]<=pw_next[k];
                        useful_macs<=useful_macs+{28'b0,pw_take};
                        if(col+1>=count)begin acc<=pw_next[0][31:0];state<=Q_SEND;end
                        else if(spatial_dw)begin
                            col<=col+1;advance_window(4'd1);state<=DW_PREP;
                        end else begin
                            col<=col+1;pw_input_addr<=pw_next_input_addr;
                            if(PW_FETCH_AHEAD&&pw_next_cache_hit&&pw_next_crosses)begin
                                xword<=line_read_data>>(32'(pw_next_input_addr[2:0])*8);
                                if(col[2:0]!=7)pw_weight<=$signed(whex[(col[2:0]+3'd1)*8+:8]);
                                state<=mem_ready?PW_X2_WAIT:PW_X2_REQ;
                            end else if(PW_HIT_BYPASS&&PW_FETCH_AHEAD&&pw_next_cache_hit)begin
                                xword<=line_read_data>>(32'(pw_next_input_addr[2:0])*8);
                                if(col[2:0]!=7)pw_weight<=$signed(whex[(col[2:0]+3'd1)*8+:8]);
                                state<=col[2:0]==7?PW_W_REQ:PW_MAC;
                            end else state<=(PW_FETCH_AHEAD&&!pw_next_cache_hit&&mem_ready)?PW_X_WAIT:PW_X_REQ;
                        end
                    end
                end
                WIN_PREP:begin
                    if(broadcast_mode&&oc[0])begin
                        xword<=activation_tile[col[6:3]];
                        operand_ready();
                    end else if(col+{29'b0,win_lane}>=count)begin
                        for(integer k=0;k<8;k=k+1)
                            if(k>=win_lane)xword[k*8+:8]<=8'b0;
                        operand_ready();
                    end else if(!window_inside)begin
                        xword[win_lane*8+:8]<=zx;
                        advance_window(4'd1);
                        if(win_lane==7||col+{29'b0,win_lane}+1>=count)operand_ready();
                        else win_lane<=win_lane+3'd1;
                    end else if(line_hit)begin
                        for(integer k=0;k<8;k=k+1)
                            if(k<win_take)
                                xword[(32'(win_lane)+k)*8+:8]<=line_read_data[(32'(win_addr[2:0])+k)*8+:8];
                        advance_window(win_take);
                        if(win_lane+win_take>=8||col+{29'b0,win_lane}+{28'b0,win_take}>=count)operand_ready();
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
                    if(win_lane+win_take>=8||col+{29'b0,win_lane}+{28'b0,win_take}>=count)operand_ready();
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
                        acc<=(op==OP_AVGPOOL)?{{13{pool_sum[18]}},pool_sum}:({{24{pool_max[7]}},pool_max}-{{24{zx[7]}},zx});
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
                Q_SEND:if(qready)begin
                    if(spatial_pw||spatial_dw)begin
                        // Issue lane zero from acc, then keep one operand
                        // registered ahead of the accepted output write.
                        q_issue_lane<=1;
                        if(pw_take>1)acc<=pw_acc[1][31:0];
                        state<=Q_STREAM;
                    end else state<=Q_WAIT;
                end
                Q_STREAM:begin
                    // pw_lane/pixel_index identify the result being retired;
                    // q_issue_lane identifies the next accumulator to issue.
                    // Both freeze under output backpressure. The requantizer
                    // is a one-stage elastic register and never drops data.
                    if(output_valid&&mem_ready)begin
                        advance_output();
                        if(pw_lane+1<pw_take)state<=Q_STREAM;
                    end
                    if(qvalid&&qready)begin
                        q_issue_lane<=q_issue_lane+1;
                        if(q_issue_lane+1<pw_take)
                            acc<=pw_acc[q_issue_lane[2:0]+3'd1][31:0];
                    end
                end
                Q_WAIT:if(output_valid&&mem_ready)advance_output();
                WRITE_OUT:if(mem_ready)advance_output();
                NEXT_LAYER:begin pc<=next_pc[23:0];di<=0;layer_count<=layer_count+1;layer_marker<=1;state<=D_REQ;end
                default:fail(8'd8);
            endcase
        end
    end
endmodule

// Numerically identical to v2_requantizer for every accepted INT32 input,
// signed INT32 multiplier, shift 0..62 and signed INT8 zero point. Arithmetic
// right shift supplies the floor quotient; guard/sticky bits implement the
// original ties-away-from-zero rounding without 64-bit abs and negate paths.
module v2_requantizer_fast (
    input logic clk, rst_n, flush,
    input logic in_valid, output logic in_ready,
    input logic signed [31:0] accumulator, multiplier,
    input logic [5:0] shift,
    input logic signed [7:0] zero_point,
    output logic out_valid, input logic out_ready,
    output logic signed [7:0] result
);
    logic signed [63:0] product, quotient;
    logic [63:0] sticky_mask;
    logic guard, sticky, round_up;
    logic signed [9:0] rounded_small, scaled_small;
    logic too_positive, too_negative;
    always_comb begin
        product=accumulator*multiplier;
        quotient=product>>>shift;
        sticky_mask=(shift>1)?((64'd1<<(shift-1))-64'd1):64'd0;
        guard=(shift!=0)&&product[shift==0?0:shift-1];
        sticky=(product&sticky_mask)!=0;
        round_up=guard&&(!product[63]||sticky);
        too_positive=!quotient[63]&&(|quotient[63:8]);
        too_negative=quotient[63]&&!( &quotient[63:8]);
        rounded_small=$signed({quotient[8],quotient[8:0]})+$signed({9'b0,round_up});
        scaled_small=rounded_small+$signed({{2{zero_point[7]}},zero_point});
    end
    assign in_ready=!out_valid||out_ready;
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin out_valid<=0;result<=0;end
        else if(flush)begin out_valid<=0;result<=0;end
        else if(in_ready)begin
            out_valid<=in_valid;
            if(in_valid)begin
                if(too_positive||(!too_negative&&scaled_small>10'sd127))result<=8'sd127;
                else if(too_negative||scaled_small< -10'sd128)result<= -8'sd128;
                else result<=scaled_small[7:0];
            end
        end
    end
endmodule
