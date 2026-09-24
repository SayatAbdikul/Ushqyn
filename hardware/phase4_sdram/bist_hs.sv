// HS-IP variant of the full-address-range SDRAM board test. 64 complete write/read sweeps transfer
// 1 GiB in aggregate. The first sweep is sequential, then an invertible seeded
// xorshift permutation visits every byte in a pseudorandom order. Every fourth
// sweep changes which address bits determine the byte pattern, so aliasing in
// any address group cannot hide consistently.
module phase4_hs_sdram_bist (
    input logic sys_clk, reset_button,
    output logic uart_tx_pin,
    output logic [5:0] led,
    output logic O_sdram_clk, O_sdram_cke, O_sdram_cs_n,
    output logic O_sdram_cas_n, O_sdram_ras_n, O_sdram_wen_n,
    output logic [3:0] O_sdram_dqm,
    output logic [10:0] O_sdram_addr,
    output logic [1:0] O_sdram_ba,
    inout wire [31:0] IO_sdram_dq
);
    localparam logic [22:0] LAST_ADDR = 23'h7fffff;
    localparam logic [6:0] LAST_PASS = 7'd64;
    localparam logic [21:0] HOLD_CYCLES = 22'd2160000; // 80 ms at 27 MHz
    logic clk, clk_sdram, pll_lock;
    phase4_sdram_pll pll(.clkin(sys_clk), .clk(clk),
                         .clk_sdram(clk_sdram), .lock(pll_lock));
    logic [2:0] reset_sync = 3'b000;
    always_ff @(posedge clk or posedge reset_button) begin
        if (reset_button) reset_sync <= 0;
        else reset_sync <= {reset_sync[1:0], 1'b1};
    end
    wire rst_n = pll_lock && reset_sync[2];

    logic rd, wr, refresh, mem_busy, data_ready;
    logic [22:0] addr;
    logic [7:0] din, dout;
    phase4_hs_byte_adapter memory (
        .clk(clk), .clk_sdram(clk_sdram), .resetn(rst_n),
        .rd(rd), .wr(wr), .refresh(refresh), .addr(addr),
        .din(din), .dout(dout), .dout32(),
        .data_ready(data_ready), .busy(mem_busy),
        .SDRAM_DQ(IO_sdram_dq), .SDRAM_A(O_sdram_addr),
        .SDRAM_BA(O_sdram_ba), .SDRAM_nCS(O_sdram_cs_n),
        .SDRAM_nWE(O_sdram_wen_n), .SDRAM_nRAS(O_sdram_ras_n),
        .SDRAM_nCAS(O_sdram_cas_n), .SDRAM_CLK(O_sdram_clk),
        .SDRAM_CKE(O_sdram_cke), .SDRAM_DQM(O_sdram_dqm)
    );

    typedef enum logic [3:0] {
        INIT, WRITE_ISSUE, WRITE_BUSY, WRITE_DONE,
        HOLD, READ_ISSUE, READ_BUSY, READ_DATA, READ_DONE,
        REF_BUSY, REF_DONE, SEND, SEND_PULSE, SEND_WAIT, FINISHED
    } state_t;
    state_t state, return_state, after_send;
    logic controller_ready;
    logic [22:0] byte_addr;
    logic [22:0] op_index;
    logic [6:0] pass_count;
    logic [21:0] hold_count;
    logic [31:0] bytes_moved;
    logic fail, done;
    logic [22:0] failure_addr;
    logic [7:0] failure_expected, failure_actual;
    logic [1:0] report_kind; // 0=pass, 1=success, 2=failure
    logic [3:0] report_index;

    function automatic logic [7:0] pattern(input logic [22:0] a,
                                            input logic [6:0] p);
        case (p[1:0])
            2'd0: pattern = a[7:0] ^ {1'b0,p};
            2'd1: pattern = a[15:8] ^ {1'b0,p};
            2'd2: pattern = {1'b0,a[22:16]} ^ {1'b0,p};
            default: pattern = a[7:0] ^ {a[14:8],a[15]} ^
                               {1'b0,a[20:16],a[22:21]} ^ {1'b0,p} ^ 8'ha5;
        endcase
    endfunction

    function automatic logic [22:0] physical_address(input logic [22:0] i,
                                                      input logic [6:0] p);
        logic [22:0] x;
        begin
            if (p == 0) physical_address = i;
            else begin
                x = i ^ {p,16'h5a37};
                x = x ^ (x << 7);
                x = x ^ (x >> 9);
                x = x ^ (x << 8);
                physical_address = x;
            end
        end
    endfunction

    logic refresh_req, refresh_ack, refresh_missed;
    logic [12:0] pending_refreshes;
    wire issue_state = state == WRITE_ISSUE || state == READ_ISSUE ||
                       state == HOLD || state == FINISHED;
    v2_sdram_refresh scheduler(
        .clk(clk), .rst_n(rst_n), .init_done(controller_ready),
        .controller_idle(issue_state && !mem_busy),
        .refresh_ack(refresh_ack), .refresh_req(refresh_req),
        .pending_refreshes(pending_refreshes),
        .refresh_deadline_missed(refresh_missed)
    );

    logic [7:0] tx_data;
    logic tx_valid, tx_ready;
    uart_tx #(.CLK_FREQ(27000000), .BAUD_RATE(115200)) tx(
        .clk(clk), .rst_n(rst_n), .tx_data(tx_data), .tx_valid(tx_valid),
        .tx_ready(tx_ready), .tx_o(uart_tx_pin)
    );
    always_comb begin
        tx_data = 8'h00;
        case (report_kind)
            2'd0: tx_data = report_index == 0 ? 8'hd4 : {1'b0,pass_count};
            2'd1: case (report_index)
                0: tx_data = 8'ha4;
                1: tx_data = {1'b0,pass_count};
                2: tx_data = bytes_moved[7:0];
                3: tx_data = bytes_moved[15:8];
                4: tx_data = bytes_moved[23:16];
                default: tx_data = bytes_moved[31:24];
            endcase
            default: case (report_index)
                0: tx_data = 8'he4;
                1: tx_data = {1'b0,pass_count};
                2: tx_data = failure_addr[7:0];
                3: tx_data = failure_addr[15:8];
                4: tx_data = {1'b0,failure_addr[22:16]};
                5: tx_data = failure_expected;
                default: tx_data = failure_actual;
            endcase
        endcase
    end
    wire report_last = report_index == (report_kind == 2'd0 ? 4'd1 :
                                         report_kind == 2'd1 ? 4'd5 : 4'd6);
    assign led = ~{fail | refresh_missed, done, pass_count[2:0], !done};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= INIT;
            return_state <= WRITE_ISSUE;
            after_send <= WRITE_ISSUE;
            controller_ready <= 0;
            rd <= 0; wr <= 0; refresh <= 0; refresh_ack <= 0;
            addr <= 0; din <= 0; byte_addr <= 0; op_index <= 0;
            pass_count <= 0; hold_count <= 0; bytes_moved <= 0;
            fail <= 0; done <= 0;
            failure_addr <= 0; failure_expected <= 0; failure_actual <= 0;
            report_kind <= 0; report_index <= 0; tx_valid <= 0;
        end else begin
            rd <= 0; wr <= 0; refresh <= 0; refresh_ack <= 0;
            tx_valid <= 0;
            if (refresh_missed && !fail) begin
                fail <= 1;
                failure_addr <= byte_addr;
                failure_expected <= 8'hff;
                failure_actual <= 8'h00;
                report_kind <= 2;
                report_index <= 0;
                after_send <= FINISHED;
                state <= SEND;
            end else case (state)
                INIT: if (!mem_busy) begin
                    controller_ready <= 1;
                    state <= WRITE_ISSUE;
                end
                WRITE_ISSUE: if (!mem_busy) begin
                    if (refresh_req) begin
                        refresh <= 1; refresh_ack <= 1;
                        return_state <= WRITE_ISSUE; state <= REF_BUSY;
                    end else begin
                        byte_addr <= physical_address(op_index, pass_count);
                        addr <= physical_address(op_index, pass_count);
                        din <= pattern(physical_address(op_index, pass_count), pass_count);
                        wr <= 1;
                        state <= WRITE_BUSY;
                    end
                end
                WRITE_BUSY: if (mem_busy) state <= WRITE_DONE;
                WRITE_DONE: if (!mem_busy) begin
                    bytes_moved <= bytes_moved + 1;
                    if (op_index == LAST_ADDR) begin
                        op_index <= 0; hold_count <= 0; state <= HOLD;
                    end else begin
                        op_index <= op_index + 1'b1; state <= WRITE_ISSUE;
                    end
                end
                HOLD: if (!mem_busy) begin
                    if (refresh_req) begin
                        refresh <= 1; refresh_ack <= 1;
                        return_state <= HOLD; state <= REF_BUSY;
                    end else if (hold_count == HOLD_CYCLES) state <= READ_ISSUE;
                    else hold_count <= hold_count + 1;
                end
                READ_ISSUE: if (!mem_busy) begin
                    if (refresh_req) begin
                        refresh <= 1; refresh_ack <= 1;
                        return_state <= READ_ISSUE; state <= REF_BUSY;
                    end else begin
                        byte_addr <= physical_address(op_index, pass_count);
                        addr <= physical_address(op_index, pass_count);
                        rd <= 1; state <= READ_BUSY;
                    end
                end
                READ_BUSY: if (mem_busy) state <= READ_DATA;
                READ_DATA: if (data_ready) begin
                    if (dout != pattern(byte_addr, pass_count)) begin
                        fail <= 1;
                        failure_addr <= byte_addr;
                        failure_expected <= pattern(byte_addr, pass_count);
                        failure_actual <= dout;
                        report_kind <= 2; report_index <= 0;
                        after_send <= FINISHED; state <= SEND;
                    end else state <= READ_DONE;
                end
                READ_DONE: if (!mem_busy) begin
                    bytes_moved <= bytes_moved + 1;
                    if (op_index == LAST_ADDR) begin
                        op_index <= 0;
                        pass_count <= pass_count + 1;
                        report_kind <= pass_count == LAST_PASS - 1 ? 2'd1 : 2'd0;
                        report_index <= 0;
                        after_send <= pass_count == LAST_PASS - 1 ? FINISHED : WRITE_ISSUE;
                        if (pass_count == LAST_PASS - 1) done <= 1;
                        state <= SEND;
                    end else begin
                        op_index <= op_index + 1'b1; state <= READ_ISSUE;
                    end
                end
                REF_BUSY: if (mem_busy) state <= REF_DONE;
                REF_DONE: if (!mem_busy) state <= return_state;
                SEND: if (tx_ready) begin tx_valid <= 1; state <= SEND_PULSE; end
                SEND_PULSE: state <= SEND_WAIT;
                SEND_WAIT: if (tx_ready) begin
                    if (report_last) state <= after_send;
                    else begin report_index <= report_index + 1; state <= SEND; end
                end
                FINISHED: if (!mem_busy && refresh_req) begin
                    refresh <= 1; refresh_ack <= 1;
                    return_state <= FINISHED; state <= REF_BUSY;
                end
                default: state <= INIT;
            endcase
        end
    end
endmodule
