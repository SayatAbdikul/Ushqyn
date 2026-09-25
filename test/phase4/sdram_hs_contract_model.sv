// Test-only model of the HS user's command/data timing, not a vendor PHY model.
// It checks the adapter's cache/write masks, line boundaries and refresh policy.
module SDRAM_Controller_HS_Top (
    output wire O_sdram_clk, O_sdram_cke, O_sdram_cs_n, O_sdram_cas_n,
    output wire O_sdram_ras_n, O_sdram_wen_n,
    output wire [3:0] O_sdram_dqm,
    output wire [10:0] O_sdram_addr,
    output wire [1:0] O_sdram_ba,
    inout wire [31:0] IO_sdram_dq,
    input wire I_sdrc_rst_n,I_sdrc_clk,I_sdram_clk,I_sdrc_cmd_en,
    input wire [2:0] I_sdrc_cmd,
    input wire I_sdrc_precharge_ctrl,I_sdram_power_down,I_sdram_selfrefresh,
    input wire [20:0] I_sdrc_addr,
    input wire [3:0] I_sdrc_dqm,
    input wire [31:0] I_sdrc_data,
    input wire [7:0] I_sdrc_data_len,
    output logic [31:0] O_sdrc_data,
    output logic O_sdrc_init_done,O_sdrc_cmd_ack
);
    logic [31:0] memory[0:2097151];
    integer remaining,index,latency;
    logic reading,writing;
    logic [20:0] base;
    integer refresh_count;
    always_ff @(posedge I_sdrc_clk) begin
        O_sdrc_cmd_ack<=0;
        if (!I_sdrc_rst_n) begin
            O_sdrc_init_done<=0;reading<=0;writing<=0;remaining<=0;
            index<=0;latency<=0;base<=0;O_sdrc_data<=0;refresh_count<=0;
        end else begin
            O_sdrc_init_done<=1;
            if (I_sdrc_cmd_en) begin
                case (I_sdrc_cmd)
                    3'b011: O_sdrc_cmd_ack<=1;
                    3'b001: begin O_sdrc_cmd_ack<=1;refresh_count<=refresh_count+1;end
                    3'b101: begin
                        assert(I_sdrc_addr[3:0]==0 && I_sdrc_data_len==15);
                        reading<=1;latency<=2;index<=0;base<=I_sdrc_addr;
                        remaining<=16;
                    end
                    3'b100: begin
                        assert(I_sdrc_addr[3:0]==0 && I_sdrc_data_len==15);
                        for(integer k=0;k<4;k=k+1)
                            if(!I_sdrc_dqm[k])memory[I_sdrc_addr][k*8+:8]<=I_sdrc_data[k*8+:8];
                        writing<=1;index<=1;base<=I_sdrc_addr;remaining<=15;
                    end
                    default: ;
                endcase
            end else if (reading) begin
                if (latency!=0) latency<=latency-1;
                else begin
                    O_sdrc_data<=memory[base+21'(index)];index<=index+1;remaining<=remaining-1;
                    if (remaining==1) begin reading<=0;O_sdrc_cmd_ack<=1;end
                end
            end else if (writing) begin
                for(integer k=0;k<4;k=k+1)
                    if(!I_sdrc_dqm[k])memory[base+21'(index)][k*8+:8]<=I_sdrc_data[k*8+:8];
                index<=index+1;remaining<=remaining-1;
                if (remaining==1) begin writing<=0;O_sdrc_cmd_ack<=1;end
            end
        end
    end
endmodule
