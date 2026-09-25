// Stop-and-wait framed host interface. CRC is checked before any mutation.
// A5 5A, version, command, sequence, address LE24, length LE16, [WRITE data], CRC LE16.
// Responses use command|80 and payload {status, data...}; length includes status.
module v2_command #(
    parameter integer TIMEOUT_CYCLES=target_pkg::RX_TIMEOUT,
    parameter bit TILED_MEM_MAP=0
)(
    input logic clk,rst_n,
    input logic rx_valid, input logic [7:0] rx_data,
    output logic tx_valid, output logic [7:0] tx_data, input logic tx_ready,
    output logic start, abort_run, clear_counters, output logic [23:0] start_pc,
    input logic busy, input logic [7:0] core_error,
    input logic mmio_while_busy,
    input logic [31:0] elapsed,compute_cycles,wait_cycles,control_cycles,useful_macs,read_bytes,write_bytes,layer_count,
    output logic mem_req,mem_wr,output logic [23:0] mem_addr,
    output logic [63:0] mem_wdata,output logic [7:0] mem_wstrb,
    input logic mem_ready,mem_rvalid,input logic [63:0] mem_rdata,
    output logic [31:0] protocol_errors
);
    import target_pkg::*;
    localparam [7:0] CAPS=1,READ=2,WRITE=3,RUN=4,STATUS=5,ABORT=6,RESET=7;
    typedef enum logic [3:0] {SYNC0,SYNC1,HEADER,PAYLOAD,CRC0,CRC1,DISPATCH,MREQ,MWAIT,RESPONSE,TX_GAP,DRAIN} state_t;
    state_t state;
    logic [7:0] header[0:7];
    logic [7:0] payload[0:MAX_TRANSFER];
    logic [6:0] index,transfer_index;
    logic [15:0] crc,rx_crc,tx_crc;
    logic [31:0] timeout_count;
    logic [16:0] drain_remaining;
    logic [7:0] response_status;
    logic [6:0] response_length,tx_index;
    wire [7:0] cmd=header[1];
    wire [15:0] length={header[7],header[6]};
    wire [23:0] address={header[5],header[4],header[3]};
    wire [24:0] transfer_end={1'b0,address}+{9'b0,length};
    wire tiled_transfer_valid=
        (address<24'h008000 && transfer_end<=25'h008000) ||
        (address>=24'h400000 && address<24'h400020 &&
         transfer_end<=25'h400020) ||
        (address>=24'h410000 && address<24'h410020 &&
         transfer_end<=25'h410020) ||
        (address>=24'h500000 && address<24'h508000 &&
         transfer_end<=25'h508000) ||
        (address>=24'h800000 && transfer_end<=25'h1000000);
    wire tiled_busy_mmio_write = TILED_MEM_MAP && mmio_while_busy &&
        cmd==WRITE && address>=24'h400000 && address<24'h400020 &&
        transfer_end<=25'h400020;
    logic [24:0] transfer_addr;
    logic [7:0] send_byte;
    integer i;
    function automatic [15:0] crc_byte(input [15:0] c,input [7:0] b);
        reg [15:0] t;
        begin t=c^{b,8'b0};for(integer k=0;k<8;k=k+1)t=t[15]?(t<<1)^16'h1021:(t<<1);crc_byte=t;end
    endfunction
    task automatic respond(input logic [7:0] status,input logic [6:0] size);
        begin response_status<=status;response_length<=size;tx_index<=0;tx_crc<=16'hffff;state<=RESPONSE;end
    endtask
    always_comb begin
        transfer_addr={1'b0,address}+{18'b0,transfer_index};
        mem_req=state==MREQ;mem_wr=cmd==WRITE;mem_addr={transfer_addr[23:3],3'b0};
        mem_wstrb=8'b1<<transfer_addr[2:0];mem_wdata={8{payload[transfer_index]}};
        start_pc=address;
        send_byte=0;
        case(tx_index)
            0:send_byte=8'ha5;
            1:send_byte=8'h5a;
            2:send_byte=8'(PROTOCOL_VERSION);
            3:send_byte=cmd|8'h80;
            4:send_byte=header[2];
            5:send_byte=header[3];
            6:send_byte=header[4];
            7:send_byte=header[5];
            8:send_byte={1'b0,response_length};
            9:send_byte=0;
            10:send_byte=response_status;
            default:begin
                if(tx_index<10+response_length)send_byte=payload[tx_index-11];
                else if(tx_index==10+response_length)send_byte=tx_crc[7:0];
                else send_byte=tx_crc[15:8];
            end
        endcase
        tx_valid=state==RESPONSE;tx_data=send_byte;
    end
    always_ff @(posedge clk or negedge rst_n)begin
        if(!rst_n)begin
            state<=SYNC0;index<=0;transfer_index<=0;crc<=16'hffff;rx_crc<=0;tx_crc<=16'hffff;timeout_count<=0;drain_remaining<=0;protocol_errors<=0;
            start<=0;abort_run<=0;clear_counters<=0;response_status<=0;response_length<=0;tx_index<=0;
            for(i=0;i<8;i=i+1)header[i]<=0;
            // Payload registers intentionally not reset; read only after fill.
        end else begin
            start<=0;abort_run<=0;clear_counters<=0;
            if(state==SYNC1||state==HEADER||state==PAYLOAD||state==CRC0||state==CRC1||state==DRAIN)begin
                if(rx_valid)timeout_count<=0;else timeout_count<=timeout_count+1;
            end else timeout_count<=0;
            if(timeout_count>=TIMEOUT_CYCLES-1)begin state<=SYNC0;timeout_count<=0;protocol_errors<=protocol_errors+1;end
            else case(state)
                SYNC0:if(rx_valid&&rx_data==8'ha5)state<=SYNC1;
                SYNC1:if(rx_valid)begin if(rx_data==8'h5a)begin state<=HEADER;index<=0;crc<=16'hffff;end else if(rx_data!=8'ha5)state<=SYNC0;end
                HEADER:if(rx_valid)begin
                    header[index[2:0]]<=rx_data;crc<=crc_byte(crc,rx_data);
                    if(index==7)begin
                        index<=0;
                        if(cmd==WRITE&&{rx_data,header[6]}>16'(MAX_TRANSFER))begin
                            drain_remaining<={1'b0,rx_data,header[6]}+17'd2;state<=DRAIN;
                        end else if(cmd==WRITE&&{rx_data,header[6]}!=0)state<=PAYLOAD;
                        else state<=CRC0;
                    end
                    else index<=index+7'd1;
                end
                PAYLOAD:if(rx_valid)begin payload[index]<=rx_data;crc<=crc_byte(crc,rx_data);if(16'(index)+16'd1==length)state<=CRC0;else index<=index+7'd1;end
                DRAIN:if(rx_valid)begin
                    if(drain_remaining==1)begin protocol_errors<=protocol_errors+1;respond(8'd4,1);end
                    else drain_remaining<=drain_remaining-17'd1;
                end
                CRC0:if(rx_valid)begin rx_crc[7:0]<=rx_data;state<=CRC1;end
                CRC1:if(rx_valid)begin if({rx_data,rx_crc[7:0]}!=crc)begin protocol_errors<=protocol_errors+1;respond(8'd1,1);end else state<=DISPATCH;end
                DISPATCH:begin
                    if(header[0]!=8'(PROTOCOL_VERSION))respond(8'd2,1);
                    else if(cmd==CAPS&&length==0)begin
                        payload[0]<=8'(NUMERICS);payload[1]<=8'(DESC_VERSION);payload[2]<=8'(LANES);payload[3]<=8'(MAX_TRANSFER);
                        payload[4]<=MEM_BYTES[7:0];payload[5]<=MEM_BYTES[15:8];payload[6]<=MEM_BYTES[23:16];payload[7]<=8'(ADDR_BITS);
                        payload[8]<=TARGET_ID[7:0];payload[9]<=TARGET_ID[15:8];
                        if(TILED_MEM_MAP)begin
                            payload[10]<=8'd1; // external-memory and DMA register capability
                            payload[11]<=8'h00;payload[12]<=8'h00;
                            payload[13]<=8'h80; // external capacity, 0x800000 bytes
                            respond(0,15);
                        end else respond(0,11);
                    end else if(cmd==STATUS&&length==0)begin
                        payload[0]<={7'b0,busy};payload[1]<=core_error;
                        for(i=0;i<4;i=i+1)begin
                            payload[2+i]<=elapsed[i*8+:8];payload[6+i]<=compute_cycles[i*8+:8];payload[10+i]<=wait_cycles[i*8+:8];payload[14+i]<=control_cycles[i*8+:8];
                            payload[18+i]<=useful_macs[i*8+:8];payload[22+i]<=read_bytes[i*8+:8];payload[26+i]<=write_bytes[i*8+:8];payload[30+i]<=layer_count[i*8+:8];payload[34+i]<=protocol_errors[i*8+:8];
                        end
                        respond(0,39);
                    end else if((cmd==ABORT||cmd==RESET)&&length==0)begin abort_run<=1;if(cmd==RESET)begin protocol_errors<=0;clear_counters<=1;end respond(0,1);end
                    else if(cmd==RUN&&length==0)begin
                        if(busy)respond(8'd3,1);
                        else if(address[5:0]!=0||{8'b0,address}>MEM_BYTES-64)respond(8'd4,1);
                        else begin start<=1;respond(0,1);end
                    end else if(cmd==READ||cmd==WRITE)begin
                        if(busy && !tiled_busy_mmio_write)respond(8'd3,1);
                        else if(length==0||length>16'(MAX_TRANSFER)||
                                (TILED_MEM_MAP ? !tiled_transfer_valid :
                                 {8'b0,address}+{16'b0,length}>MEM_BYTES))respond(8'd4,1);
                        else begin transfer_index<=0;state<=MREQ;end
                    end else respond(8'd5,1);
                end
                MREQ:if(mem_ready)begin
                    if(cmd==READ)state<=MWAIT;
                    else if(16'(transfer_index)+16'd1==length)respond(0,1);
                    else transfer_index<=transfer_index+7'd1;
                end
                MWAIT:if(mem_rvalid)begin
                    payload[transfer_index]<=mem_rdata[transfer_addr[2:0]*8+:8];
                    if(16'(transfer_index)+16'd1==length)respond(0,length[6:0]+7'd1);
                    else begin transfer_index<=transfer_index+7'd1;state<=MREQ;end
                end
                RESPONSE:if(tx_ready)begin
                    if(tx_index>=2&&tx_index<10+response_length)tx_crc<=crc_byte(tx_crc,send_byte);
                    if(tx_index==11+response_length)state<=SYNC0;
                    else begin tx_index<=tx_index+7'd1;state<=TX_GAP;end
                end
                TX_GAP:state<=RESPONSE; // allows registered UART ready to fall
                default:state<=SYNC0;
            endcase
        end
    end
endmodule
