// Numerics v2: one elastic output register. Inputs accepted only on ready/valid.
module v2_requantizer (
    input logic clk, rst_n, flush,
    input logic in_valid, output logic in_ready,
    input logic signed [31:0] accumulator, multiplier,
    input logic [5:0] shift,
    input logic signed [7:0] zero_point,
    output logic out_valid, input logic out_ready,
    output logic signed [7:0] result
);
    logic signed [63:0] product;
    logic [63:0] magnitude, rounded;
    logic signed [64:0] scaled;
    always_comb begin
        product = accumulator * multiplier;
        magnitude = product[63] ? (~$unsigned(product)+64'd1) : $unsigned(product);
        rounded = (shift==0) ? magnitude : (magnitude+(64'd1<<(shift-1)))>>shift;
        scaled = (product[63] ? -$signed({1'b0,rounded}) : $signed({1'b0,rounded})) + {{57{zero_point[7]}},zero_point};
    end
    assign in_ready = !out_valid || out_ready;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin out_valid<=0; result<=0; end
        else if (flush) begin out_valid<=0; result<=0; end
        else if (in_ready) begin
            out_valid<=in_valid;
            if (in_valid) begin
                if (scaled>65'sd127) result<=8'sd127;
                else if (scaled < -65'sd128) result<=-8'sd128;
                else result<=scaled[7:0];
            end
        end
    end
endmodule
