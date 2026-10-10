module lfsr (
    input clk,
    input reset_n,
    output reg [7:0] data
);
    always @(posedge clk) begin
        data <= {data[6:0], data[0] ^ data[3] ^ data[5] ^ data[6]};
    end

    always @(reset_n) begin
        data <= 8'b10001010;
    end
endmodule
