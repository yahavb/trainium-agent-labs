module lfsr (
    input  clk,
    input  reset_n,
    output reg [7:0] data
);
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n)
            data <= 8'b10001010;
        else
            data <= {data[0] ^ data[3] ^ data[5] ^ data[6], data[7:1]};
    end
endmodule
