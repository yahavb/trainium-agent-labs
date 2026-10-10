module lfsr (
    input clk,
    input reset_n,
    output reg [7:0] data
);

    always @(posedge clk) begin
        if (!reset_n) begin
            data <= 8'b10001010;
        end else begin
            reg new_bit;
            new_bit = data[0] ^ data[3] ^ data[5] ^ data[6];
            data <= {data[6:0], new_bit};
        end
    end

endmodule
