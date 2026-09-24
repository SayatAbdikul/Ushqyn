// 27 MHz system clock and 180-degree SDRAM clock for GW2AR-18C.
// Parameters follow the Gowin rPLL primitive settings used by the
// Apache-2.0 NESTang Tang Nano 20K SDRAM example (commit 918ae414).
module phase4_sdram_pll(input clkin, output clk, clk_sdram, lock);
    rPLL #(
        .FCLKIN("27"), .DEVICE("GW2AR-18C"),
        .FBDIV_SEL(1), .IDIV_SEL(1), .ODIV_SEL(32),
        .PSDA_SEL("1010"), .DUTYDA_SEL("1000"),
        .CLKOUT_FT_DIR(1'b1), .CLKOUTP_FT_DIR(1'b1),
        .CLKOUT_DLY_STEP(0), .CLKOUTP_DLY_STEP(0)
    ) pll (
        .CLKIN(clkin), .CLKOUT(clk), .CLKOUTP(clk_sdram), .LOCK(lock),
        .RESET(1'b0), .RESET_P(1'b0), .CLKFB(1'b0),
        .FBDSEL(6'b0), .IDSEL(6'b0), .ODSEL(6'b0),
        .PSDA(4'b0), .DUTYDA(4'b0), .FDLY(4'b0),
        .CLKOUTD(), .CLKOUTD3()
    );
endmodule
