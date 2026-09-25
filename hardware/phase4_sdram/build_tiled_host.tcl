set root [file normalize [file join [file dirname [info script]] ../..]]
set ip [file join $root work/phase4/ip-generated/sdram_controller_hs.v]
if {![file exists $ip]} {error "Generate matching SDRAM HS IP first: $ip"}
create_project -name phase4_tiled_host -dir [pwd] -pn GW2AR-LV18QN88C8/I7 -device_version C
set_device -device_version C GW2AR-LV18QN88C8/I7
foreach source {
    rtl/v2/target_pkg.sv
    hardware/phase4_sdram/pll_dma_20.v
    hardware/phase4_sdram/tiled_host.sv
    rtl/v2/requantizer.sv
    rtl/v2/scratchpad.sv
    rtl/v2/engine.sv
    rtl/v2/tile_dma.sv
    rtl/v2/tiled_core.sv
    rtl/v2/command.sv
    rtl/v2/tiled_host_bridge.sv
    rtl/v2/tile_sequencer.sv
    rtl/v2/hs_sdram_burst_port.sv
    rtl/v2/sdram_refresh.sv
    rtl/v2/uart_rx.sv
    rtl/v2/uart_tx.sv
} {add_file [file join $root $source]}
add_file $ip
add_file [file join $root hardware/phase4_sdram/tiled_host.cst]
add_file [file join $root hardware/phase4_sdram/bist.sdc]
set_option -top_module phase4_tiled_host
set_option -verilog_std sysv2017
set_option -output_base_name phase4_tiled_host
run all
exit
