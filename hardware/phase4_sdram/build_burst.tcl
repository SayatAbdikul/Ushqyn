set root [file normalize [file join [file dirname [info script]] ../..]]
set ip [file join $root work/phase4/ip-generated/sdram_controller_hs.v]
if {![file exists $ip]} {error "Generate matching SDRAM HS IP first: $ip"}
create_project -name phase4_burst_smoke -dir [pwd] -pn GW2AR-LV18QN88C8/I7 -device_version C
set_device -device_version C GW2AR-LV18QN88C8/I7
add_file [file join $root hardware/phase4_sdram/pll.v]
add_file $ip
add_file [file join $root hardware/phase4_sdram/burst_smoke.sv]
add_file [file join $root rtl/v2/hs_sdram_port.sv]
add_file [file join $root rtl/v2/sdram_refresh.sv]
add_file [file join $root rtl/v2/uart_tx.sv]
add_file [file join $root hardware/phase4_sdram/bist.cst]
add_file [file join $root hardware/phase4_sdram/bist.sdc]
set_option -top_module phase4_burst_smoke
set_option -verilog_std sysv2017
set_option -output_base_name phase4_burst_smoke
run all
exit
