set root [file normalize [file join [file dirname [info script]] ../..]]
create_project -name phase4_sdram_bist -dir [pwd] -pn GW2AR-LV18QN88C8/I7 -device_version C
set_device -device_version C GW2AR-LV18QN88C8/I7
add_file [file join $root hardware/phase4_sdram/pll.v]
add_file [file join $root hardware/phase4_sdram/third_party/nestang_sdram.v]
add_file [file join $root hardware/phase4_sdram/bist.sv]
add_file [file join $root rtl/v2/sdram_refresh.sv]
add_file [file join $root rtl/v2/uart_tx.sv]
add_file [file join $root hardware/phase4_sdram/bist.cst]
add_file [file join $root hardware/phase4_sdram/bist.sdc]
set_option -top_module phase4_sdram_bist
set_option -verilog_std sysv2017
set_option -output_base_name phase4_sdram_bist
run all
exit
