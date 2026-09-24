create_clock -name clk27m -period 37.037 -waveform {0 18.518} [get_ports {sys_clk}]
set_false_path -from [get_ports {reset_button}]
set_false_path -to [get_ports {uart_tx_pin}]
set_false_path -to [get_ports {led*}]
