create_clock -name clk27m -period 37.037 -waveform {0 18.518} [get_ports {sys_clk}]
# UART RX and reset are asynchronous and enter explicit synchronizers.
set_false_path -from [get_ports {uart_rx}]
set_false_path -from [get_ports {reset_button}]
# UART TX and status LEDs are not source-synchronous interfaces.
set_false_path -to [get_ports {uart_tx}]
set_false_path -to [get_ports {led*}]
