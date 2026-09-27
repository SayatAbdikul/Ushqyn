"""Check the generated host's 27 MHz / 750 kbaud 36-cycle UART divisor."""
import cocotb
from cocotb.triggers import Timer


@cocotb.test()
async def uart_27mhz_750k_round_trip(d):
    samples = []
    seen = []
    d.clk.value = 0
    d.rst_n.value = 0
    d.rx_i.value = 1

    async def cycle(count):
        for _ in range(count):
            d.clk.value = 0
            await Timer(5, units='ns')
            d.clk.value = 1
            await Timer(5, units='ns')
            samples.append(int(d.tx_o.value))
            if int(d.received_valid.value):
                seen.append(int(d.received.value))

    await cycle(8)
    d.rst_n.value = 1
    await cycle(8)
    words = [0x00, 0xFF, 0x55, 0xA6, 0x81]
    for value in words:
        for bit in [0]+[(value >> i) & 1 for i in range(8)]+[1, 1, 1]:
            d.rx_i.value = bit
            await cycle(36)
    d.rx_i.value = 1
    await cycle(420)
    assert seen == words, ('UART RX at 36 cycles/bit', seen)

    decoded = []
    cursor = 1
    while cursor + 342 < len(samples):
        if samples[cursor-1] and not samples[cursor]:
            assert not samples[cursor+18], 'transmit start-bit center'
            bits = [samples[cursor+54+36*i] for i in range(8)]
            assert samples[cursor+342], 'transmit stop-bit center'
            decoded.append(sum(bit << i for i, bit in enumerate(bits)))
            cursor += 360
        else:
            cursor += 1
    assert decoded == words, ('UART TX at 36 cycles/bit', decoded)
