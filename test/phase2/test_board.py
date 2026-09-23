"""Exercise the exact physical top including reset synchronizer and UART wires."""
import cocotb
from cocotb.triggers import Timer
from host import frame,parse_response,CAPS,WRITE,READ,RUN,STATUS
from hardware_v2 import Descriptor

@cocotb.test()
async def serial_board_top(d):
    period=10
    async def cycle(n=1):
        for _ in range(n):d.sys_clk.value=0;await Timer(5,units='ns');d.sys_clk.value=1;await Timer(5,units='ns')
    d.sys_rst_n.value=0;d.uart_rx.value=1;await cycle(3);d.sys_rst_n.value=1;await cycle(8)
    async def send(data):
        for byte in data:
            for bit in [0]+[(byte>>i)&1 for i in range(8)]+[1]:d.uart_rx.value=bit;await cycle(period)
        d.uart_rx.value=1
    async def receive():
        data=[];needed=None
        for _ in range(100):
            for _ in range(10000):
                if int(d.uart_tx.value)==0:break
                await cycle()
            else:raise AssertionError('UART response timeout')
            await cycle(period+period//2);byte=0
            for bit in range(8):byte|=int(d.uart_tx.value)<<bit;await cycle(period)
            assert int(d.uart_tx.value)==1,'UART stop bit';await cycle(period//2)
            data.append(byte)
            if len(data)==10:needed=12+int.from_bytes(bytes(data[8:10]),'little')
            if needed and len(data)==needed:return parse_response(bytes(data))
        raise AssertionError('UART response too long')
    async def call(cmd,addr=0,data=b'',length=0):
        # Response may begin before the host has finished the stop bit.
        reader=cocotb.start_soon(receive())
        await send(frame(cmd,address=addr,data=data,length=length));r=await reader
        assert r['status']==0,r
        return r['data']
    assert (await call(CAPS))[:4]==bytes([2,2,8,64])
    await call(WRITE,0,Descriptor(0).encode());assert await call(READ,0,length=64)==Descriptor(0).encode()
    await call(RUN);r=await call(STATUS);assert r[0]==0 and r[1]==0
