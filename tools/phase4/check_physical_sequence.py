#!/usr/bin/env python3
"""Physical autonomous liveness guard, abort/recovery and memory coherence."""
import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

import serial

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'compiler'))
from hardware_v2 import Descriptor
from phase4_sequence import compile_sequence
from run_physical_sequence import RecoverableUploadClient, execute
from host import RESET, ABORT, STATUS, decode_status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',default='/dev/cu.usbserial-20250303171')
    p.add_argument('--bitstream',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    plan=json.loads((ROOT/'work/phase4/kws-pingpong-plan.json').read_text())
    image=(ROOT/'work/phase4/kws-pingpong-image.bin').read_bytes()
    commands,payload,schedule=compile_sequence(plan,image,True,False)
    # Use the already resident KWS/VWW-independent schedule data freshly.
    with serial.Serial(a.port,115200,timeout=5,write_timeout=5) as port:
        port.reset_input_buffer();c=RecoverableUploadClient(port)
        c.capabilities()
        deadline=time.monotonic()+30
        while decode_status(c.exchange(STATUS))['busy']:
            if time.monotonic()>deadline:raise TimeoutError('previous work did not drain')
        c.exchange(RESET)
        # Directed cache coherence: neighboring bytes must survive masked
        # writes, alternating tags and line/bank/end-of-device boundaries.
        regions=[0x400000,0x400040,0x1fffc0,0x200000,0x7fffc0]
        expected={base:bytearray((i*31+base//64)&255 for i in range(64)) for base in regions}
        for base,data in expected.items():c.write_external(base,data)
        for trial in range(25):
            base=regions[trial%len(regions)];offset=(trial*11)%64
            value=bytes([(trial*19+7)&255]);c.write_external(base+offset,value)
            expected[base][offset]=value[0]
            for address,data in expected.items():
                if c.read_external(address,64)!=data:raise AssertionError('masked write/cache tag coherence')
        # A long first Conv is large enough for an immediate conflicting DMA
        # to be rejected by the sequencer before any live SRAM is changed.
        tile=schedule['tiles'][0]
        for load in tile['loads']:
            raw=payload[load['ext']:load['ext']+load['bytes']]
            if load['role']=='input':raw=(ROOT/'work/phase4/rtl-kws/input.bin').read_bytes()
            c.write(load['sram'],raw)
        def cmd(op,flags=0,x=0,y=0,z=0):return struct.pack('<BBHIII',op,flags,0,x,y,z)
        guard=(cmd(2,0,tile['pc'],tile['live'][0]|tile['live'][1]<<16)+
               cmd(1,1,0x400000,tile['live'][0]+128,64)+cmd(3,3)+cmd(0))
        c.write(0x500000,guard);c.write(0x410000,b'\x01')
        deadline=time.monotonic()+5
        while True:
            status=decode_status(c.exchange(STATUS))
            if not status['busy']:break
            if time.monotonic()>deadline:raise TimeoutError('guard recovery')
        if status['error']!=9:raise AssertionError(f'live guard not enforced: {status}')
        c.exchange(RESET)
        # Long DMA-only command list ensures ABORT reaches an active program.
        stream=b''.join(cmd(1,1,0x400000,0,32768)+cmd(3,2) for _ in range(400))+cmd(0)
        c.write(0x500000,stream);c.write(0x410000,b'\x01')
        before=decode_status(c.exchange(STATUS))
        if not before['busy']:raise AssertionError('abort test did not reach busy execution')
        c.exchange(ABORT)
        deadline=time.monotonic()+5
        while True:
            after=decode_status(c.exchange(STATUS))
            if not after['busy']:break
            if time.monotonic()>deadline:raise TimeoutError('abort drain')
        c.exchange(RESET)
        # The DMA's abort error is sticky until the next transfer (the
        # established DMA ABI). Recovery must execute and verify new work.
        recovery=cmd(1,1,regions[0],0,64)+cmd(3,2)+cmd(0)
        c.write(0x500000,recovery);c.write(0x410000,b'\x01');c.wait_idle(5)
        if c.read(0x410000,2)!=b'\0\0':raise AssertionError('reset did not clear sequencer error')
        if c.read(0,64)!=expected[regions[0]]:raise AssertionError('post-abort transfer mismatch')
    result={'status':'passed','physical_board':True,
            'bitstream_sha256':hashlib.sha256(a.bitstream.read_bytes()).hexdigest(),
            'masked_write_trials':25,'neighbor_readbacks':125,
            'cache_line_addresses':regions,'live_guard_error':9,
            'abort_while_busy':True,'reset_and_transfer_recovery':True}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
