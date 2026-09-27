#!/usr/bin/env python3
"""Short guarded UART WRITE batching experiment on an idle board.

Existing receiver discards RX bytes while responding. Zero-byte guard intervals
keep successive frames separated on the wire; every response is still checked.
This tests a bounded batch/guard pair, not a generally queue-capable protocol.
RUN/control writes are never batched and there is no automatic retry.
"""
import argparse
import fcntl
import json
from pathlib import Path
import statistics
import sys
import time

import serial
from variants import ROOT, sha, check_frozen
import run_screening as screening

sys.path[:0]=[str(ROOT/'tools/phase4'),str(ROOT/'tools/phase2')]
from tiled_host import TiledClient
from host import frame, parse_response, WRITE, STATUS, decode_status


class BatchedClient(TiledClient):
    def write_guarded(self,address,data,guard=32,batch=8):
        if not(0x800000<=address<=0xffffff and address+len(data)<=0x1000000):
            raise ValueError('batch only idle SDRAM data writes')
        if guard not in (32,64) or not 1<=batch<=8:raise ValueError('unscreened batching parameters')
        transfers=[]
        for offset in range(0,len(data),64):
            seq=self.sequence;self.sequence=(seq+1)&255
            transfers.append((seq,address+offset,data[offset:offset+64]))
        for start in range(0,len(transfers),batch):
            group=transfers[start:start+batch]
            packet=b''.join(frame(WRITE,seq,addr,chunk)+bytes(guard) for seq,addr,chunk in group)
            if self.serial.write(packet)!=len(packet):raise IOError('partial batch write; no retry')
            for seq,addr,_ in group:
                header=self.serial.read(10)
                if len(header)!=10:raise TimeoutError('missing batch response; no retry')
                size=int.from_bytes(header[8:10],'little')
                if size!=1:raise ValueError('WRITE response size')
                response=parse_response(header+self.serial.read(size+2))
                if (response['command'],response['sequence'],response['address'],response['status'],response['data'])!=(WRITE,seq,addr,0,b''):
                    raise ValueError('batch response mismatch')


def run(output,port):
    check_frozen();screening.require_board_free(port)
    # The current image must be one that just passed the physical compute screen.
    evidence=ROOT/'work/phase6/experiments-v1/quick-b/report.json'
    prior=json.loads(evidence.read_text())
    if prior['status']!='passed-short-screen':raise ValueError('missing board predecessor')
    if output.exists():raise FileExistsError('preserve previous evidence')
    output.mkdir(parents=True)
    lock=(ROOT/'work/phase6/physical-board.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    report=dict(status='running',physical_board=True,source_sha256=sha(Path(__file__)),
        predecessor_sha256=sha(evidence),predecessor_image=prior['programming'][-1]['bitstream_sha256'],
        baud=750000,address=0xf00000,bytes=8192,batch=8,records=[])
    def save():screening.save_json(output/'report.json',report)
    save()
    try:
        with serial.Serial(port,750000,timeout=3,write_timeout=3) as uart:
            uart.reset_input_buffer();client=BatchedClient(uart);client.capabilities()
            status=decode_status(client.exchange(STATUS))
            if status['busy'] or status['error'] or status['protocol_errors']:raise ValueError('board not idle and clean')
            for guard in (0,64,32):
                for repeat in range(3):
                    data=bytes((i*73+repeat*31+guard)%256 for i in range(8192))
                    begin=time.monotonic()
                    if guard:client.write_guarded(0xf00000,data,guard)
                    else:client.write(0xf00000,data)
                    uploaded=time.monotonic()-begin
                    begin=time.monotonic();readback=client.read(0xf00000,len(data));read_seconds=time.monotonic()-begin
                    if readback!=data:raise ValueError('batched SDRAM readback mismatch')
                    status=decode_status(client.exchange(STATUS))
                    if status['busy'] or status['error'] or status['protocol_errors']:raise ValueError('protocol status failure')
                    report['records'].append(dict(guard_bytes=guard,repeat=repeat+1,upload_seconds=uploaded,
                        readback_seconds=read_seconds,upload_bytes_per_second=len(data)/uploaded,matched=True))
                    save()
        report['summary']={str(g):dict(median_upload_seconds=statistics.median(r['upload_seconds'] for r in report['records'] if r['guard_bytes']==g)) for g in (0,64,32)}
        report['status']='passed-short-transfer-screen';save();print(json.dumps(report['summary']),flush=True)
    except BaseException as error:
        report.update(status='failed',failure=repr(error));save();raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',default='/dev/cu.usbserial-20250303171')
    args=parser.parse_args()
    if not args.run:raise SystemExit('Use --run for this nine-upload screen; no automatic FPGA programming.')
    run(args.output,args.port)
