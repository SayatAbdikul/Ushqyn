#!/usr/bin/env python3
"""Framed phase-2 serial client; RUN is never automatically retried."""
import argparse,binascii,json,struct,time
from pathlib import Path

CAPS,READ,WRITE,RUN,STATUS,ABORT,RESET=range(1,8)

def frame(cmd,seq=0,address=0,data=b'',length=0,version=2):
    if not 0<=address<1<<24 or not 0<=length<=65535:raise ValueError('frame fields')
    if cmd==WRITE:length=len(data)
    elif data:raise ValueError('payload only valid for WRITE')
    body=bytes([version,cmd,seq])+address.to_bytes(3,'little')+struct.pack('<H',length)+data
    return b'\xa5\x5a'+body+struct.pack('<H',binascii.crc_hqx(body,0xffff))


def parse_response(packet):
    if len(packet)<13 or packet[:2]!=b'\xa5\x5a':raise ValueError('response framing')
    length=int.from_bytes(packet[8:10],'little')
    if len(packet)!=12+length or not 1<=length<=65:raise ValueError('response length')
    if binascii.crc_hqx(packet[2:-2],0xffff)!=int.from_bytes(packet[-2:],'little'):raise ValueError('response CRC')
    if packet[2]!=2 or not packet[3]&128:raise ValueError('response version/command')
    return dict(command=packet[3]&127,sequence=packet[4],address=int.from_bytes(packet[5:8],'little'),status=packet[10],data=packet[11:-2])


def decode_status(data):
    if len(data)!=38:raise ValueError('status response size')
    names=['elapsed','compute_cycles','wait_cycles','control_cycles','useful_macs','read_bytes','write_bytes','layer_count','protocol_errors']
    return dict(busy=bool(data[0]),error=data[1],**dict(zip(names,struct.unpack('<9I',data[2:]))))


class Client:
    def __init__(self,serial):self.serial=serial;self.sequence=0
    def exchange(self,cmd,address=0,data=b'',length=0):
        seq=self.sequence;self.sequence=(seq+1)&255
        self.serial.write(frame(cmd,seq,address,data,length))
        # No blind retry: after an ambiguous RUN, query status explicitly.
        header=self.serial.read(10)
        if len(header)!=10:raise TimeoutError('truncated response header; do not automatically repeat RUN')
        size=int.from_bytes(header[8:10],'little')
        if not 1<=size<=65:raise ValueError('response size')
        result=parse_response(header+self.serial.read(size+2))
        if (result['command'],result['sequence'],result['address'])!=(cmd,seq,address):raise ValueError('response correlation')
        if result['status']:raise RuntimeError(f"device command status {result['status']}")
        return result['data']
    def capabilities(self):
        data=self.exchange(CAPS)
        if len(data)!=10 or data[:4]!=bytes([2,2,8,64]) or data[7]!=24 or int.from_bytes(data[8:10],'little')!=8194:raise ValueError('incompatible target')
        return int.from_bytes(data[4:7],'little')
    def write(self,address,data):
        for offset in range(0,len(data),64):self.exchange(WRITE,address+offset,data[offset:offset+64])
    def read(self,address,length):
        return b''.join(self.exchange(READ,address+offset,length=min(64,length-offset)) for offset in range(0,length,64))
    def run(self,pc=0,timeout=5):
        self.exchange(RUN,pc);deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            result=decode_status(self.exchange(STATUS))
            if not result['busy']:
                if result['error']:raise RuntimeError(f"engine error {result['error']}")
                return result
        raise TimeoutError('execution timeout; query or ABORT before retry')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',required=True);p.add_argument('--image',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    import serial
    with serial.Serial(a.port,115200,timeout=2,write_timeout=2) as uart:
        client=Client(uart);capacity=client.capabilities();report={'capacity':capacity,'physical_board':True}
        if a.image:
            if not a.manifest:raise ValueError('board image requires its manifest')
            import hashlib
            metadata=json.loads(a.manifest.read_text());image=a.image.read_bytes()
            if metadata.get('image_sha256')!=hashlib.sha256(image).hexdigest() or metadata.get('target')!='tang-nano-20k-v2' or len(image)!=capacity:raise ValueError('image/manifest mismatch')
            client.exchange(RESET);client.write(0,image)
            if client.read(0,len(image))!=image:raise ValueError('program readback mismatch')
            report['run']=client.run(metadata['entry']);report['image_sha256']=metadata['image_sha256']
        a.report.write_text(json.dumps(report,indent=2)+'\n')
