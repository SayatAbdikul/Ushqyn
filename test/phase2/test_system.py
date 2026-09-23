import os,sys,random,struct,json
from pathlib import Path
import numpy as np
import cocotb
from cocotb.triggers import Timer
from host import frame,parse_response,decode_status,CAPS,READ,WRITE,RUN,STATUS,ABORT,RESET
from hardware_v2 import Descriptor

async def tick(d,n=1):
    for _ in range(n):
        d.clk.value=0;await Timer(5,units='ns');d.clk.value=1;await Timer(5,units='ns')

class Link:
    def __init__(self,d):self.d=d;self.seq=0;self.rng=random.Random(5)
    async def packet(self,data,stalls=True):
        d=self.d;d.tx_ready.value=0
        for b in data:
            d.rx_data.value=b;d.rx_valid.value=1;await tick(d);d.rx_valid.value=0
            if stalls:await tick(d,self.rng.randrange(3))
        d.rx_valid.value=0;reply=[];total=None
        for _ in range(100000):
            ready=not stalls or self.rng.randrange(4)!=0
            d.tx_ready.value=ready;d.clk.value=0;await Timer(5,units='ns')
            if ready and int(d.tx_valid.value):
                reply.append(int(d.tx_data.value))
                if len(reply)==10:total=12+int.from_bytes(bytes(reply[8:10]),'little')
            d.clk.value=1;await Timer(5,units='ns')
            if total and len(reply)==total:return parse_response(bytes(reply))
        raise AssertionError('response timeout')
    async def call(self,cmd,addr=0,data=b'',length=0,status=0):
        seq=self.seq;self.seq=(seq+1)&255;r=await self.packet(frame(cmd,seq,addr,data,length))
        assert (r['command'],r['sequence'],r['status'])==(cmd,seq,status),r
        return r['data']
    async def write(self,addr,data):
        for i in range(0,len(data),64):await self.call(WRITE,addr+i,data[i:i+64])
    async def read(self,addr,length):
        return b''.join([await self.call(READ,addr+i,length=min(64,length-i)) for i in range(0,length,64)])
    async def wait(self):
        for _ in range(2000):
            r=decode_status(await self.call(STATUS))
            if not r['busy']:return r
        raise AssertionError('core did not finish')

async def initialize(d):
    d.clk.value=0;d.rst_n.value=0;d.rx_valid.value=0;d.rx_data.value=0;d.tx_ready.value=0;await tick(d,3);d.rst_n.value=1;await tick(d,3);return Link(d)

@cocotb.test()
async def protocol_and_inference(d):
    l=await initialize(d)
    assert (await l.call(CAPS))[:4]==bytes([2,2,8,64])
    payload=bytes(range(64));await l.write(30001,payload);assert await l.read(30001,64)==payload
    bad=bytearray(frame(WRITE,1,30001,b'bad'));bad[-1]^=1
    assert (await l.packet(bad))['status']==1
    assert await l.read(30001,64)==payload
    # An oversized WRITE contains a syntactically valid nested WRITE frame;
    # every declared byte must be drained, never treated as a new command.
    oversized=b'x'*64+frame(WRITE,9,30001,b'evil')
    assert (await l.packet(frame(WRITE,4,30001,oversized)))['status']==4
    assert await l.read(30001,64)==payload
    assert (await l.packet(frame(CAPS,version=3)))['status']==2
    await l.call(READ,32760,length=16,status=4);await l.call(RUN,1,status=4)
    # Drop an incomplete frame, wait beyond the watchdog, recover.
    d.tx_ready.value=0
    for b in frame(WRITE,3,30001,b'incomplete')[:-5]:d.rx_valid.value=1;d.rx_data.value=b;await tick(d)
    d.rx_valid.value=0;await tick(d,110);await l.call(CAPS)
    # FC count=11 exercises both eight-lane reductions and masked tails.
    desc=Descriptor(1,input=512,output=544,weight=576,params=640,count=11,outputs=2,row_stride=16,next_pc=64)
    await l.write(0,desc.encode()+Descriptor(0).encode())
    x=np.array([-128,-100,-3,-1,0,1,3,20,40,100,127],np.int8);w=np.array([[1,-1,2,-2,1,1,-1,0,1,0,1],[-1,1,-2,2,-1,-1,1,0,-1,0,-1]],np.int8)
    weights=np.zeros((2,16),np.int8);weights[:,:11]=w
    await l.write(512,x.tobytes()+b'\0'*5);await l.write(576,weights.tobytes())
    zx=-117;bias=[17,-19];M=1073741824;S=33;zy=-3
    params=b''.join(struct.pack('<iiBbbbbb',bias[c]-zx*int(w[c].sum()),M,S,zy,zx,-128,127,0)+b'\0\0' for c in range(2));await l.write(640,params)
    for run in range(3):
        await l.call(RUN);r=await l.wait();assert r['error']==0,r
        expected=[]
        for c in range(2):
            a=bias[c]+sum((int(x[k])-zx)*int(w[c,k]) for k in range(11));q,rem=divmod(abs(a*M),1<<S);q+=2*rem>=1<<S;expected.append(max(-128,min(127,(-q if a<0 else q)+zy)))
        assert await l.read(544,2)==np.array(expected,np.int8).tobytes()
        assert r['useful_macs']==22 and r['layer_count']==1 and r['elapsed']==r['compute_cycles']+r['wait_cycles']+r['control_cycles'],r
        x=np.roll(x,1);await l.write(512,x.tobytes())
    # Long job allows checking busy ownership and ABORT.
    long=Descriptor(1,input=1024,output=28000,weight=4096,params=24000,count=784,outputs=12,row_stride=784,next_pc=64)
    await l.write(0,long.encode());await l.write(24000,params[:16]*12)
    await l.call(RUN);await l.call(WRITE,30001,b'x',status=3);await l.call(READ,30001,length=1,status=3);await l.call(RUN,status=3)
    await l.call(ABORT);assert not (await l.wait())['busy'];await l.call(RESET)
    assert await l.read(30001,64)==payload
    cleared=decode_status(await l.call(STATUS));assert cleared['elapsed']==0 and cleared['error']==0
    for field,value in [('count',0),('input',0x100000),('next_pc',1),('stride_w',2)]:
        from dataclasses import replace
        await l.write(0,replace(desc,**{field:value}).encode());await l.call(RUN);assert (await l.wait())['error']==1
    await l.write(0,desc.encode())
    invalid=bytearray(params);invalid[8]=63;await l.write(640,invalid);await l.call(RUN);assert (await l.wait())['error']==4
    overflow=struct.pack('<iiBbbbbb',2147483647,1073741824,33,0,0,-128,127,0)+b'\0\0'
    await l.write(512,bytes([127])*16);await l.write(576,bytes([127])*32);await l.write(640,overflow*2)
    await l.call(RUN);assert (await l.wait())['error']==5
    wrong=bytearray(Descriptor(0).encode());wrong[4]=1;await l.write(0,wrong);await l.call(RUN);assert (await l.wait())['error']==2
    await l.write(0,Descriptor(0).encode());await l.call(RUN);assert (await l.wait())['error']==0

@cocotb.test()
async def complete_mlp_jobs(d):
    root=Path(os.environ['REPO_ROOT']);fixture=root/'work/phase2/mlp'
    if not (fixture/'board.bin').exists():
        raise AssertionError('run tools/phase2/prepare_mlp.py before full phase-2 validation')
    l=await initialize(d);meta=json.loads((fixture/'board.json').read_text());f=np.load(fixture/'checks.npz')
    await l.write(0,(fixture/'board.bin').read_bytes()[:meta['used_bytes']])
    records=[]
    count=int(os.environ.get('PHASE2_JOBS','1000'))
    for i in range(count):
        await l.write(meta['inputs']['input'],f['inputs'][i].tobytes());await l.call(RUN)
        # Fast-forward a bounded number of real RTL cycles, then inspect STATUS.
        await tick(d,15000)
        r=await l.wait();assert r['error']==0,r
        for name,address in meta['outputs'].items():assert await l.read(address,f['outputs'][i].size)==f['outputs'][i].tobytes(),f'job {i}'
        if i<3:
            for name,address in meta['tensors'].items():
                key='layer_'+name
                if key in f:assert await l.read(address,f[key][i].size)==f[key][i].tobytes(),name
        assert r['useful_macs']==meta['macs'] and r['elapsed']==r['compute_cycles']+r['wait_cycles']+r['control_cycles']
        records.append(r)
        if (i+1)%100==0:d._log.info('Strict MLP jobs: %d/%d',i+1,count)
    (fixture/'rtl-results.json').write_text(json.dumps(dict(scope='RTL simulation of board system and synchronous inferred SRAM; not physical board',jobs=count,integer_mismatches=0,counters=records),indent=2)+'\n')
