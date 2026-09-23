import json
import os
from pathlib import Path
import numpy as np
import cocotb
from hardware_v2 import TARGET
from test_system import initialize,tick
from host import CAPS,RUN


@cocotb.test()
async def change_mlp_cnn_mlp_without_bitstream_reload(d):
    root=Path(os.environ['REPO_ROOT'])
    link=await initialize(d)
    caps=await link.call(CAPS)
    assert int.from_bytes(caps[8:10],'little')==TARGET['target_id']
    for label,fixture,cycles in (
        ('mlp',root/'work/phase2/mlp',15000),
        ('smallcnn',root/'work/phase3/smallcnn',185000),
        ('mlp',root/'work/phase2/mlp',15000)):
        meta=json.loads((fixture/'board.json').read_text())
        check=np.load(fixture/'checks.npz',allow_pickle=False)
        await link.write(0,(fixture/'board.bin').read_bytes()[:meta['used_bytes']])
        assert await link.read(0,meta['used_bytes'])==(fixture/'board.bin').read_bytes()[:meta['used_bytes']]
        await link.write(meta['inputs']['input'],check['inputs'][0].tobytes())
        await link.call(RUN)
        await tick(d,cycles)
        status=await link.wait()
        assert status['error']==0,(label,status)
        output_address=next(iter(meta['outputs'].values()))
        assert await link.read(output_address,check['outputs'][0].size)==check['outputs'][0].tobytes(),label
        assert status['useful_macs']==meta['macs']
        assert status['elapsed']==status['compute_cycles']+status['wait_cycles']+status['control_cycles']
