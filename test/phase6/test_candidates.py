"""Execute optimizer bytecode on sequencer + engine + SRAM + DMA RTL.

External RAM is a stalled abstract port, not the physical SDRAM controller.
The synthetic fixture deliberately crosses the 16 KiB tiling threshold.
"""
import hashlib
import json
from pathlib import Path
import random

import cocotb
from cocotb.triggers import Timer

from integer_reference import evaluate
from scheduler.current_abi import CostModel, optimize
from scheduler.fixtures import wide_chain

ROOT = Path(__file__).resolve().parents[2]


def crc16(data):
    crc = 65535
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ (0x1021 if crc & 0x8000 else 0)) & 65535
    return crc


@cocotb.test()
async def generated_candidates_match_integer_oracle(d):
    memory = bytearray(8*1024*1024)
    rng = random.Random(6106)
    pending, held, response = None, None, []
    d.clk.value = 0; d.rst_n.value = 0
    d.rx_valid.value = 0; d.rx_data.value = 0; d.tx_ready.value = 1
    d.ext_ready.value = 0; d.ext_rvalid.value = 0; d.ext_rdata.value = 0
    d.memory_initialized.value = 1; d.memory_port_busy.value = 0

    async def step(byte=None):
        nonlocal pending, held
        d.clk.value = 0; d.rx_valid.value = byte is not None; d.rx_data.value = byte or 0
        ready = pending is None and rng.randrange(5) != 0
        d.ext_ready.value = ready; d.ext_rvalid.value = 0
        if pending:
            delay, data = pending
            if delay == 0:
                d.ext_rvalid.value = 1; d.ext_rdata.value = data; pending = None
            else:
                pending = delay-1, data
        await Timer(5, units='ns')
        if int(d.tx_valid.value):
            response.append(int(d.tx_data.value))
        req = (int(d.ext_addr.value), int(d.ext_wr.value), int(d.ext_wdata.value), int(d.ext_wstrb.value)) if int(d.ext_req.value) else None
        if held is not None:
            assert req == held, 'request changed under external stall'
        held = req if req is not None and not ready else None
        if req is not None and ready:
            address, write, data, mask = req
            assert address % 8 == 0 and address+8 <= len(memory)
            if write:
                for lane in range(8):
                    if mask & (1 << lane):
                        memory[address+lane] = (data >> (8*lane)) & 255
            else:
                assert pending is None
                pending = rng.randrange(1, 5), int.from_bytes(memory[address:address+8], 'little')
        d.clk.value = 1
        await Timer(5, units='ns')

    sequence = 0

    async def command(kind, address=0, data=b'', length=0):
        nonlocal sequence
        sequence = (sequence+1) & 255
        contents = bytes([2, kind, sequence])+address.to_bytes(3, 'little')+(len(data) if data else length).to_bytes(2, 'little')+data
        frame = b'\xa5\x5a'+contents+crc16(contents).to_bytes(2, 'little')
        assert not response
        for byte in frame:
            await step(byte)
        for _ in range(30000):
            if len(response) >= 10 and len(response) == 12+int.from_bytes(bytes(response[8:10]), 'little'):
                break
            await step()
        else:
            raise AssertionError('framed command timeout')
        packet = bytes(response); response.clear()
        assert packet[:3] == b'\xa5\x5a\x02' and packet[3:5] == bytes([kind | 128, sequence])
        assert crc16(packet[2:-2]) == int.from_bytes(packet[-2:], 'little') and packet[10] == 0
        await step()
        return packet[11:-2]

    await step(); d.rst_n.value = 1; await step()
    program, x = wide_chain()
    expected = evaluate(program, {'x': x})['y'].tobytes()
    costs = CostModel(json.loads((ROOT/'docs/research/evidence/phase4/physical-sequence-costs.json').read_text()))
    artifacts, _ = optimize(program, costs)
    seen, records = {}, {}
    for policy, artifact in artifacts.items():
        key = hashlib.sha256(artifact['commands']+artifact['payload']).hexdigest()
        if key in seen:
            records[policy] = {'identical_artifact_to': seen[key]}
            continue
        seen[key] = policy
        await command(7)
        memory[:] = bytes(len(memory))
        memory[:len(artifact['payload'])] = artifact['payload']; memory[:x.nbytes] = x.tobytes()
        for offset in range(0, len(artifact['commands']), 64):
            await command(3, 0x500000+offset, artifact['commands'][offset:offset+64])
        await command(3, 0x410000, b'\x01')
        for _ in range(3000000):
            await step()
            if not int(d.seq_busy.value):
                break
        else:
            raise AssertionError(f'{policy} timeout')
        regs = await command(2, 0x410000, length=32)
        assert regs[1] == 0 and regs[4:8] == b'SEQ4', (policy, regs)
        output = artifact['schedule']['final_output']
        assert memory[output['ext']:output['ext']+output['bytes']] == expected, policy
        records[policy] = {'status': 'passed', 'simulated_cycles': int.from_bytes(regs[8:12], 'little'),
                           'simulated_overlap_cycles': int.from_bytes(regs[20:24], 'little'),
                           'commands_sha256': hashlib.sha256(artifact['commands']).hexdigest(),
                           'payload_sha256': hashlib.sha256(artifact['payload']).hexdigest()}
        d._log.info('%s passed (%s)', policy, records[policy])
    assert records['prefer16-prefetch']['simulated_overlap_cycles'] > 0
    files = list((ROOT/'rtl/v2').glob('*.sv')) + [Path(__file__), ROOT/'compiler/scheduler/fixtures.py', ROOT/'compiler/scheduler/current_abi.py']
    report = {'status': 'passed', 'scope': 'synthetic chain on sequencer/engine/DMA/SRAM RTL with stalled abstract external RAM; not physical latency',
              'physical_board': False, 'random_stall_seed': 6106, 'candidates': records,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (ROOT/'work/phase6/rtl-candidates.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
