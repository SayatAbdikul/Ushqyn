"""All256 INT8 codes for every current activation map, backpressure and flush."""
from collections import deque
import json
import os
from pathlib import Path
import random

import cocotb
from cocotb.triggers import Timer


@cocotb.test()
async def exact_mapping_stream(d):
    report_path = Path(os.environ['FUSION_PROBE'])
    root = next(p for p in report_path.parents if (p/'rtl/phase6').exists())
    report = json.loads(report_path.read_text())
    rng = random.Random(62944)
    d.clk.value = 0; d.rst_n.value = 0; d.flush.value = 0
    d.table_we.value = 0; d.table_index.value = 0; d.table_value.value = 0
    d.in_valid.value = 0; d.in_value.value = 0; d.out_ready.value = 0

    async def tick():
        d.clk.value = 0; await Timer(5, units='ns')
        d.clk.value = 1; await Timer(5, units='ns')

    await tick(); d.rst_n.value = 1; await tick()
    for case in report['cases']:
        table = (root/case['table']).read_bytes()
        assert len(table) == 256
        d.table_we.value = 1
        for i, value in enumerate(table):
            d.table_index.value = i; d.table_value.value = value
            await tick()
        d.table_we.value = 0
        for stalled in (False, True):
            codes = list(range(256))
            shuffled = codes.copy(); rng.shuffle(shuffled); codes += shuffled
            issued = retired = 0
            expected = deque()
            held = None
            cycles = 0
            while retired < len(codes):
                code = codes[issued] if issued < len(codes) else 0
                valid = issued < len(codes)
                ready = not stalled or rng.randrange(4) == 0
                d.clk.value = 0; d.in_valid.value = valid
                d.in_value.value = code; d.out_ready.value = ready
                await Timer(5, units='ns')
                ov, value = int(d.out_valid.value), int(d.result.value)
                if held is not None:
                    assert ov and value == held, 'fused result changed while stalled'
                if ov and ready:
                    assert expected and value == expected.popleft(), case['key']
                    retired += 1
                held = value if ov and not ready else None
                if valid and int(d.in_ready.value):
                    expected.append(table[code]); issued += 1
                d.clk.value = 1; await Timer(5, units='ns')
                cycles += 1
                assert cycles < 20000
            assert not expected and issued == len(codes)
            if not stalled:
                assert cycles == len(codes)+1, 'epilogue must sustain one byte per cycle'
            d.in_valid.value = 0; d.out_ready.value = 1; await tick()
        # A held lookup can be cancelled without exposing a stale next result.
        d.in_value.value = 35; d.in_valid.value = 1; d.out_ready.value = 0
        await tick(); d.in_valid.value = 0
        assert int(d.out_valid.value) and int(d.result.value) == table[35]
        d.flush.value = 1; await tick(); d.flush.value = 0
        assert not int(d.out_valid.value)
        d.in_valid.value = 1; d.in_value.value = 117
        await tick(); d.in_valid.value = 0
        assert int(d.out_valid.value) and int(d.result.value) == table[117]
        d.out_ready.value = 1; await tick()
