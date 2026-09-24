"""Boardless refresh scheduling; behavioral ack is not vendor-IP validation."""

import cocotb
from cocotb.triggers import Timer


@cocotb.test()
async def refresh_cadence_stalls_catchup_and_reset(d):
    d.clk.value = 0; d.rst_n.value = 0
    d.init_done.value = 0; d.controller_idle.value = 0; d.refresh_ack.value = 0
    outstanding = None
    issued = acked = 0

    async def step(idle=True):
        nonlocal outstanding, issued, acked
        d.clk.value = 0
        d.controller_idle.value = int(idle and outstanding is None)
        d.refresh_ack.value = int(outstanding == 0)
        await Timer(5, units='ns')
        request = bool(int(d.refresh_req.value))
        if request:
            assert idle and outstanding is None
            issued += 1
        if outstanding == 0:
            acked += 1
            outstanding = None
        elif outstanding is not None:
            outstanding -= 1
        if request:
            outstanding = 2
        d.clk.value = 1
        await Timer(5, units='ns')
        assert int(d.pending_refreshes.value) >= int(outstanding is not None)

    await step(); d.rst_n.value = 1
    for _ in range(32):
        await step()
    assert issued == 0 and int(d.pending_refreshes.value) == 0
    d.init_done.value = 1
    for _ in range(80):
        await step()
    assert issued >= 9 and acked >= 9
    assert int(d.pending_refreshes.value) <= 1
    before = issued
    for _ in range(80):
        await step(idle=False)
    assert issued == before
    assert int(d.pending_refreshes.value) >= 10
    for _ in range(200):
        await step()
    assert issued > before + 10
    assert issued >= acked
    assert int(d.pending_refreshes.value) <= 1
    assert not int(d.refresh_deadline_missed.value)
    d.rst_n.value = 0
    await step()
    assert int(d.pending_refreshes.value) == 0
    assert not int(d.refresh_deadline_missed.value)
