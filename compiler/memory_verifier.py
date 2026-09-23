"""Independent address/liveness checks for a proposed physical SRAM layout."""


def verify(layout, capacity, program_bytes):
    if capacity<1024 or capacity & (capacity-1) or program_bytes<0 or program_bytes>capacity or program_bytes%8:
        raise ValueError('invalid physical SRAM geometry')
    seen = set()
    high = program_bytes
    for region in layout:
        name, start, size = region['name'], region['offset'], region['size']
        if name in seen or start < program_bytes or start % 8 or size <= 0 or size % 8:
            raise ValueError('duplicate, unaligned or program-overlapping allocation')
        seen.add(name)
        if start + size > capacity or not 0<region['logical_bytes']<=size:
            raise ValueError('allocation exceeds physical SRAM')
        if region['first'] > region['last']:
            raise ValueError('reversed lifetime')
        high = max(high, start + size)
    for i, a in enumerate(layout):
        for b in layout[i+1:]:
            live = not (a['last'] < b['first'] or b['last'] < a['first'])
            overlap = a['offset'] < b['offset']+b['size'] and b['offset'] < a['offset']+a['size']
            if live and overlap:
                raise ValueError(f'live overlap: {a["name"]}, {b["name"]}')
    epochs = {r['first'] for r in layout} | {r['last'] for r in layout}
    peak_live = max((sum(r['size'] for r in layout if r['first'] <= t <= r['last'])
                     for t in epochs), default=0)
    return dict(logical_bytes=sum(r['logical_bytes'] for r in layout),
                allocated_high_watermark=high, peak_live_bytes=peak_live,
                physical_capacity_bytes=capacity, reserved_program_bytes=program_bytes)
