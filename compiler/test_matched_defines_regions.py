"""Independent bytearray execution of emitted DMA/COPY rectangle programs."""
import random
import unittest

from scheduler.matched_defines_regions import (SRAM_BYTES, align8,
    rectangle_temp_bytes, emit_read_rectangle, emit_write_rectangle,
    emit_copy_rectangle)


class MemoryMachine:
    def __init__(self, seed):
        rng = random.Random(seed)
        self.sram = bytearray(rng.randbytes(SRAM_BYTES))
        self.external = bytearray(rng.randbytes(65536))
        self.commands = []

    def dma(self, direction, ext, sram, size, role):
        assert direction in ('to_sram','from_sram')
        assert ext >= 0 and sram >= 0 and ext % 8 == sram % 8 == 0
        assert 0 < size <= 32768 and align8(ext+size) <= len(self.external)
        assert align8(sram+size) <= SRAM_BYTES
        self.commands.append(('dma',direction,ext,sram,size,role))
        if direction == 'to_sram': self.sram[sram:sram+size] = self.external[ext:ext+size]
        else: self.external[ext:ext+size] = self.sram[sram:sram+size]

    def copy(self, source, destination, size, role):
        assert source >= 0 and source % 8 == 0 and destination >= 0
        assert 0 < size <= 32768 and align8(source+size) <= SRAM_BYTES
        assert destination+size <= SRAM_BYTES
        self.commands.append(('copy',source,destination,size,role))
        self.sram[destination:destination+size] = bytes(self.sram[source:source+size])


def selected_bytes(data, base, shape, rectangle):
    channels, height, width = shape
    y0,x0,y1,x1 = rectangle
    return bytes(data[base+((c*height+y)*width+x)]
                 for c in range(channels) for y in range(y0,y1) for x in range(x0,x1))


def ignore_temporary(actual, expected, base, size):
    expected[base:base+size] = actual[base:base+size]
    return expected


class RectangleTests(unittest.TestCase):
    def test_randomized_reads_preserve_destination_guard_and_source(self):
        rng = random.Random(67102)
        for case in range(250):
            shape = (rng.randrange(1,4),rng.randrange(1,12),rng.randrange(1,256))
            y0,x0 = rng.randrange(shape[1]),rng.randrange(shape[2])
            rectangle = (y0,x0,rng.randrange(y0+1,shape[1]+1),rng.randrange(x0+1,shape[2]+1))
            machine = MemoryMachine(case)
            before_ext = bytes(machine.external)
            expected = bytearray(machine.sram)
            block = selected_bytes(before_ext,64,shape,rectangle)
            expected[256:256+len(block)] = block
            result = emit_read_rectangle(64,shape,rectangle,256,32000,machine.dma,machine.copy)
            self.assertEqual(machine.external,before_ext)
            self.assertEqual(machine.sram,ignore_temporary(machine.sram,expected,32000,result['temporary_bytes']))
            self.assertEqual(result['logical_bytes'],len(block))

    def test_randomized_writes_preserve_every_external_neighbor(self):
        rng = random.Random(67103)
        for case in range(250):
            shape = (rng.randrange(1,4),rng.randrange(1,12),rng.randrange(1,256))
            y0,x0 = rng.randrange(shape[1]),rng.randrange(shape[2])
            y1,x1 = rng.randrange(y0+1,shape[1]+1),rng.randrange(x0+1,shape[2]+1)
            rectangle = (y0,x0,y1,x1)
            machine = MemoryMachine(case+1000)
            before_sram = bytearray(machine.sram)
            expected = bytearray(machine.external)
            packed_index = 256
            for c in range(shape[0]):
                for y in range(y0,y1):
                    for x in range(x0,x1):
                        expected[64+(c*shape[1]+y)*shape[2]+x] = before_sram[packed_index]
                        packed_index += 1
            result = emit_write_rectangle(64,shape,rectangle,256,32000,machine.dma,machine.copy)
            self.assertEqual(machine.external,expected)
            self.assertEqual(machine.sram,ignore_temporary(machine.sram,before_sram,32000,result['temporary_bytes']))

    def test_randomized_sram_rectangles_preserve_surrounding_bytes(self):
        rng = random.Random(67104)
        for case in range(300):
            shape = (rng.randrange(1,4),rng.randrange(1,12),rng.randrange(1,256))
            y0,x0 = rng.randrange(shape[1]),rng.randrange(shape[2])
            y1,x1 = rng.randrange(y0+1,shape[1]+1),rng.randrange(x0+1,shape[2]+1)
            rh,rw = y1-y0,x1-x0
            destination_shape = (shape[0],rh+rng.randrange(4),rw+rng.randrange(8))
            dy,dx = rng.randrange(destination_shape[1]-rh+1),rng.randrange(destination_shape[2]-rw+1)
            machine = MemoryMachine(case+2000)
            expected = bytearray(machine.sram)
            before_external = bytes(machine.external)
            block = iter(selected_bytes(machine.sram,256,shape,(y0,x0,y1,x1)))
            for c in range(shape[0]):
                for y in range(dy,dy+rh):
                    for x in range(dx,dx+rw):
                        expected[16384+(c*destination_shape[1]+y)*destination_shape[2]+x] = next(block)
            emit_copy_rectangle(256,shape,(y0,x0,y1,x1),16384,destination_shape,
                                (dy,dx),32000,machine.copy)
            self.assertEqual(machine.external,before_external)
            self.assertEqual(machine.sram,ignore_temporary(machine.sram,expected,32000,16))

    def test_all_source_destination_alignments_and_single_byte_edges(self):
        # A source-prefix of seven and destination-prefix of zero is the
        # troublesome case that needs a separate guard before row staging.
        for source_prefix in range(8):
            for destination_prefix in range(8):
                machine = MemoryMachine(source_prefix*8+destination_prefix)
                before = bytearray(machine.sram)
                source, target = 256+source_prefix,16384+destination_prefix
                emit_copy_rectangle(256,(1,1,16),(0,source_prefix,1,source_prefix+1),
                    16384,(1,1,16),(0,destination_prefix),32000,machine.copy)
                before[target] = before[source]
                self.assertEqual(machine.sram,ignore_temporary(machine.sram,before,32000,16))

    def test_invalid_geometry_and_overlapping_storage_rejected_before_emission(self):
        bad = [
            lambda f:emit_read_rectangle(1,(1,2,3),(0,0,2,3),256,32000,f,f),
            lambda f:emit_read_rectangle(0,(1,2,3),(-1,0,2,3),256,32000,f,f),
            lambda f:emit_read_rectangle(0,(1,2,3),(0,0,2,3),256,248,f,f),
            lambda f:emit_write_rectangle(0,(1,2,3),(0,0,2,3),0,32000,f,f),
            lambda f:emit_copy_rectangle(256,(1,2,3),(0,0,2,3),256,(1,2,3),(0,0),32000,f),
            lambda f:emit_copy_rectangle(256,(1,2,3),(0,0,2,3),512,(2,2,3),(0,0),32000,f),
            lambda f:emit_copy_rectangle(256,(1,2,3),(0,0,2,3),512,(1,2,3),(1,0),32000,f),
        ]
        for call in bad:
            emitted=[]
            with self.assertRaises(ValueError):call(lambda *args:emitted.append(args))
            self.assertEqual(emitted,[])

    def test_temporary_formula_covers_worst_unaligned_envelope(self):
        for width in range(1,256):
            size=rectangle_temp_bytes(width)
            self.assertEqual(size % 8,0)
            self.assertGreaterEqual(size-24,align8(7+width))

    def test_contiguous_tensor_and_aligned_strips_use_direct_commands(self):
        for operation in (emit_read_rectangle,emit_write_rectangle):
            machine=MemoryMachine(901)
            operation(64,(2,3,5),(0,0,3,5),256,32000,machine.dma,machine.copy)
            self.assertEqual(len(machine.commands),1)
            self.assertEqual(machine.commands[0][0],'dma')
            self.assertEqual(machine.commands[0][4],30)
            machine=MemoryMachine(902)
            operation(64,(2,8,16),(1,0,5,16),256,32000,machine.dma,machine.copy)
            self.assertEqual(len(machine.commands),2)
            self.assertTrue(all(c[0]=='dma' and c[4]==64 for c in machine.commands))
        machine=MemoryMachine(903)
        emit_copy_rectangle(256,(2,3,5),(0,0,3,5),16384,(2,3,5),
                            (0,0),32000,machine.copy)
        self.assertEqual(len(machine.commands),1)
        self.assertEqual(machine.commands[0][3],30)
        machine=MemoryMachine(904)
        emit_copy_rectangle(256,(2,8,16),(1,0,5,16),16384,(2,8,16),
                            (3,0),32000,machine.copy)
        self.assertEqual(len(machine.commands),2)
        self.assertTrue(all(c[0]=='copy' and c[3]==64 for c in machine.commands))


if __name__ == '__main__':
    unittest.main()
