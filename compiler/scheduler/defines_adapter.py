"""Geometry-only input adapter for pinned upstream DeFiNES.

Quantization barriers remain explicit nodes. Identity-like activation equations
describe dependence/shape ONLY; DeFiNES is not used as our arithmetic oracle.
Do not interpret these workloads or upstream ideal-array costs as executable
Tang Nano schedules. Halo packing, cached-row placement and kernel timing still
need a complete backend before a full B3 performance comparison is eligible.
"""
from .contract import require


def segment_workload(program,start,stop):
    require(0<=start<stop<=len(program.layers),'invalid segment')
    shape=program.tensors[program.layers[start].inputs[0]].shape
    require(len(shape)==4 and shape[0]==1,'requires batch-one NCHW')
    result={-1:{'equation':'input','loop_dim_size':{'B':1,'K':shape[1],'OY':shape[2],'OX':shape[3]},
        'precision':8,'core_allocation':1,'memory_operand_links':{'O':'I1'}}}
    previous=-1;channel_dim='K'
    for index in range(start,stop):
        l=program.layers[index];shape=program.tensors[l.output].shape
        require(l.op in ('Conv','Relu','Clip') and len(shape)==4,'unsupported spatial operator')
        row={'core_allocation':1,'operand_precision':{'O':32,'O_final':8,'I':8},
             'operand_source':{'I':[previous]},'memory_operand_links':{'O':'O','I':'I1'},
             'original_layer':index,'original_op':l.op,'quantization_barrier':True}
        if l.op=='Conv':
            w=l.parameters['weight'];kh,kw=w.shape[2:];sh,sw=l.attributes.get('strides',[1,1]);dh,dw=l.attributes.get('dilations',[1,1])
            depthwise=l.attributes.get('group',1)!=1
            require(not depthwise or l.attributes['group']==shape[1]==program.tensors[l.inputs[0]].shape[1],'unsupported grouped convolution')
            out_channel='G' if depthwise else 'K';in_channel='G' if depthwise else 'C'
            row.update(equation=('O[b][g][oy][ox]+=W[g][fy][fx]*I[b][g][iy][ix]' if depthwise else
                                 'O[b][k][oy][ox]+=W[k][c][fy][fx]*I[b][c][iy][ix]'),
                       equation_relations=[f'ix={sw}*ox+{dw}*fx',f'iy={sh}*oy+{dh}*fy'],
                       loop_dim_size={'B':1,out_channel:shape[1],'OY':shape[2],'OX':shape[3],'FY':kh,'FX':kw},
                       constant_operands=['W'],spatial_mapping={'D1':('FX',min(8,kw))})
            if not depthwise: row['loop_dim_size']['C']=w.shape[1]
            row['operand_source']['W']=[];row['operand_precision']['W']=8;row['memory_operand_links']['W']='I2'
            row['operand_source_dimension_mapping']={'I':{'IX':'OX','IY':'OY',in_channel:channel_dim}}
        else:
            out_channel='G'
            row.update(equation='O[b][g][oy][ox]=I[b][g][oy][ox]',
                       loop_dim_size={'B':1,'G':shape[1],'OY':shape[2],'OX':shape[3]},
                       constant_operands=[],spatial_mapping={'D1':('OX',min(8,shape[3]))},
                       operand_source_dimension_mapping={'I':{'OX':'OX','OY':'OY','G':channel_dim}})
        result[index]=row;previous=index;channel_dim=out_channel
    return result
