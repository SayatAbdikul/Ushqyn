#!/usr/bin/env python3
"""Audit frozen KWS/VWW operator geometries without claiming model execution."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/research/evidence/phase4/inventory-audit.json'
summary={}
for model in ('kws','vww'):
    path=ROOT/f'benchmarks/manifests/{model}.canonical-inventory.json'
    inventory=json.loads(path.read_text())
    tensors={t['name']:t for t in inventory['tensors']}
    kinds=Counter()
    unsupported=[]
    placement_pressure=[]
    weight_pressure=[]
    persistent_parameters=0
    for i,node in enumerate(inventory['operators']):
        op=node['op'];a=node['attributes']
        shape=tensors[node['inputs'][0]]['shape']
        output=tensors[node['outputs'][0]]['shape']
        if op=='Conv':
            w=tensors[node['inputs'][1]]['shape']
            kh,kw=w[2:]
            groups=a.get('group',1)
            if groups==1:
                kind='pointwise' if (kh,kw)==(1,1) else 'ordinary_conv'
            elif groups==shape[1]==output[1] and w[1]==1:
                kind='depthwise'
            else:
                kind='unsupported_group_conv'
            allowed=(kind!='unsupported_group_conv' and max(kh,kw)<=31 and
                     all(s in (1,2) for s in a.get('strides',[1,1])) and
                     a.get('dilations',[1,1])==[1,1] and
                     max(shape[2:])<=255 and max(shape[1],output[1])<=1024)
            reduction=math.prod(w[1:])
            packed_weight_bytes=w[0]*((reduction+7)//8*8)
            parameter_bytes=w[0]*16
            persistent_parameters+=packed_weight_bytes+parameter_bytes
            if packed_weight_bytes+parameter_bytes>32768:
                weight_pressure.append(dict(index=i,kind=kind,
                    packed_weight_bytes=packed_weight_bytes,
                    parameter_bytes=parameter_bytes))
        elif op in ('AveragePool','GlobalAveragePool'):
            kind='average_pool'
            kh,kw=a.get('kernel_shape',shape[2:])
            sh,sw=a.get('strides',[1,1])
            allowed=(max(kh,kw)<=31 and not any(a.get('pads',[0,0,0,0])) and
                     sh in (1,2,kh) and sw in (1,2,kw) and
                     (sh<=2 or shape[2]-kh<sh) and (sw<=2 or shape[3]-kw<sw) and
                     not a.get('count_include_pad',0))
        elif op in ('Relu','Clip','Gemm','MaxPool','Reshape','Flatten','Identity'):
            kind=op.lower();allowed=True
            if op=='Gemm':
                w=tensors[node['inputs'][1]]['shape']
                packed_weight_bytes=w[0]*((w[1]+7)//8*8)
                parameter_bytes=w[0]*16
                persistent_parameters+=packed_weight_bytes+parameter_bytes
                if packed_weight_bytes+parameter_bytes>32768:
                    weight_pressure.append(dict(index=i,kind=kind,
                        packed_weight_bytes=packed_weight_bytes,
                        parameter_bytes=parameter_bytes))
        elif op=='Transpose' and i==0 and model=='vww' and a.get('perm')==[0,3,1,2]:
            kind='declared_host_layout';allowed=True
        else:
            kind=f'unsupported_{op.lower()}';allowed=False
        kinds[kind]+=1
        if not allowed:unsupported.append(dict(index=i,op=op,name=node['name'],kind=kind))
        if op in ('Conv','AveragePool','GlobalAveragePool','MaxPool'):
            input_bytes=1
            output_bytes=1
            for dim in shape:input_bytes*=dim
            for dim in output:output_bytes*=dim
            if input_bytes+output_bytes>32768:
                placement_pressure.append(dict(index=i,kind=kind,
                    input_bytes=input_bytes,output_bytes=output_bytes,
                    live_pair_bytes=input_bytes+output_bytes))
    summary[model]=dict(inventory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        operators=len(inventory['operators']),classes=dict(kinds),
                        unsupported_geometry=unsupported,
                        layers_exceeding_sram_for_input_output_pair=placement_pressure,
                        layers_exceeding_sram_for_weights=weight_pressure,
                        all_persistent_weight_parameter_bytes=persistent_parameters)
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:dict(operators=v['operators'],classes=v['classes'],
                         unsupported=len(v['unsupported_geometry']),
                         placement_pressure=len(v['layers_exceeding_sram_for_input_output_pair']),
                         weight_pressure=len(v['layers_exceeding_sram_for_weights']),
                         all_persistent_weight_parameter_bytes=v['all_persistent_weight_parameter_bytes'])
                  for k,v in summary.items()},indent=2))
