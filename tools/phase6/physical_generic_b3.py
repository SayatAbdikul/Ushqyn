#!/usr/bin/env python3
"""Measure the already native-validated generic B3 incumbent on the fixed FPGA.

The generic compiler's schedule metadata omits command_count, which the older
physical runner requires. Only a copied schedule JSON gains this derived field;
the commands, payload, inputs, output and oracle bytes must stay identical.
The copied fixture is rerun on the frozen native RTL executable at both seeds.
"""
import argparse
import json
from pathlib import Path
import shutil

import matched_campaign as campaign
from matched_defines_baseline import native_measure

ROOT=campaign.ROOT
BASE=ROOT/'work/phase6/generic-b3-board-v1'
B3=ROOT/'work/phase6/matched-baselines-v1/b3-final/report.json'
B1B2=ROOT/'work/phase6/matched-baselines-v1/b1b2-final-v1/report.json'
CURRENT=ROOT/'work/phase6/matched-baselines-v1/current-v4/report.json'


def adapt(row, model, sample, folder):
    source=campaign.local(row['directory'])
    campaign.verify(source,row['files'])
    folder.mkdir(parents=True,exist_ok=False)
    for name in row['files']:
        shutil.copyfile(source/name,folder/name)
    schedule=campaign.read(folder/'schedule.json')
    commands=(folder/'commands.bin').read_bytes()
    if (len(commands)%16 or schedule['program_sha256']!=campaign.sha(folder/'commands.bin')
            or schedule['image_sha256']!=campaign.sha(folder/'payload.bin')
            or schedule['final_output']['bytes']!=(folder/'output.bin').stat().st_size):
        raise ValueError('generic candidate schedule/fixture mismatch')
    if 'command_count' in schedule and schedule['command_count']!=len(commands)//16:
        raise ValueError('generic command count mismatch')
    schedule['command_count']=len(commands)//16
    schedule.setdefault('snapshot_regions',{})
    campaign.save(folder/'schedule.json',schedule)
    files={name:campaign.sha(folder/name) for name in row['files']}
    if any(files[name]!=row['files'][name] for name in row['files'] if name!='schedule.json'):
        raise ValueError('generic program or oracle bytes changed by adapter')
    native=[native_measure(folder,seed,folder.parent/f'{model}-{sample}-s{seed}.json')
            for seed in (0,6063)]
    if any(n['status']!='passed' or n['fixture_files']!=files or
           n['executable_sha256']!=campaign.NATIVE_SHA for n in native):
        raise ValueError('adapted generic fixture has no bound native pass')
    result=campaign.fixture(campaign.relative(folder),files,model,sample,native,row['replay'])
    result['source_fixture']=row['directory']
    result['source_files']=row['files']
    result['metadata_adapter']='added command_count and empty snapshot_regions; all execution and oracle bytes identical'
    return result


def prepare(output=BASE,repeats=3):
    output=Path(output).resolve()
    if (output/'plan.json').is_file():
        plan=campaign.read(output/'plan.json')
        if plan['repeats']!=repeats or plan['planned']!=sum(len(groups) for groups in
                plan['measurement_groups'].values())*(1+2*repeats):
            raise ValueError('existing physical plan differs from requested repeats')
        campaign.audit_plan(plan)
        return plan
    b3=campaign.read(B3)
    if b3['status']!='passed' or b3['native_sha256']!=campaign.NATIVE_SHA:
        raise ValueError('generic B3 result incomplete')
    campaign.verify(ROOT,b3['compiler_sources'])
    plan=campaign.prepare(B1B2,None,repeats,CURRENT)
    if any(b3['models'][m]['graph_sha256']!=plan['common_graph_sha256'][m]
           for m in ('kws','vww')):
        raise ValueError('generic B3 does not use the common exact graph')
    plan['policies']={'current':plan['policies']['current']}
    plan['measurement_groups']={'kws':{'current':['current','B3-adaptation']},
                                'vww':{'current':['current'],'B3-adaptation':['B3-adaptation']}}
    b3_policy={}
    for model in ('kws','vww'):
        b3_policy[model]={}
        for sample in ('pinned','stress'):
            row=b3['selected'][model][sample]
            current=plan['policies']['current'][model][sample]
            if model=='kws':
                if any(row['files'][name]!=current['files'][name] for name in campaign.REQUIRED):
                    raise ValueError('KWS B3/current alias bytes differ')
                b3_policy[model][sample]=current
            else:
                b3_policy[model][sample]=adapt(row,model,sample,
                    output/'fixtures'/f'{model}-{sample}')
    plan['policies']['B3-adaptation']=b3_policy
    plan['planned']=sum(len(groups) for groups in plan['measurement_groups'].values())*(1+2*repeats)
    plan['b3_coverage']=b3['coverage']
    plan['source_sha256'].update(b3['compiler_sources'])
    plan['source_sha256'][campaign.relative(B3)]=campaign.sha(B3)
    plan['source_sha256'][campaign.relative(Path(__file__))]=campaign.sha(Path(__file__))
    plan['scope']='short generic B3 vs same-image current control; KWS is byte-identical alias, VWW independently measured'
    campaign.audit_plan(plan)
    campaign.save(output/'plan.json',plan)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=BASE)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--port',default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    plan=prepare(args.output,args.repeats)
    if args.run:
        result=campaign.run(plan,Path(args.output)/'physical',args.port)
        print(json.dumps(result['comparisons'],indent=2,sort_keys=True))
    else:
        print(json.dumps(dict(status='preflight-passed',planned=plan['planned'],
                              measurement_groups=plan['measurement_groups']),indent=2))


if __name__=='__main__':main()
