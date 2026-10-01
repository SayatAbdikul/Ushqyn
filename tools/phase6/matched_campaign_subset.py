#!/usr/bin/env python3
"""Run B3 and current controls after the independently completed B1/B2 screen.

The full preflight is retained. This wrapper changes only which already
validated policies are measured; hardware, warmups and timing are unchanged.
"""
import argparse
import json
from pathlib import Path

import matched_campaign as campaign


def subset(plan, names):
    names=set(names)
    if not names<=set(plan['policies']) or 'current' not in names:
        raise ValueError('select existing policies and retain the current control')
    result=dict(plan)
    result['policies']={k:v for k,v in plan['policies'].items() if k in names}
    groups={}
    for model,old in plan['measurement_groups'].items():
        groups[model]={}
        for members in old.values():
            selected=[name for name in members if name in names]
            if selected:groups[model][selected[0]]=selected
    result['measurement_groups']=groups
    result['planned']=sum(len(rows) for rows in groups.values())*(1+2*plan['repeats'])
    result['source_sha256']=dict(plan['source_sha256'])
    result['source_sha256'][campaign.relative(Path(__file__))]=campaign.sha(Path(__file__))
    result['scope']='short matched B3 subset with repeated current controls; separate physical session from B1/B2 tuning'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b3',type=Path,required=True)
    parser.add_argument('--b1b2',type=Path,default=campaign.BASE/'b1b2-final-v1/report.json')
    parser.add_argument('--current',type=Path,default=campaign.BASE/'current-v4/report.json')
    parser.add_argument('--include',nargs='+')
    parser.add_argument('--output',type=Path,default=campaign.BASE/'physical-b3-v1')
    parser.add_argument('--port',default='/dev/cu.usbserial-20250303171')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    full=campaign.prepare(args.b1b2,args.b3,args.repeats,args.current)
    names=args.include or [n for n in full['policies'] if n=='current' or n.startswith('B3-adaptation')]
    plan=subset(full,names)
    campaign.audit_plan(plan)
    if args.run:
        result=campaign.run(plan,args.output,args.port)
        print(json.dumps(result['best_matched_baselines'],indent=2,sort_keys=True))
    else:
        print(json.dumps(dict(status='preflight-passed',planned=plan['planned'],
            policies=list(plan['policies']),measurement_groups=plan['measurement_groups']),indent=2,sort_keys=True))


if __name__=='__main__':main()
