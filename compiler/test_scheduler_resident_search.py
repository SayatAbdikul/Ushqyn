"""Check search quality against independently enumerated executable schedules."""
import itertools
import pytest
from scheduler.fixtures import wide_chain
from scheduler.resident import compile_resident, eligible
from scheduler.resident_search import optimize_resident
from scheduler.resident_verify import replay_resident
from scheduler.event_cost import estimate
from phase4_compile import compile_tiled


@pytest.mark.parametrize('spatial_pw',[False,True])
def test_search_bound_and_result_against_complete_small_catalogue(spatial_pw):
    p,x=wide_chain();plans={h:compile_tiled(p,prefer_half=h) for h in (False,True)}
    heads=[i for i,l in enumerate(plans[False][0]['layers']) if l['tiles']]
    pairs=eligible(p);scores=[]
    for values in itertools.product((False,True),repeat=len(heads)+len(pairs)+1):
        choices=dict(zip(heads,values));fused=[i for i,on in zip(pairs,values[len(heads):]) if on]
        a=compile_resident(p,fused=fused,overlap=values[-1],tile_choices=choices,prepared_plans=plans)
        scores.append(estimate(a[0],a[1],spatial_pw=spatial_pw)['elapsed_cycles'])
    a,r=optimize_resident(p,None,spatial_pw=spatial_pw)
    s=a[2]
    assert replay_resident(p,a[0],a[1],{'x':x},run_contracts=s['run_contracts'],final_output=s['final_output'])['status']=='passed'
    assert r['lower_bound']<=min(scores)<=r['objective']
    assert r['objective']==min(scores)
    assert r['objective']<=min(b['predicted_cycles'] for b in r['baselines'].values())
    _,again=optimize_resident(p,None,spatial_pw=spatial_pw)
    assert r['selected']==again['selected'] and r['history']==again['history']


def test_budget_keeps_verified_fallback():
    p,x=wide_chain();a,r=optimize_resident(p,None,max_evaluations=8)
    assert r['evaluations']<=8
    assert r['objective']<=r['baselines']['half-materialize-overlap']['predicted_cycles']
    s=a[2]
    replay_resident(p,a[0],a[1],{'x':x},run_contracts=s['run_contracts'],final_output=s['final_output'])
