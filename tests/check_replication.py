"""Compare full-run aggregate output and reconstruct counts from per-window records."""
import sys
import numpy as np
import pandas as pd

def check(reference, actual, paths):
    a, b = (pd.read_csv(p) for p in (reference, actual))
    pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=0,atol=5e-7)
    p=pd.read_csv(paths)
    assert len(p)==30000 and len(a)==9
    assert not p.duplicated(["tp","sl","window"]).any()
    for r in a.itertuples():
        z=p[(p.tp==r.tp)&(p.sl==r.sl)]
        e=z["count"]>=r.m
        va=e&z.valid_A; ve=e&z.valid_E; j=va&ve
        ra=z.reject_A; re=z.reject_E
        actual_counts=[len(z),e.sum(),va.sum(),ve.sum(),j.sum(),(va&ra).sum(),(ve&re).sum(),(j&ra&~re).sum(),(j&~ra&re).sum()]
        expected=[r.total,r.eligible,r.valid_A,r.valid_E,r.joint_valid,r.reject_A,r.reject_E,r.A_only,r.E_only]
        assert actual_counts==expected,(r,actual_counts)
        for g in ("A","E"):
            count=getattr(r,"reject_"+g); valid=getattr(r,"valid_"+g)
            assert np.isclose(count/r.total,getattr(r,g+"_overall"))
            if valid:
                assert np.isclose(count/valid,getattr(r,g+"_conditional"))
        if r.joint_valid:
            assert np.isclose((r.A_only-r.E_only)/r.joint_valid,r.delta)
    print("Nine aggregate contrasts and 30,000 per-window records verified.")

if __name__=="__main__":
    check(*sys.argv[1:])
