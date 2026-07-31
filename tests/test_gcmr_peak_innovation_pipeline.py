import unittest
import numpy as np
from gnss_doppler_lab.gcmr_peak_innovation_pipeline import EventRecord, GCMRPeakInnovationPipeline


def event(t, prns=('G01','G02','G03'), bump=0.):
    # W prior samples have a distinctive trend; epl is the subsequent epoch.
    histories={p: np.array([[.8+.01*i, 10+i, 1.2+.01*i] for i in range(3)],float) for p in prns}
    epl=np.array([[.83,13.,1.23] for _ in prns],float)+bump
    cn0=np.array([38.+i for i in range(len(prns))]); el=np.array([25.+i for i in range(len(prns))])
    pairs={(a,b):np.array([.1*(i+j+1),min(el[i],el[j]),max(el[i],el[j])])
           for i,a in enumerate(prns) for j,b in enumerate(prns) if i<j}
    return EventRecord(t,tuple(prns),epl,histories,cn0,el,pairs)

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.p=GCMRPeakInnovationPipeline(3,hidden_size=4,epochs=3,seed=7)
        self.train=[event(i) for i in range(4)]
        self.valid=[event(10+i, bump=.001*i) for i in range(4)]
    def test_attack_cannot_fit_and_outputs_named_rows(self):
        with self.assertRaises(RuntimeError): self.p.score_attack(event(99))
        self.p.fit_normal(self.train,self.valid)
        before=self.p.network.gru.weight_ih_l0.detach().clone()
        out=self.p.score_attack(event(99,('G09','G17','G31','G32'),bump=.2), destruction_seed=3)
        self.assertEqual(out.n,4); self.assertEqual(set(out.scores),{'A0','A1','A2','A3','A4','Full'})
        self.assertTrue(np.isfinite(list(out.scores.values())).all()); self.assertTrue(np.isfinite(out.destroyed_pair_score))
        self.assertTrue((before==self.p.network.gru.weight_ih_l0.detach()).all())
    def test_causal_history_and_no_identity_input(self):
        self.p.fit_normal(self.train,self.valid)
        a=event(1); b=event(1,bump=99.)
        # Prediction sees only normalized history, not current target/future value.
        ra=self.p._predict_residual(a); rb=self.p._predict_residual(b)
        self.assertTrue(np.allclose((rb-ra), b.epl[:,[0,1,2]]/b.epl[:,1,None]-a.epl/a.epl[:,1,None]))
        self.assertEqual(self.p.network.gru.input_size,3)
        self.assertFalse(any('G01' in str(x) for x in self.p.network.state_dict().values()))
    def test_missing_pair_and_wrong_history_are_rejected(self):
        e=event(1); bad=EventRecord(e.time,e.prns,e.epl,{**e.histories,'G01':np.ones((2,3))},e.cn0,e.elevation,e.pair_conditions)
        with self.assertRaises(ValueError): self.p._arrays(bad)
        with self.assertRaises(ValueError): self.p._arrays(EventRecord(e.time,e.prns,e.epl,e.histories,e.cn0,e.elevation,{}))

if __name__ == '__main__': unittest.main()
