import copy
import unittest
from .momentum_analysis import SEEDS,BATCHES,METHODS,BETAS,LRS,TOTAL,analyse_base,analyse_controls


def family():
    rows=[]
    for m in METHODS:
        for b in BATCHES:
            for beta in BETAS:
                for lr in LRS:
                    for s in SEEDS:
                        loss=5+(0 if lr==.01 else .02)-(0 if beta==.9 else (-.01 if b==BATCHES[0] else .03))
                        rows.append(dict(method=m,batch_tokens=b,momentum=beta,lr=lr,seed=s,total_tokens=TOTAL,status='complete',
                            full_development_nll=loss,evaluations=[dict(step=0,validation_nll=9.),dict(step=100,validation_nll=loss)]))
    return rows


def controls(base,choices):
    output=[]
    for kind in ['matched','extended']:
        for m in METHODS:
            for b in BATCHES if kind=='matched' else (BATCHES[-1],):
                for beta in BETAS:
                    for s in SEEDS:
                        lr=choices[f'{m}:{b}']['selected_lr']
                        row=next(r for r in base if (r['method'],r['batch_tokens'],r['momentum'],r['lr'],r['seed'])==(m,b,beta,lr,s))
                        output.append(dict(row,control_kind=kind,total_tokens=128*b if kind=='matched' else 2*TOTAL))
    return output


class MomentumTests(unittest.TestCase):
    def test_balanced_positive_interaction(self):
        r=analyse_base(family())
        self.assertTrue(r['both_methods_pass'])
        self.assertTrue(all(x['tested_preference_reversal'] for x in r['contrasts']))
        self.assertFalse(r['full_goal_qualified'])
        self.assertTrue(r['batch_rate_growth_gate_still_failed'])

    def test_missing_and_duplicate_family(self):
        rows=family()
        with self.assertRaises(ValueError):analyse_base(rows[:-1])
        with self.assertRaises(ValueError):analyse_base(rows+[rows[0]])

    def test_one_bad_seed_cannot_average_away(self):
        rows=family()
        for r in rows:
            if r['method']=='spd' and r['seed']==SEEDS[1] and r['batch_tokens']==BATCHES[-1] and r['momentum']==.8:
                r['full_development_nll']+=.04
        result=analyse_base(rows)
        self.assertTrue(result['primary_by_method']['muon']['passed'])
        self.assertFalse(result['primary_by_method']['spd']['passed'])

    def test_one_bad_lr_cannot_be_selected_away(self):
        rows=family()
        for r in rows:
            if r['lr']==.02 and r['batch_tokens']==BATCHES[-1] and r['momentum']==.8:r['full_development_nll']+=.04
        r=analyse_base(rows)
        self.assertFalse(r['both_methods_pass'])
        self.assertTrue(all(x['large_gain']>0 for x in r['secondary_contrasts']))

    def test_materiality_is_per_seed_across_both_rates(self):
        rows=family()
        for r in rows:
            if r['seed']==SEEDS[1] and r['batch_tokens']==BATCHES[-1] and r['momentum']==.8:r['full_development_nll']+=.028
        result=analyse_base(rows)
        self.assertTrue(all(x['all_fixed_rate_signs_pass'] for x in result['primary_by_method'].values()))
        self.assertFalse(result['both_methods_pass'])

    def test_instability_cannot_select_bad_rate(self):
        rows=family();rows[0].update(status='numerical_instability',full_development_nll=None,evaluations=[])
        r=analyse_base(rows)
        self.assertFalse(r['primary_by_method']['muon']['passed'])
        self.assertEqual(r['control_rate_choices'][f'muon:{BATCHES[0]}']['selected_lr'],.02)

    def test_control_rate_pools_both_betas_and_seeds(self):
        rows=family()
        for r in rows:
            if r['lr']==.01:r['full_development_nll']=4 if r['momentum']==.8 else 6
            else:r['full_development_nll']=4.8
        result=analyse_base(rows)
        self.assertTrue(all(x['selected_lr']==.02 for x in result['control_rate_choices'].values()))
        self.assertEqual(result['secondary_lr_choices'][f'muon:{BATCHES[0]}:0.8']['selected_lr'],.01)

    def test_control_ties_choose_smaller_lr(self):
        rows=family()
        for r in rows:r['full_development_nll']=5
        self.assertTrue(all(x['selected_lr']==.01 for x in analyse_base(rows)['control_rate_choices'].values()))

    def test_controls_complete_and_same_fixed_rate(self):
        rows=family();choices=analyse_base(rows)['control_rate_choices'];c=controls(rows,choices)
        report=analyse_controls(rows,c,choices)
        self.assertEqual(len(report['records']),4)
        self.assertTrue(all(abs(x['retained_fraction']-1)<1e-12 for x in report['records']))
        with self.assertRaises(ValueError):analyse_controls(rows,c[:-1],choices)
        c[0]['lr']=.02
        with self.assertRaises(ValueError):analyse_controls(rows,c,choices)

    def test_controls_do_not_rescue_base_failure(self):
        rows=family()
        for r in rows:
            if r['momentum']==.8:r['full_development_nll']+=.1
        base=analyse_base(rows);c=controls(rows,base['control_rate_choices'])
        for r in c:
            if r['momentum']==.8:r['full_development_nll']-=.5
        result=analyse_controls(rows,c,base['control_rate_choices'])
        self.assertFalse(base['both_methods_pass'])
        self.assertFalse(result['full_goal_qualified'])
        self.assertTrue(all(x['extended_large_gain']>0 for x in result['records']))


if __name__=='__main__':unittest.main()
