import unittest
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .report_contract import classify_outcome, spatial_tracking, paired_route_bootstrap, project_saved_trace
from .metrics import project_trajectory_to_polyline
from .prepublication_analysis import condition_key, boundaries
from .pace_intervention import replace_pace_scalar


class ReportingTest(unittest.TestCase):
    def row(self, **kw):
        return dict(task_success=True, task_arrived=True, survived=False,
                    task_success_time_s=4., time_to_failure_s=5.,
                    active_cross_track_rmse_m=.05, **kw)

    def test_afterarrival_is_not_prearrival(self):
        r = classify_outcome(self.row())
        self.assertTrue(r['active_survival'])
        self.assertTrue(r['postarrival_physical_failure'])
        self.assertFalse(r['full_horizon_survival'])
        self.assertTrue(r['adherence_success'])

    def test_invalid_success_rejected(self):
        r = self.row(); r['time_to_failure_s'] = 3.
        with self.assertRaises(ValueError): classify_outcome(r)

    def test_nonarrival_failure_and_missing_cte(self):
        r = self.row(); r.update(task_arrived=False, task_success=False, task_success_time_s='nan')
        self.assertFalse(classify_outcome(r)['active_survival'])
        r = self.row(); r['active_cross_track_rmse_m'] = 'nan'
        self.assertFalse(classify_outcome(r)['adherence_success'])

    def test_initialization_failure_retained_as_failure(self):
        r=self.row();r.update(task_arrived=False,task_success=False,task_success_time_s='nan',
                             initialization_failure=True,time_to_failure_s=0.)
        self.assertTrue(classify_outcome(r)['prearrival_physical_failure'])
        self.assertFalse(classify_outcome(r)['adherence_success'])

    def test_weighting_and_margin(self):
        r = spatial_tracking([.1,.2,.3,.4], [.1]*4, [.2]*4, [.1]*4, [.2], margin_m=.1)
        self.assertAlmostEqual(r['spatial_cte_rmse_m'], .1)
        self.assertAlmostEqual(r['spatial_height_mae_m'], .1)
        self.assertLess(r['height_scored_fraction'], 1.)
        zero = spatial_tracking([0,0], [1,1], [1,1], [0,0], [])
        self.assertTrue(np.isnan(zero['spatial_cte_rmse_m']))

    def test_backward_motion_not_double_counted(self):
        r = spatial_tracking([.2,.1,.2,.3], [0,1,1,0], [0]*4, [0]*4, [])
        self.assertAlmostEqual(r['travelled_progress_m'], .3)
        self.assertAlmostEqual(r['spatial_cte_rmse_m'], 0.)

    def test_cluster_bootstrap(self):
        a = paired_route_bootstrap({'a':np.ones(3), 'b':np.ones(2)})
        self.assertEqual(a['independent_routes'], 5)
        self.assertEqual(a['mean'], 1.)
        self.assertAlmostEqual(a['ci_low'], 1.)

    def test_scalar_only_is_exact_and_nonmutating(self):
        g = torch.arange(64.).reshape(4,16); original = g.clone()
        result = replace_pace_scalar(g, [1,2,3,4])
        torch.testing.assert_close(g, original, rtol=0, atol=0)
        torch.testing.assert_close(result[:,:15], g[:,:15], rtol=0, atol=0)
        torch.testing.assert_close(result[:,-1], torch.tensor([1.,2.,3.,4.]))
        with self.assertRaises(ValueError): replace_pace_scalar(g, [1])

    def test_fast_projection_preserves_cursor_semantics(self):
        rng=np.random.default_rng(42)
        path=np.cumsum(rng.normal(size=(120,2)),axis=0)
        xy=path[np.linspace(0,119,300).astype(int)]+rng.normal(scale=.1,size=(300,2))
        fast=project_saved_trace(xy,path)
        slow=project_trajectory_to_polyline(xy,path)
        for a,b in zip(fast,slow): np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)

    def test_pair_effective_profile_not_unused_default(self):
        a=dict(path_shape='a',repeat='1',transition_fraction='.2',requested_height='.28')
        b=dict(a,requested_height='.23185')
        self.assertEqual(condition_key(a),condition_key(b))
        a['transition_fraction']=b['transition_fraction']='nan'
        self.assertNotEqual(condition_key(a),condition_key(b))

    def test_boundaries_follow_materialized_height_changes(self):
        s=dict(path_cumulative_m=[0,.8,1.6,2.4,3.2,4.],height_segment_m=.8,
               height_schedule_m=[.17,.17,.29,.29,.17])
        np.testing.assert_allclose(boundaries(s),[1.6,3.2])

    def test_evaluator_scalar_only_preserves_actual_constructed_goals(self):
        # Compile only the pure goal function: importing evaluate_policy would
        # launch Kit. This checks integration with the real geometry builders.
        from scripts.diffusion_policy.train.conditioning import goal_builder as builders
        import scripts.diffusion_policy.evaluation as package
        sys.modules.setdefault('evaluation',package)
        source=Path(__file__).resolve().parents[1]/'evaluate_policy.py'
        tree=ast.parse(source.read_text(encoding='utf-8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_vectorized_goals')
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
        ast.fix_missing_locations(module)
        context={k:getattr(builders,k) for k in dir(builders) if not k.startswith('__')}
        context.update(np=np,torch=torch)
        exec(compile(module,str(source),'exec'),context)
        func=context['compute_vectorized_goals']
        path=np.column_stack((np.linspace(0,4,101),np.zeros(101),np.full(101,.2932))).astype(np.float32)
        plan=SimpleNamespace(path_w=path,cumulative_lengths=np.linspace(0,4,101),yaws_w=np.zeros(101),time_indexed=False)
        env=SimpleNamespace(_robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[.3,0,.29]]),root_quat_w=torch.tensor([[1.,0,0,0]]))))
        arguments=dict(reference_step=50,speed_budget_start_arc=np.array([0.]),speed_budget_target_arc=np.array([4.]),speed_budget_max_mps=1.5)
        for coherent in (False,True):
            normal=func(env,[plan],np.array([.65]),np.array([0]),100,(.5,1,1.5),.02,2.,torch.device('cpu'),'hindsight_geom_profile16',pace_consistent_preview=coherent,**arguments)
            replaced=func(env,[plan],np.array([.65]),np.array([0]),100,(.5,1,1.5),.02,2.,torch.device('cpu'),'hindsight_geom_profile16',pace_consistent_preview=coherent,scalar_speed_override=np.array([.35]),**arguments)
            torch.testing.assert_close(normal[:,:15],replaced[:,:15],rtol=0,atol=0)
            self.assertAlmostEqual(float(replaced[0,-1]),.35,places=6)


if __name__ == '__main__': unittest.main()
