import io
from functools import partial

import numpy as np

import networkx as nx
import yaml
from matplotlib import pyplot as plt
from ompl import base as ob
from ompl import geometric as og


# There are no constraints on states

def valid(si, state):
    return si.satisfiesBounds(state)


def _a(state):
    return (state[0], state[1])
# but on edges


class EstimatedMotionValidator(ob.MotionValidator):

    def __init__(self, si, map, th=0.9, theta=0, min_r=0.15, max_r=0.5):
        super(EstimatedMotionValidator, self).__init__(si)
        self.map = map
        self.th = th
        self.theta = theta
        self.min_r = min_r
        self.max_r = max_r
        # We keep track of the number of edges tested
        self.num_short = 0
        self.num_nt = 0
        self.num_t = 0

    def checkMotion(self, s1, s2, k=None):
        s1 = _a(s1)
        p, c = self.map.traversable((s1[0], s1[1], self.theta), _a(s2), frame='abs')
        if np.isfinite(c):
            if p > self.th:
                self.num_t += 1
                return True
            self.num_nt += 1
        else:
            self.num_short += 1
        if k is not None:
            k.first = s1
            k.second = 0
        return False

    def __repr__(self):
        return (f'Motion Validator: {self.num_short} too short, {self.num_nt} not traversable, {self.num_t} traversable')

# One possible objective is the minimization of (estimated) duration


class DurationObjective(ob.MechanicalWorkOptimizationObjective):

    def __init__(self, si, map, theta):
        super(DurationObjective, self).__init__(si)
        self.map = map
        self.theta = theta

    # No cost remaining on a state
    def stateCost(self, s):
        return ob.Cost(0)

    # but there is a cost while moving, given by the estimator predicted duration
    def motionCost(self, s1, s2):
        s1 = _a(s1)
        _, t = self.map.traversable((s1[0], s1[1], self.theta), _a(s2), frame='abs')
        return ob.Cost(float(t))


def getBalancedObjective1(si, m, theta=0, k=1000.0):
    o1 = ob.PathLengthOptimizationObjective(si)
    o2 = SurvivalObjective(si, m, theta)
    opt = ob.MultiOptimizationObjective(si)
    opt.addObjective(o1, 1.0)
    opt.addObjective(o2, k)
    return opt


class SurvivalObjective(ob.MechanicalWorkOptimizationObjective):

    def __init__(self, si, map, theta):
        super(SurvivalObjective, self).__init__(si)
        self.map = map
        self.theta = theta

    # No cost remaining on a state
    def stateCost(self, s):
        return ob.Cost(0)

    # but there is a cost while moving, given by the estimator predicted duration
    def motionCost(self, s1, s2):
        p, t = self.map.traversable((s1[0], s1[1], self.theta), _a(s2), frame='abs')
        if p > 0:
            r = -np.log(p)
        else:
            r = np.inf
        return ob.Cost(float(r))


class Plan:

    def __init__(self, map, pose, target, max_time, objective='duration', planner=og.RRTstar,
                 threshold=0.9, tolerance=0.3, planner_options={'range': 0.45}):
        space = ob.RealVectorStateSpace(2)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0)
        bounds.setHigh(map.size)
        space.setBounds(bounds)
        s = ob.State(space)
        t = ob.State(space)

        self.theta = pose[-1]

        for arg, state in zip([pose[:2], target], [s, t]):
            for i, x in enumerate(arg):
                state[i] = x

        ss = og.SimpleSetup(space)
        ss.setStartAndGoalStates(s, t, tolerance)

        si = ss.getSpaceInformation()
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(partial(valid, si)))

        self.motion_val = EstimatedMotionValidator(si, map, threshold, theta=self.theta)
        si.setMotionValidator(self.motion_val)

        if objective == 'duration':
            self.do = DurationObjective(si, map, self.theta)
        elif objective == 'survival':
            self.do = SurvivalObjective(si, map, self.theta)
        elif objective == 'balanced':
            self.do = getBalancedObjective1(si, map, self.theta)
        else:  # the objective is the euclidean path length
            self.do = ob.PathLengthOptimizationObjective(si)
        ss.setOptimizationObjective(self.do)

        # RRTstar BITstar BFMT RRTsharp
        p = planner(si)
        if 'range' in planner_options:
            try:
                p.setRange(planner_options.get('range'))
            except AttributeError:
                pass

        ss.setPlanner(p)
        ss.setup()

        self.ss = ss
        self.max_time = max_time
        self.map = map
        self.s = pose
        self.t = list(target) + [pose[-1]]
        self.planner_options = planner_options
        if planner == og.RRTstar:
            self.planner_name = 'RRT*'
        else:
            self.planner_name = ''
        self.threshold = threshold
        self.tolerance = tolerance
        self.comp_duration = 0
        self.objective = objective

    def solve(self, dt=5, max_time=None):
        t = 0
        while t < (max_time or self.max_time):
            result = self.ss.solve(dt).getStatus()
            t += dt
            # print(t, result.name)
            self.comp_duration += dt
            if result == ob.PlannerStatus.EXACT_SOLUTION:
                break
        self.state = result.name
        return result.name

    @property
    def path(self):
        if self.ss.haveSolutionPath():
            solution = self.ss.getSolutionPath()
            return np.array([[x[0], x[1]] for x in solution.getStates()])

    @property
    def cost(self):
        if self.ss.haveSolutionPath():
            solution = self.ss.getSolutionPath()
            return solution.cost(self.ss.getOptimizationObjective()).value()

    # def plot(self, size=(10, 10), edge_color='orange', cmap=plt.cm.Greys_r):
    #     xy = self.path
    #     plt.figure(figsize=size)
    #     self.map.plot()
    #     plt.axis('equal')
    #     plt.axis('off')
    #     if xy is not None:
    #         plt.plot(*xy.T, 'b.-', alpha=0.3, linewidth=5)
    #         plt.plot(*self.s, 'ro')
    #         plt.plot(*self.t, 'bo')
    #
    #     planner = self.ss.getPlanner()
    #     pd = ob.PlannerData(self.ss.getSpaceInformation())
    #     planner.getPlannerData(pd)
    #     if edge_color == 'probability':
    #         pd.computeEdgeWeights(
    #             SurvivalObjective(self.ss.getSpaceInformation(), self.map, self.theta))
    #     g = nx.read_graphml(io.StringIO(pd.printGraphML()))
    #     ps = {n: list(map(float, data['coords'].split(',')[:2])) for n, data in g.nodes(data=True)}
    #     if edge_color == 'probability':
    #         edge_color = [np.exp(-d['weight']) for _, _, d in g.edges(data=True)]
    #         kwargs = {'edge_cmap': plt.cm.RdYlGn, 'edge_vmin': 0, 'edge_vmax': 1}
    #     else:
    #         kwargs = {}
    #     nx.draw_networkx(g, pos=ps, with_labels=False, edge_color=edge_color, node_color='grey',
    #                      node_size=1, arrows=False, alpha=1, width=1, **kwargs)

    def plot(self, size=(10, 10), save_to='', with_orientation=True, cmap=plt.cm.Greys_r,
             node_color='grey', with_nodes=True, edge_vmin=0, edge_vmax=1, width=1, node_size=1,
             edge_cmap=plt.cm.RdYlGn, path_linewidth=7, path_alpha=0.3, map_vmin=None,
             map_vmax=None, st_markersize=12, external_figure=False, margin=1, pad_inches=0.1):
        if not external_figure:
            plt.figure(figsize=size)
        plt.margins(margin)
        self.map.plot(cmap=cmap, vmin=map_vmin, vmax=map_vmax)
        g = self.graph
        edge_color = [d['probability'] for _, _, d in g.edges(data=True)]
        if with_nodes:
            f = nx.draw_networkx
        else:
            f = nx.draw_networkx_edges
        f(g, pos={n: d['pos'][:2] for n, d in g.nodes(data=True)},
          with_labels=False,
          edge_color=edge_color, node_color=node_color,
          node_size=node_size, arrows=False, alpha=1, width=width, edge_vmin=edge_vmin,
          edge_vmax=edge_vmax, edge_cmap=edge_cmap)
        xy = self.path
        if 'EXACT' in self.state:
            c = 'blue'
        else:
            c = 'cyan'
        plt.plot(*xy.T, '.-', alpha=path_alpha, linewidth=path_linewidth, color=c)
        plt.plot(*self.s[:2], 'o', color='cyan', markersize=st_markersize)
        plt.plot(*self.t[:2], 'o', color='blue', markersize=st_markersize)
        if with_orientation:
            plt.quiver(self.s[0], self.s[1], np.cos(self.theta), -np.sin(self.theta), color='cyan')
            plt.quiver(self.t[0], self.t[1], np.cos(self.theta), -np.sin(self.theta), color='blue')

        plt.axis('equal')
        plt.axis('off')
        if save_to:
            plt.savefig(save_to, pad_inches=pad_inches, transparent=True)

    @property
    def graph(self):
        ros = self.map.to_ros
        fros = self.map.from_ros
        planner = self.ss.getPlanner()
        pd = ob.PlannerData(self.ss.getSpaceInformation())
        planner.getPlannerData(pd)
        g = nx.read_graphml(io.StringIO(pd.printGraphML()))
        for n, data in g.nodes(data=True):
            x = list(map(float, data['coords'].split(',')[:2]))
            data['pos'] = x
            data['e_pos'] = ros(*x, with_z=True)
        traversable = self.map.traversable
        for s, t, data in g.edges(data=True):
            sx, sy = g.node[s]['pos']
            tx, ty = g.node[t]['pos']
            p, d = traversable((sx, sy, self.theta), (tx, ty), frame='abs')
            data['probability'] = p
            data['duration'] = d
        return g

    @property
    def path_data(self):
        ros = self.map.to_ros
        solution = self.ss.getSolutionPath()
        ss = [(x[0], x[1], self.theta) for x in solution.getStates()]
        xyt = [ros(x, y, theta) for x, y, theta in ss]
        ps, ts = np.array([self.map.traversable(s, t[:2], frame='abs')
                           for s, t in zip(ss, ss[1:])]).T
        return {'poses': [[round(x, 3) for x in pose] for pose in xyt],
                'probabilities': [round(x, 6) for x in ps.tolist()],
                'durations': [round(x, 3) for x in ts.tolist()],
                'duration': round(float(np.sum(ts)), 3),
                'probability': round(float(np.product(ps)), 6)}

    def data(self, with_simplified_path=True):
        if self.ss.haveSolutionPath():
            ros = self.map.to_ros
            data = {'s': ros(*self.s, with_z=True),
                    't': ros(*self.t),
                    'state': self.state,
                    'objective': self.objective,
                    'planner': {'name': self.planner_name, 'options': self.planner_options},
                    'threshold': self.threshold,
                    'target_tolerance': self.tolerance,
                    'computation_duration': self.comp_duration,
                    'motion_validator': {'too_short': self.motion_val.num_short,
                                         'not_traversable': self.motion_val.num_nt,
                                         'traversable': self.motion_val.num_t}}
            path = self.path_data
            data['path'] = path
            if with_simplified_path:
                self.ss.simplifySolution()
                simpl_path = self.path_data
                data['simplified_path'] = simpl_path
        else:
            data = {}
        return data

    def save(self, folder='', **kwargs):
        with open(f'{folder}/solution.yaml', 'w') as f:
            f.write(yaml.dump(self.data()))
        nx.write_gpickle(self.graph, f'{folder}/graph.pickle')
        self.plot(save_to=f'{folder}/solution.png', **kwargs)
