import io
from functools import lru_cache, partial

import numpy as np

import networkx as nx
import yaml
from matplotlib import pyplot as plt
from ompl import base as ob
from ompl import control as oc

from .map import move_pose


def tuple_from_state(state):
    return (state.getX(), state.getY(), state.getYaw())


def state_from_tuple(state, t):
    state.setX(t[0])
    state.setY(t[1])
    state.setYaw(t[2])


def tuple_from_control(control, allow_moving_backward=True, vmax=0.08):
    if allow_moving_backward:
        return (np.sign(control[0]) * vmax, control[1])
    else:
        return (vmax, control[1])


def valid(si, state):
    return si.satisfiesBounds(state)


def invalidate_state(state):
    state.setX(-1)


def control_success_probability(map_, pose, v_lin, v_ang, dt=1):
    p, _ = map_.traversable(pose, (v_lin, v_ang))
    return p


def propagate(data, map_, threshold, allow_moving_backward, start, control, duration, state):
    v, omega = tuple_from_control(control, allow_moving_backward)
    x, y, theta = pose = tuple_from_state(start)
    coords = move_pose(x, y, theta, v, omega, dt=duration)
    if control_success_probability(map_, pose, v_lin=v, v_ang=omega, dt=duration) > threshold:
        data['t'] += 1
        state_from_tuple(state, coords)
    else:
        data['nt'] += 1
        # A trick as the control planners have not function that check the validity of a motion
        # Because `propagateWhileValid` does it automatically (the caller of this function)
        invalidate_state(state)


@lru_cache(maxsize=None)
def control(pose1, pose2, v=0.08):
    omega = pose2[2] - pose1[2]
    if omega > np.pi:
        omega -= 2 * np.pi
    if omega < -np.pi:
        omega += 2 * np.pi
    pose3 = move_pose(*pose1, v=v, omega=omega)
    e = np.linalg.norm(np.array(pose3)[:2] - np.array(pose2)[:2])
    if np.abs(e) < 1e-3:
        return (v, omega)
    return (-v, omega)


def plot_edge(g, n1, n2, data, subdivisions=10, color=None, **kwargs):
    x = g.node[n1]['pos']
    poses = np.array([move_pose(*x, v=data['v'], omega=data['omega'], dt=dt)
                      for dt in np.linspace(0, 1, subdivisions)])
    if color is None:
        q = data['p']
        color = (1 - q, q, 0)
    x, y, theta = poses.T
    plt.plot(x, y, color=color, **kwargs)


class DurationObjective(ob.MechanicalWorkOptimizationObjective):

    # All motions have the same duration

    def __init__(self, si):
        super(DurationObjective, self).__init__(si)

    def stateCost(self, s):
        return ob.Cost(0)

    def motionCost(self, s1, s2):
        return ob.Cost(1)


def getBalancedObjective1(si, k=1.0):
    o1 = ob.PathLengthOptimizationObjective(si)
    o2 = DurationObjective(si)
    opt = ob.MultiOptimizationObjective(si)
    opt.addObjective(o1, 1.0)
    opt.addObjective(o2, k)
    return opt


class Plan:

    def __init__(self, map_, pose, target, max_time=120, allow_moving_backward=True,
                 omega_max=0.5, vmax=0.08, objective='duration', planner=oc.SST,
                 threshold=0.9, tolerance=0.3, control_duration=1, k=1.0, min_y=None, max_y=None):
        space = ob.SE2StateSpace()
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0)
        bounds.setHigh(map_.size)
        if min_y is not None:
            bounds.setLow(min_y)
        if max_y is not None:
            bounds.setHigh(max_y)
        space.setBounds(bounds)
        cspace = oc.RealVectorControlSpace(space, 2)
        cbounds = ob.RealVectorBounds(2)
        cbounds.setLow(-omega_max)
        cbounds.setHigh(omega_max)
        cspace.setBounds(cbounds)
        ss = oc.SimpleSetup(cspace)
        si = ss.getSpaceInformation()
        ss.setStateValidityChecker(ob.StateValidityCheckerFn(partial(valid, si)))
        self.propagation_data = {'t': 0, 'nt': 0}
        ss.setStatePropagator(
            oc.StatePropagatorFn(
                partial(propagate, self.propagation_data, map_, threshold, allow_moving_backward)))
        start = ob.State(space)
        state_from_tuple(start(), pose)
        goal = ob.State(space)
        state_from_tuple(goal(), target)
        ss.setStartAndGoalStates(start, goal, tolerance)
        p = planner(si)
        ss.setPlanner(p)
        si.setPropagationStepSize(control_duration)
        si.setMinMaxControlDuration(control_duration, control_duration)

        if objective == 'duration':
            do = DurationObjective(si)
        elif objective == 'balanced':
            do = getBalancedObjective1(si, k=k)
        else:
            do = ob.PathLengthOptimizationObjective(si)
        ss.setOptimizationObjective(do)

        ss.setup()
        self.ss = ss
        self.map = map_
        self.s = pose
        self.t = target
        self.objective = objective
        self.planner_name = 'SST'
        self.planner_options = {}
        self.threshold = threshold
        self.tolerance = tolerance
        self.comp_duration = 0
        self.max_time = max_time
        self.allow_moving_backward = allow_moving_backward

    def solve(self, dt=5, max_time=None):
        t = 0
        while t < (max_time or self.max_time):
            result = self.ss.solve(dt).getStatus()
            t += dt
            self.comp_duration += dt
            if result == ob.PlannerStatus.EXACT_SOLUTION:
                break
        self.state = result.name
        return result.name

    @property
    def path(self):
        if self.ss.haveSolutionPath():
            solution = self.ss.getSolutionPath()
            states = np.array([tuple_from_state(x) for x in solution.getStates()])
            controls = np.array([tuple_from_control(x, self.allow_moving_backward)
                                 for x in solution.getControls()])
            durations = list(solution.getControlDurations())
            return states, controls, durations

    def path_data(self, ros):
        states, controls, durations = self.path
        xyt = [ros(x, y, theta) for x, y, theta in states]
        ps, ls = np.array([self.map.traversable(tuple(s), tuple(c))
                           for s, c in zip(states, controls)]).T
        return {'poses': [[round(float(x), 3) for x in pose] for pose in xyt],
                'controls': [[round(float(x), 3) for x in control] for control in controls],
                'probabilities': [round(float(x), 6) for x in ps],
                'lengths': [round(x, 3) for x in ls.tolist()],
                'durations': [round(x, 3) for x in durations],
                'duration': round(float(np.sum(durations)), 3),
                'probability': round(float(np.product(ps)), 6)}

    def data(self, ros=True):
        if self.ss.haveSolutionPath():
            if ros:
                ros = self.map.to_ros
            else:
                ros = lambda *x, **k: x
            data = {'s': list(ros(*self.s, with_z=True)),
                    't': list(ros(*self.t)),
                    'state': self.state,
                    'objective': self.objective,
                    'planner': {'name': self.planner_name, 'options': self.planner_options},
                    'threshold': self.threshold,
                    'target_tolerance': self.tolerance,
                    'computation_duration': self.comp_duration}
            path = self.path_data(ros=ros)
            data['path'] = path
        else:
            data = {}
        return data

    def curve(self, dt=0.1, ros=True):
        xyt = []
        if ros:
            f = self.map.to_ros
        else:
            f = lambda *x: x
        for s, c, t in zip(*self.path):
            poses = np.array([f(*move_pose(*s, *c, dt=x))
                              for x in np.arange(0, t, dt)])
            xyt.append(poses)
        return np.concatenate(xyt)

    def graph(self, with_p=False):
        planner = self.ss.getPlanner()
        pd = ob.PlannerData(self.ss.getSpaceInformation())
        planner.getPlannerData(pd)
        g = nx.read_graphml(io.StringIO(pd.printGraphML()))
        traversable = self.map.traversable
        for n, data in g.nodes(data=True):
            x = tuple(map(float, data['coords'].split(',')[:3]))
            data['pos'] = x
        for n1, n2, data in list(g.edges(data=True)):
            x = g.node[n1]['pos']
            y = g.node[n2]['pos']
            v, omega = control(x, y)
            data['v'] = v
            data['omega'] = omega
            if with_p:
                data['p'], _ = traversable(x, (v, omega))
        return g

    def plot(self, size=(10, 10), plt=plt, save_to='', with_arrows=True, with_graph=False,
             with_nodes=False, edge_color='blue', edge_width=1, path_width=9, path_alpha=0.2,
             map_vmin=None, map_vmax=None, map_cmap=plt.cm.Greys, external_figure=True, dpi=200,
             found=False):
        if not external_figure:
            plt.figure(figsize=size)
        self.map.plot(cmap=map_cmap, vmin=map_vmin, vmax=map_vmax)
        if with_graph:
            g = self.graph(with_p=(edge_color is not None))
            for n1, n2, data in list(g.edges(data=True)):
                plot_edge(g, n1, n2, data, color=edge_color, linewidth=edge_width)
        xss = []
        yss = []
        for s, c, t in zip(*self.path):
            poses = np.array([move_pose(*s, *tuple_from_control(c), dt=dt)
                              for dt in np.linspace(0, t, 10)])
            x, y, yaw = poses[:].T
            e = np.array([[np.cos(-a), np.sin(-a)] for a in yaw[:1]])
            if with_arrows:
                plt.quiver(x[:1], y[:1], e[:, 0], e[:, 1], width=0.003)
            xss.append(x)
            yss.append(y)
        xs = np.concatenate(xss)
        ys = np.concatenate(yss)

        if 'EXACT' in self.state or found:
            c = 'blue'
        else:
            c = 'cyan'

        plt.plot(xs, ys, color=c, linewidth=path_width, alpha=path_alpha)

        plt.plot(*self.s[:2], 'o', color='cyan', alpha=1)
        plt.plot(*self.t[:2], 'o', color='blue', alpha=1)
        plt.quiver(self.s[0], self.s[1], np.cos(self.s[2]), -np.sin(self.s[2]),
                   color='cyan', alpha=1)
        plt.quiver(self.t[0], self.t[1], np.cos(self.t[2]), -np.sin(self.t[2]),
                   color='blue', alpha=1)

        plt.axis('equal')
        plt.axis('off')
        if save_to:
            plt.savefig(save_to, dpi=dpi)

        if with_graph:
            return g, len(xss)

    def save(self, folder='', ros=True, **kwargs):
        with open(f'{folder}/solution.yaml', 'w') as f:
            f.write(yaml.dump(self.data(ros=ros)))
        self.plot(save_to=f'{folder}/solution.png', **kwargs)
