import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import time, random, os
import seaborn as sns
import scipy
import glob
import pickle
import collections
import heapq
import copy


########################################################
# Superclass for any spreading model
########################################################

class SpreadingModel:
    # you probably do not want to touch this class
    def __init__(self, number_of_units=1):
        self.number_of_units = number_of_units  # only relevant for deterministic solution
        pass

    def states(self):
        return ['I', 'S']

    def get_number_of_units(self):
        try:
            return self.number_of_units
        except:
            return 1.0

    def colors(self):
        palette = sns.color_palette("muted", len(self.states()))
        colors = {s: list(palette[i]) for i, s in enumerate(self.states())}
        colors['I_total'] = 'gray'  # just to be save
        return colors

    def get_init_labeling(self, G, number_of_seeds=3, seed_state='I', non_seed_state='S'):
        assert (number_of_seeds <= G.number_of_nodes())
        nodes = list(G.nodes())
        seeds = {random.choice(nodes)}
        while len(seeds) < number_of_seeds:
            s = random.choice(list(seeds))
            s_neighbors = list(G.neighbors(s))
            new_seed = random.choice(s_neighbors)
            seeds.add(new_seed)
        return {n: seed_state if n in seeds else non_seed_state for n in range(G.number_of_nodes())}

    def reject(self, G, src_node, old_state, new_state, global_clock):
        return False  # do not reject

    # wrapper (logic in generate event)
    # this method is the important one in the super-class, do not overwrite it
    def next_event(self, G, src_node, global_clock, init_event=False):
        event_id = G.nodes[src_node]['event_id']
        event_id += 1
        new_time, new_state = self.generate_event(G, src_node, global_clock, init_event=init_event)
        G.nodes[src_node]['event_id'] = event_id

        # build event
        event_type = 'model'
        event_content = (src_node, new_state, event_id)
        event = (new_time, event_type, event_content)
        return event

    def generate_event(self, G, src_node, global_clock, init_event=False):
        return global_clock + random.random(), random.choice(self.states())

    def aggregate(self, node_state_counts):
        return node_state_counts

    # for the deterministic solution
    def ode_init(self):
        raise NotImplementedError
        # return [(1.0/len(self.states()))*self.get_number_of_units() for _ in self.states()]

    def ode_func(self, population_vector, t):
        raise NotImplementedError
        # return [0.001*i for i in range(len(self.states()))]

    def aggregate_ode(self, sol):
        return sol

    def ode_states(self):
        return self.states()


########################################################
# Classical SIS Model
########################################################

class SISmodel(SpreadingModel):
    def __init__(self, infection_rate):
        self.infection_rate = infection_rate

    def states(self):
        return ['I', 'S']

    def get_init_labeling(self, G):
        init_node_state = {n: ('I' if random.random() > 0.9 else 'S') for n in range(G.number_of_nodes())}
        return init_node_state

    def colors(self):
        return {'S': sns.xkcd_rgb['denim blue'], 'I': sns.xkcd_rgb['pinkish red']}

    def generate_event(self, G, src_node, global_clock, init_event=False):
        if G.nodes[src_node]['state'] == 'I':
            new_state = 'S'
            fire_time = -np.log(random.random())  # recov-rate is alsways 1
        else:
            new_state = 'I'
            inf_neighbors = len([n for n in G.neighbors(src_node) if G.nodes[n]['state'] == 'I'])
            if inf_neighbors == 0:
                fire_time = 10000000 + random.random()
            else:
                node_rate = inf_neighbors * self.infection_rate
                fire_time = -np.log(random.random()) / node_rate

        new_time = global_clock + fire_time
        return new_time, new_state


########################################################
# Classical SIR Model
########################################################

class SIRmodel(SpreadingModel):
    def __init__(self, infection_rate):
        self.infection_rate = infection_rate

    def states(self):
        return ['I', 'S', 'R']

    def colors(self):
        return {'S': sns.xkcd_rgb['denim blue'], 'I': sns.xkcd_rgb['pinkish red'], 'R': sns.xkcd_rgb['medium green']}

    def get_init_labeling(self, G):
        init_node_state = {n: ('I' if random.random() > 0.9 else 'S') for n in range(G.number_of_nodes())}
        return init_node_state

    def generate_event(self, G, src_node, global_clock, init_event=False):
        if G.nodes[src_node]['state'] == 'I':
            new_state = 'R'
            fire_time = -np.log(random.random())
        elif G.nodes[src_node]['state'] == 'S':
            new_state = 'I'
            inf_neighbors = len([n for n in G.neighbors(src_node) if G.nodes[n]['state'] == 'I'])
            if inf_neighbors == 0:
                fire_time = 10000000 + random.random()
            else:
                node_rate = inf_neighbors * self.infection_rate
                fire_time = -np.log(random.random()) / node_rate
        else:
            new_state = 'R'
            fire_time = 10000000 + random.random()

        new_time = global_clock + fire_time
        return new_time, new_state


########################################################
# Corona Model (inspired by Alison Hill)
########################################################

class CoronaHill(SpreadingModel):
    # find the excellent online tool at: https://alhill.shinyapps.io/COVID19seir/
    # conversion to a networked model based on scaling infection rate based on the mean degree of the network

    def __init__(self, scale_by_mean_degree=True, number_of_units=1, scale_inf_rate=1):

        b1 = 0.500  # / number of nodes      # infection rate from i1
        b2 = 0.100  # / number of nodes      # infection rate from i2
        b3 = 0.100  # / number of nodes      # infection rate from i3
        a = 0.200  # e to i1
        g1 = 0.133  # i1 to r
        g2 = 0.125  # i2 to r
        g3 = 0.075  # i3 to r
        p1 = 0.033  # i1 to i2
        p2 = 0.042  # i2 to i3
        u = 0.050  # i3 to death

        self.s_to_e_dueto_i1 = b1 * scale_inf_rate
        self.s_to_e_dueto_i2 = b2 * scale_inf_rate
        self.s_to_e_dueto_i3 = b3 * scale_inf_rate
        self.e_to_i1 = a
        self.i1_to_i2 = p1
        self.i2_to_i3 = p2
        self.i3_to_d = u
        self.i1_to_r = g1
        self.i2_to_r = g2
        self.i3_to_r = g3
        self.scale_by_mean_degree = scale_by_mean_degree

        self.number_of_units = number_of_units  # only relevant for deterministic ODE

    def states(self):
        return ['S', 'E', 'I1', 'I2', 'I3', 'R', 'D']

    def colors(self):
        colors = {'S': sns.xkcd_rgb['denim blue'], 'E': sns.xkcd_rgb['bright orange'], 'I1': sns.xkcd_rgb['light red'],
                  'I2': sns.xkcd_rgb['pinkish red'], 'I3': sns.xkcd_rgb['deep pink'], 'R': sns.xkcd_rgb['medium green'],
                  'D': sns.xkcd_rgb['black']}
        colors['I_total'] = 'gray'  # need to add states from finalize
        return colors

    def get_init_labeling(self, G):
        return super().get_init_labeling(G=G, seed_state='E', number_of_seeds=5, non_seed_state='S')

    def aggregate(self, node_state_counts):
        node_state_counts['I_total'] = [0 for _ in range(len(node_state_counts['I1']))]
        for i, v in enumerate(node_state_counts['I1']):
            node_state_counts['I_total'][i] += v
        for i, v in enumerate(node_state_counts['I2']):
            node_state_counts['I_total'][i] += v
        for i, v in enumerate(node_state_counts['I3']):
            node_state_counts['I_total'][i] += v
        return node_state_counts

    def generate_event(self, G, src_node, global_clock, init_event=False):
        if G.nodes[src_node]['state'] == 'S':
            new_state = 'E'
            neighbors = G.neighbors(src_node)
            count_i1 = len([n for n in neighbors if G.nodes[n]['state'] == 'I1'])
            count_i2 = len([n for n in neighbors if G.nodes[n]['state'] == 'I2'])
            count_i3 = len([n for n in neighbors if G.nodes[n]['state'] == 'I3'])
            if count_i1 + count_i2 + count_i3 == 0:
                fire_time = 10000000 + random.random()
            else:
                node_rate = count_i1 * self.s_to_e_dueto_i1 + count_i2 * self.s_to_e_dueto_i2 + count_i3 * self.s_to_e_dueto_i3
                if self.scale_by_mean_degree:
                    mean_degree = (2 * len(G.edges())) / G.number_of_nodes()
                    node_rate /= mean_degree
                fire_time = -np.log(random.random()) / node_rate

        elif G.nodes[src_node]['state'] == 'E':
            new_state = 'I1'
            fire_time = -np.log(random.random()) / self.e_to_i1

        elif G.nodes[src_node]['state'] == 'I1':
            new_state_c1 = 'I2'
            fire_time_c1 = -np.log(random.random()) / self.i1_to_i2
            new_state_c2 = 'R'
            fire_time_c2 = -np.log(random.random()) / self.i1_to_r
            if fire_time_c1 < fire_time_c2:
                new_state = new_state_c1
                fire_time = fire_time_c1
            else:
                new_state = new_state_c2
                fire_time = fire_time_c2

        elif G.nodes[src_node]['state'] == 'I2':
            new_state_c1 = 'I3'
            fire_time_c1 = -np.log(random.random()) / self.i2_to_i3
            new_state_c2 = 'R'
            fire_time_c2 = -np.log(random.random()) / self.i2_to_r
            if fire_time_c1 < fire_time_c2:
                new_state = new_state_c1
                fire_time = fire_time_c1
            else:
                new_state = new_state_c2
                fire_time = fire_time_c2

        elif G.nodes[src_node]['state'] == 'I3':
            new_state_c1 = 'D'
            fire_time_c1 = -np.log(random.random()) / self.i3_to_d
            new_state_c2 = 'R'
            fire_time_c2 = -np.log(random.random()) / self.i3_to_r
            if fire_time_c1 < fire_time_c2:
                new_state = new_state_c1
                fire_time = fire_time_c1
            else:
                new_state = new_state_c2
                fire_time = fire_time_c2
        elif G.nodes[src_node]['state'] == 'R':
            new_state = 'R'
            fire_time = 10000000 + random.random()
        elif G.nodes[src_node]['state'] == 'D':
            new_state = 'D'
            fire_time = 10000000 + random.random()
        else:
            print('no matching state')
            assert (False)

        new_time = global_clock + fire_time
        return new_time, new_state

    # ODE

    # has to be a vector in the order of models.states()
    def ode_init(self):
        init = [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
        init = [x * self.number_of_units for x in init]
        return init

    def ode_func(self, population_vector, t):
        s = population_vector[0]
        e = population_vector[1]
        i1 = population_vector[2]
        i2 = population_vector[3]
        i3 = population_vector[4]
        r = population_vector[5]
        d = population_vector[6]

        s_grad = -(
                self.s_to_e_dueto_i1 / self.number_of_units * i1 + self.s_to_e_dueto_i2 / self.number_of_units * i3 + self.s_to_e_dueto_i3 / self.number_of_units * i3) * s
        e_grad = (
                         self.s_to_e_dueto_i1 / self.number_of_units * i1 + self.s_to_e_dueto_i2 / self.number_of_units * i3 + self.s_to_e_dueto_i3 / self.number_of_units * i3) * s - self.e_to_i1 * e
        i1_grad = self.e_to_i1 * e - (self.i1_to_r + self.i1_to_i2) * i1
        i2_grad = self.i1_to_i2 * i1 - (self.i2_to_r + self.i2_to_i3) * i2
        i3_grad = self.i2_to_i3 * i2 - (self.i3_to_r + self.i3_to_d) * i3
        r_grad = self.i1_to_r * i1 + self.i2_to_r * i2 + self.i3_to_r * i3
        d_grad = self.i3_to_d * i3

        grad = [s_grad, e_grad, i1_grad, i2_grad, i3_grad, r_grad, d_grad]

        return grad


########################################################
# Model by José Lourenço et al.
# (not tested, no deads yet)
# Oxford model: https://www.medrxiv.org/content/10.1101/2020.03.24.20042291v1.full.pdf
########################################################
class CoronaLourenco(SpreadingModel):
    def __init__(self, scale_by_mean_degree=True, number_of_units=1, r_0=2.75):

        self.sigma = 1.0 / 4.5  # recovery rate
        self.r_0 = r_0
        self.beta = self.sigma * self.r_0  # infection rate

        self.scale_by_mean_degree = scale_by_mean_degree
        self.number_of_units = number_of_units

    def states(self):
        return ['I', 'S', 'R']

    def colors(self):
        return {'S': sns.xkcd_rgb['denim blue'], 'I': sns.xkcd_rgb['pinkish red'], 'R': sns.xkcd_rgb['medium green']}

    def get_init_labeling(self, G):
        return super().get_init_labeling(G=G, seed_state='I', number_of_seeds=3, non_seed_state='S')

    def generate_event(self, G, src_node, global_clock, init_event=False):
        if G.nodes[src_node]['state'] == 'I':
            new_state = 'R'
            recovery_rate = self.sigma
            fire_time = -np.log(random.random()) / recovery_rate
        elif G.nodes[src_node]['state'] == 'S':
            new_state = 'I'
            inf_neighbors = len([n for n in G.neighbors(src_node) if G.nodes[n]['state'] == 'I'])
            if inf_neighbors == 0:
                fire_time = 10000000 + random.random()
            else:
                node_rate = inf_neighbors * self.beta
                if self.scale_by_mean_degree:
                    mean_degree = (2 * len(G.edges())) / G.number_of_nodes()
                    node_rate /= (mean_degree - self.r_0)  # TODO this is the correct scaling
                fire_time = -np.log(random.random()) / node_rate
        else:
            new_state = 'R'
            fire_time = 10000000 + random.random()

        new_time = global_clock + fire_time
        return new_time, new_state

    # has to be a vector in the order of models.states()
    def ode_init(self):
        init = [0.03, 0.97, 0.0]
        init = [x * self.number_of_units for x in init]
        return init

    def ode_func(self, population_vector, t):
        i = population_vector[0]
        s = population_vector[1]
        r = population_vector[2]

        s_grad = -(self.beta / self.number_of_units * i) * s
        i_grad = (self.beta / self.number_of_units * i) * s - (self.sigma * i)
        r_grad = self.sigma * i

        grad = [i_grad, s_grad, r_grad]

        return grad


class CoronaHelmholz(SpreadingModel):
    def __init__(self, G, number_of_units=1, r_0=2.75):
        self.mean_degree = (2 * len(G.edges())) / G.number_of_nodes()
        self.recovery_from_c = 1 / 5.2  # this is the recovery rate assuming gamma/suppression = 0, it is equal to c_out, could be 3.2
        self.r_0_hat = r_0
        self.infection_rate_ODE = self.r_0_hat * self.recovery_from_c  # free variable in helmholz paper

        # print('infec rate ODE : ', self.infection_rate_ODE )

        assert (self.mean_degree > self.r_0_hat)

        self.infection_rate = self.infection_rate_ODE / (self.mean_degree - self.r_0_hat)
        self.suppression = 0.2  # beta or gamma, how much interaction takes still place when sick

        self.e_to_c = 1 / 5.2
        self.c_out = self.recovery_from_c
        self.c_rec_prop = 0.08
        self.i_out = 1 / 5
        self.i_rec_prop = 0.8
        self.h_out = 1 / 10
        self.h_rec_prop = 0.74
        self.u_out = 1 / 8
        self.u_rec_prop = 0.46

        self.number_of_nodes = G.number_of_nodes()

    def states(self):
        return ['S', 'E', 'C', 'I', 'H', 'U', 'R', 'D']

    def ode_states(self):
        return ['S', 'E', 'C', 'I', 'H', 'U', 'R', 'D', 'I_total']

    def colors(self):
        colors = {'S': (76 / 255, 114 / 255, 176 / 255), 'E': (253 / 255, 170 / 255, 72 / 255),
                  'C': (255 / 255, 71 / 255, 76 / 255),
                  'I': (203 / 255, 1 / 255, 98 / 255), 'H': (146 / 255, 30 / 255, 156 / 255),
                  'U': (80 / 255, 20 / 255, 100 / 255),
                  'D': sns.xkcd_rgb['black'], 'R': (85 / 255, 168 / 255, 104 / 255)}
        colors['I_total'] = 'gray'
        return colors

    def markov_firing(self, state_to_rate):
        state_rate = list(state_to_rate.items())
        state_firetime = [(s, -np.log(random.random()) / r) for s, r in state_rate]
        state_firetime = sorted(state_firetime, key=lambda s_t: s_t[1])
        min_state = state_firetime[0][0]
        min_time = state_firetime[0][1]
        return min_state, min_time

    def get_init_labeling(self, G, neighbours=False):
        if neighbours:
            return super().get_init_labeling(G=G, seed_state='C', number_of_seeds=3, non_seed_state='S')

        seeds = np.random.choice(list(G.nodes), 3, replace=True)
        return {n: 'C' if n in seeds else 'S' for n in G.nodes()}

    def generate_event(self, G, src_node, global_clock, init_event=False):
        local_state = G.nodes[src_node]['state']
        inf_rate = self.infection_rate
        neighbors = G.neighbors(src_node)

        neighbor_states = (([G.nodes[n]['state'] for n in neighbors if G.nodes[n]['state']]))
        neighbors_carrier = neighbor_states.count('C')
        neighbors_infected = neighbor_states.count('I')

        if local_state == 'S':
            if neighbors_carrier + neighbors_infected == 0:
                return 100000000.0, 'S'
            actual_inf_rate = neighbors_carrier * inf_rate + neighbors_infected * inf_rate * self.suppression
            state_to_rate = {'E': actual_inf_rate}
            new_state, fire_time = self.markov_firing(state_to_rate)
        elif local_state == 'E':
            state_to_rate = {'C': self.e_to_c}
            new_state, fire_time = self.markov_firing(state_to_rate)
        elif local_state == 'C':
            state_to_rate = {'I': self.c_out * (1 - self.c_rec_prop), 'R': self.c_out * self.c_rec_prop}
            new_state, fire_time = self.markov_firing(state_to_rate)
        elif local_state == 'I':
            state_to_rate = {'H': self.i_out * (1 - self.i_rec_prop), 'R': self.i_out * self.i_rec_prop}
            new_state, fire_time = self.markov_firing(state_to_rate)
        elif local_state == 'H':
            state_to_rate = {'U': self.h_out * (1 - self.h_rec_prop), 'R': self.h_out * self.h_rec_prop}
            new_state, fire_time = self.markov_firing(state_to_rate)
        elif local_state == 'U':
            state_to_rate = {'D': self.u_out * (1 - self.u_rec_prop), 'R': self.u_out * self.u_rec_prop}
            new_state, fire_time = self.markov_firing(state_to_rate)
        else:  # dummy event
            new_state = local_state
            fire_time = 1000000 + random.random()

        new_time = global_clock + fire_time
        return new_time, new_state

    def aggregate(self, node_state_counts):
        node_state_counts['I_total'] = [0 for _ in range(len(node_state_counts['I']))]
        for i, v in enumerate(node_state_counts['E']):
            node_state_counts['I_total'][i] += v
        for i, v in enumerate(node_state_counts['C']):
            node_state_counts['I_total'][i] += v
        for i, v in enumerate(node_state_counts['I']):
            node_state_counts['I_total'][i] += v
        for i, v in enumerate(node_state_counts['H']):
            node_state_counts['I_total'][i] += v
        for i, v in enumerate(node_state_counts['U']):
            node_state_counts['I_total'][i] += v
        return node_state_counts

    # has to be a vector in the order of models.states()
    def ode_init(self):
        # ['S', 'E', 'C', 'I', 'H', 'U', 'R', 'D']
        seed = 3
        nodes = self.number_of_nodes
        init = [(nodes - seed) / nodes, 0.0, seed/nodes, 0.0, 0.0, 0.0, 0.0, 0.0]
        return init

    def ode_func(self, population_vector, t):
        s = population_vector[0]
        e = population_vector[1]
        c = population_vector[2]
        i = population_vector[3]
        h = population_vector[4]
        u = population_vector[5]
        r = population_vector[6]
        d = population_vector[7]

        s_grad = -self.infection_rate_ODE * (c + self.suppression * i) * s
        e_grad = self.infection_rate_ODE * (c + self.suppression * i) * s - self.e_to_c * e
        c_grad = self.e_to_c * e - self.c_out * c
        i_grad = c * self.c_out * (1 - self.c_rec_prop) - self.i_out * i
        h_grad = i * self.i_out * (1 - self.i_rec_prop) - self.h_out * h
        u_grad = h * self.h_out * (1 - self.h_rec_prop) - self.u_out * u

        r_grad = h * self.h_out * self.h_rec_prop + i * self.i_out * self.i_rec_prop + u * self.u_out * self.u_rec_prop + c * self.c_out * self.recovery_from_c
        d_grad = u * self.u_out * (1 - self.u_rec_prop)

        grad = [s_grad, e_grad, c_grad] + [i_grad, h_grad, u_grad] + [r_grad, d_grad]

        return grad

    def ssa(self, graph, time_point_samples, rt_log=None):
        rt_dict = dict()
        global_clock = 0.0
        x_values = list(time_point_samples)
        y_values = {state: list() for state in self.states()}  # record of trajectory

        #inf_rate = self.infection_rate_ODE/graph.number_of_nodes()
        inf_rate = self.infection_rate

        init_map = dict()
        for s in self.states():
            if s not in init_map:
                init_map[s] = 0
        for n in graph.nodes():
            init_map[graph.nodes[n]['state']] += 1


        agents = list(range(np.sum(list(init_map.values()))))
        agents_temp = list(agents)
        state_map = dict()
        for state, count in init_map.items():
            state_map[state] = agents_temp[:count]
            agents_temp = agents_temp[count:]

        for a in agents:
            rt_dict[(a, 'infected')] = 0 # infected zero and active at horizon
            rt_dict[(a, 'start')] = time_point_samples[-1] + 10  # infected zero and active at horizon
        for a in agents:
            if not a in state_map['S']:
                rt_dict[(a, 'start')] = 0.0

        while True:
            reaction_to_rate = dict()
            reaction_to_rate[('S', 'E')] = len(state_map['S']) * inf_rate * (len(state_map['C']) + self.suppression *  len(state_map['I']))
            reaction_to_rate[('E', 'C')] = len(state_map['E']) * self.e_to_c
            reaction_to_rate[('C', 'I')] = len(state_map['C']) * self.c_out * (1-self.c_rec_prop)
            reaction_to_rate[('C', 'R')] = len(state_map['C']) * self.c_out * (self.c_rec_prop)
            reaction_to_rate[('I', 'H')] = len(state_map['I']) * self.i_out * (1-self.i_rec_prop)
            reaction_to_rate[('I', 'R')] = len(state_map['I']) * self.i_out * (self.i_rec_prop)
            reaction_to_rate[('H', 'U')] = len(state_map['H']) * self.h_out * (1-self.h_rec_prop)
            reaction_to_rate[('H', 'R')] = len(state_map['H']) * self.h_out * (self.h_rec_prop)
            reaction_to_rate[('U', 'D')] = len(state_map['U']) * self.u_out * (1-self.u_rec_prop)
            reaction_to_rate[('U', 'R')] = len(state_map['U']) * self.u_out * (self.u_rec_prop)

            reaction, fire_time = self.markov_firing(reaction_to_rate)
            global_clock += fire_time
            while global_clock > x_values[0]:
                for state in self.states():
                    y_values[state].append(len(state_map[state]))
                x_values = x_values[1:]
                if len(x_values) == 0:
                    for a in agents:
                        rt_log.add_data_point(rt_dict[(a, 'start')], rt_dict[(a, 'infected')])
                    return y_values

            state_minus, state_plus = reaction
            assert(len(state_map[state_minus]) > 0 )
            agent = state_map[state_minus][0]
            state_map[state_plus] = state_map[state_plus] + [state_map[state_minus][0]]
            state_map[state_minus] = state_map[state_minus][1:]

            if state_plus == 'E':
                rt_dict[(agent, 'start')] = global_clock
                car_neighbors = state_map['C']
                inf_neighbors = state_map['I']
                z = len(car_neighbors) + len(inf_neighbors) * 0.2  # hard coded gamma!!!
                assert (len(car_neighbors) + 0.2 * len(inf_neighbors) == z)
                for n_c in car_neighbors:
                    rt_dict[(n_c, 'infected')] += 1.0 / z
                for n_i in inf_neighbors:
                    rt_dict[(n_i, 'infected')] += 0.2 / z



    def aggregate_ode(self, sol):
        sol_e = sol[:, 1]
        sol_c = sol[:, 2]
        sol_i = sol[:, 3]
        sol_h = sol[:, 4]
        sol_u = sol[:, 5]
        sol_total = sol_e+sol_c+sol_i+sol_h+sol_u
        sol = sol.tolist()
        for i, l in enumerate(sol):
            sol[i].append(sol_total[i])
        sol = np.array(sol)
        return sol

class CoronaBase(SpreadingModel):
    def __init__(self, number_of_units=1, hospital_capacity=None, scale_by_mean_degree=True):

        # Q = quarantined, H = in hospital, m = mild course of disease, s = severe course of disease

        # high level config
        self.r_0 = 5.0  # TODO: set higher for severe
        self.mean_days_until_recovery = 15
        self.fraction_of_mild = 4 / 5
        self.mean_days_latent = 1
        self.mean_days_until_death = 15

        # concrete parameters
        self.infection_rate = self.r_0 * (1 / self.mean_days_until_recovery)

        self.q_effect = 0.1  # infection rate of quarantined spreader is only 10% of self.infection_rate
        self.self_q_mild = 0.1  # rate to go to q because self is infected when mild course (relates to probabaility to become aware of infection)
        self.self_q_severe = 0.5  # becoming aware of infection is easier if severe course
        self.precautionary_q = 0.5  # rate to go to q because neighbor is infected and aware (relates to probabaility to be detected by contact tracing)
        self.lift_q = 1 / 17  # does not apply when in I_severe

        self.e_to_im = (1 / self.mean_days_latent) * self.fraction_of_mild  # rate for mild (or symptomless) course
        self.e_to_is = (self.mean_days_latent) * (1 - self.fraction_of_mild)  # rate for severe course

        self.im_to_r = 1 / self.mean_days_until_recovery

        is_to_r = 1 / 6
        is_to_ish = 4 / 6  # go to hospital in 2/3 cases
        is_to_d = 1 / 6
        is_leave_rate = 1 / 5  # 5 days in state severe

        self.is_to_r = is_to_r * is_leave_rate
        self.is_to_ish = is_to_ish * is_leave_rate
        self.is_to_d = is_to_d * is_leave_rate

        ish_to_d = 1 / 5  # recover in hosptial with chance 4/5
        ish_to_r = 4 / 5
        ish_leave_rate = 1 / 10  # leave hospital after 10 days

        self.ish_to_d = ish_to_d * ish_leave_rate
        self.ish_to_r = ish_to_r * ish_leave_rate

        self.number_of_units = number_of_units
        self.hospital_capacity = hospital_capacity
        self.nodes_in_hospital = list()
        self.scale_by_mean_degree = scale_by_mean_degree

    def markov_firing(self, state_to_rate):
        state_rate = list(state_to_rate.items())
        state_firetime = [(s, -np.log(random.random()) / r) for s, r in state_rate]
        state_firetime = sorted(state_firetime, key=lambda s_t: s_t[1])
        min_state = state_firetime[0][0]
        min_time = state_firetime[0][1]
        return min_state, min_time

    def get_init_labeling(self, G):
        return super().get_init_labeling(G=G, seed_state='E', number_of_seeds=3, non_seed_state='S')

    def reject(self, G, src_node, old_state, new_state, global_clock):
        if self.hospital_capacity is None:
            return False

        if 'H' in new_state:
            new_frac_in_h = (len(self.nodes_in_hospital) + 1) / G.number_of_nodes()
            if new_frac_in_h > self.hospital_capacity:
                self.nodes_in_hospital.append(src_node)
                return True  # reject - ICU is full
            else:
                return False
        if 'H' in old_state:
            self.nodes_in_hospital.remove(src_node)
        return False

    def states(self):
        return ['S', 'E', 'Im', 'Is', 'R'] + ['SQ', 'EQ', 'ImQ', 'IsQ', 'RQ'] + ['IsH', 'D']

    def generate_event(self, G, src_node, global_clock, init_event=False):
        local_state = G.nodes[src_node]['state']
        neighbors = G.neighbors(src_node)
        assert (local_state in self.states())

        # divide neighbors into spreaders that know they are infected and those who do not
        neighbor_states = (([G.nodes[n]['state'] for n in neighbors if G.nodes[n]['state']]))
        neighbors_aware = neighbor_states.count('ImQ') + neighbor_states.count('IsQ') + neighbor_states.count('IsH')
        neighbors_unaware = neighbor_states.count('Im') + neighbor_states.count('Is')

        # scale inf rate
        inf_rate = self.infection_rate
        if self.scale_by_mean_degree:
            mean_degree = (2 * len(G.edges())) / G.number_of_nodes()
            inf_rate /= mean_degree

        if local_state == 'S':
            if neighbors_aware + neighbors_unaware == 0:
                return 100000000.0, 'S'
            actual_inf_rate = neighbors_unaware * inf_rate + neighbors_aware * inf_rate * self.q_effect
            # scale infection rate down for neighbors in q
            state_to_rate = {'SQ': neighbors_aware * self.precautionary_q, 'E': actual_inf_rate}
            new_state, fire_time = self.markov_firing(state_to_rate)
        elif local_state == 'SQ':
            if neighbors_aware + neighbors_unaware == 0:
                return 100000000.0, 'SQ'
            actual_inf_rate = neighbors_unaware * inf_rate + neighbors_aware * inf_rate * self.q_effect
            actual_inf_rate *= self.q_effect  # apply scaling twice cause self is in q and neighbors are potentially in q
            state_to_rate = {'S': self.lift_q, 'EQ': actual_inf_rate}
            new_state, fire_time = self.markov_firing(state_to_rate)

        elif local_state == 'E':
            state_to_rate = {'EQ': neighbors_aware * self.precautionary_q, 'Is': self.e_to_is, 'Im': self.e_to_im}
            new_state, fire_time = self.markov_firing(state_to_rate)
        elif local_state == 'EQ':
            state_to_rate = {'E': self.lift_q, 'IsQ': self.e_to_is, 'ImQ': self.e_to_im}
            new_state, fire_time = self.markov_firing(state_to_rate)

        elif local_state == 'Im':
            state_to_rate = {'ImQ': neighbors_aware * self.precautionary_q + self.self_q_mild, 'R': self.im_to_r}
            new_state, fire_time = self.markov_firing(state_to_rate)
        elif local_state == 'ImQ':
            state_to_rate = {'Im': self.lift_q, 'RQ': self.im_to_r}
            new_state, fire_time = self.markov_firing(state_to_rate)

        elif local_state == 'Is':
            state_to_rate = {'IsQ': neighbors_aware * self.precautionary_q + self.self_q_severe, 'R': self.is_to_r,
                             'D': self.is_to_d, 'IsH': self.is_to_ish}
            new_state, fire_time = self.markov_firing(state_to_rate)
        elif local_state == 'IsQ':
            state_to_rate = {'RQ': self.is_to_r, 'D': self.is_to_d, 'IsH': self.is_to_ish}
            new_state, fire_time = self.markov_firing(state_to_rate)

        elif local_state == 'IsH':
            # when you recover in hospital you do not go into Q after that
            state_to_rate = {'R': self.ish_to_r, 'D': self.ish_to_d}
            new_state, fire_time = self.markov_firing(state_to_rate)

        elif local_state == 'RQ':
            state_to_rate = {'R': self.lift_q}
            new_state, fire_time = self.markov_firing(state_to_rate)

        else:  # dummy event
            new_state = local_state
            fire_time = 1000000 + random.random()

        new_time = global_clock + fire_time

        return new_time, new_state


########################################################
# Corona Model (inspired by Alison Hill)
########################################################

class CoronaHillWSourceTracing(SpreadingModel):
    # find the excellent online tool at: https://alhill.shinyapps.io/COVID19seir/
    # conversion to a networked model based on scaling infection rate based on the mean degree of the network

    def __init__(self, scale_by_mean_degree=True, init_exposed=None, number_of_units=1, scale_inf_rate=1):

        b1 = 0.500  # / number of nodes      # infection rate from i1
        b2 = 0.100  # / number of nodes      # infection rate from i2
        b3 = 0.100  # / number of nodes      # infection rate from i3
        a = 0.200  # e to i1
        g1 = 0.133  # i1 to r
        g2 = 0.125  # i2 to r
        g3 = 0.075  # i3 to r
        p1 = 0.033  # i1 to i2
        p2 = 0.042  # i2 to i3
        u = 0.050  # i3 to death

        # quarantine rates
        q1 = 0.5  # i1 to i1q  # less likely to detect
        q2 = 1.0  # i2 to i2q
        q3 = 1.0  # i3 to i3q

        self.s_to_e_dueto_i1 = b1 * scale_inf_rate
        self.s_to_e_dueto_i2 = b2 * scale_inf_rate
        self.s_to_e_dueto_i3 = b3 * scale_inf_rate
        self.e_to_i1 = a
        self.i1_to_i2 = p1
        self.i2_to_i3 = p2
        self.i3_to_d = u
        self.i1_to_r = g1
        self.i2_to_r = g2
        self.i3_to_r = g3
        self.i1_to_i1q = q1
        self.i2_to_i2q = q2
        self.i3_to_i3q = q3
        self.scale_by_mean_degree = scale_by_mean_degree
        self.init_exposed = init_exposed

        self.number_of_units = number_of_units  # only relevant for deterministic ODE

        self.minqtime = 14  # minimal time to be in q

    def states(self):
        return ['S', 'E', 'SQ', 'EQ', 'I1', 'I2', 'I3', 'I1Q', 'I2Q', 'I3Q', 'R', 'D']

    def colors(self):
        colors = {'S': sns.xkcd_rgb['denim blue'], 'E': sns.xkcd_rgb['bright orange'], 'I1': sns.xkcd_rgb['light red'],
                  'I2': sns.xkcd_rgb['pinkish red'], 'I3': sns.xkcd_rgb['deep pink'], 'R': sns.xkcd_rgb['medium green'],
                  'D': sns.xkcd_rgb['black']}
        colors['I_total'] = 'gray'  # need to add states from finalize
        return colors

    def get_init_labeling(self, G):
        if self.init_exposed is not None:
            init_node_state = {n: 'S' for n in range(G.number_of_nodes())}
            for exp_node in self.init_exposed:
                init_node_state[exp_node] = 'E'
            return init_node_state
        init_node_state = {n: ('E' if random.random() > 0.90 else 'S') for n in range(G.number_of_nodes())}
        return init_node_state

    def aggregate(self, node_state_counts):
        node_state_counts['I_total'] = [0 for _ in range(len(node_state_counts['I1']))]
        for i, v in enumerate(node_state_counts['I1']):
            node_state_counts['I_total'][i] += v
        for i, v in enumerate(node_state_counts['I2']):
            node_state_counts['I_total'][i] += v
        for i, v in enumerate(node_state_counts['I3']):
            node_state_counts['I_total'][i] += v
        return node_state_counts

    def generate_event(self, G, src_node, global_clock, init_event=False):
        if init_event:
            G.nodes[src_node]['min_lifting_time'] = 1000000000.0  # dummy

        if G.nodes[src_node]['state'] == 'S':
            new_state = 'E'
            neighbors = G.neighbors(src_node)
            count_i1 = len([n for n in neighbors if G.nodes[n]['state'] == 'I1'])
            count_i2 = len([n for n in neighbors if G.nodes[n]['state'] == 'I2'])
            count_i3 = len([n for n in neighbors if G.nodes[n]['state'] == 'I3'])
            if count_i1 + count_i2 + count_i3 == 0:
                fire_time = 10000000 + random.random()
            else:
                node_rate = count_i1 * self.s_to_e_dueto_i1 + count_i2 * self.s_to_e_dueto_i2 + count_i3 * self.s_to_e_dueto_i3
                if self.scale_by_mean_degree:
                    mean_degree = (2 * len(G.edges())) / G.number_of_nodes()
                    node_rate /= mean_degree
                fire_time = -np.log(random.random()) / node_rate

        elif G.nodes[src_node]['state'] == 'E':
            new_state = 'I1'
            fire_time = -np.log(random.random()) / self.e_to_i1

        elif G.nodes[src_node]['state'] == 'I1':
            u1, u2 = np.random.rand(2)
            rate0 = self.i1_to_i2 + self.i1_to_r + self.i1_to_i1q
            fire_time = -np.log(u1) / rate0
            u2 *= rate0
            if u2 < self.i1_to_i2:
                new_state = 'I2'
            elif u2 < self.i1_to_i2 + self.i1_to_r:
                new_state = 'R'
            else:
                new_state = 'I1Q'

        elif G.nodes[src_node]['state'] == 'I2':
            u1, u2 = np.random.rand(2)
            rate0 = self.i2_to_i3 + self.i2_to_r + self.i2_to_i2q
            fire_time = -np.log(u1) / rate0
            u2 *= rate0
            if u2 < self.i2_to_i3:
                new_state = 'I3'
            elif u2 < self.i2_to_i3 + self.i2_to_r:
                new_state = 'R'
            else:
                new_state = 'I2Q'

        elif G.nodes[src_node]['state'] == 'I3':
            u1, u2 = np.random.rand(2)
            rate0 = self.i3_to_d + self.i3_to_r + self.i3_to_i3q
            fire_time = -np.log(u1) / rate0
            u2 *= rate0
            if u2 < self.i3_to_d:
                new_state = 'D'
            elif u2 < self.i3_to_d + self.i3_to_r:
                new_state = 'R'
            else:
                new_state = 'I3Q'

        elif G.nodes[src_node]['state'] == 'I1Q':
            new_state_c1 = 'I2Q'
            fire_time_c1 = -np.log(random.random()) / self.i1_to_i2
            new_state_c2 = 'R'
            fire_time_c2 = -np.log(random.random()) / self.i1_to_r
            if fire_time_c1 < fire_time_c2:
                new_state = new_state_c1
                fire_time = fire_time_c1
            else:
                new_state = new_state_c2
                fire_time = fire_time_c2

        elif G.nodes[src_node]['state'] == 'I2Q':
            new_state_c1 = 'I3Q'
            fire_time_c1 = -np.log(random.random()) / self.i2_to_i3
            new_state_c2 = 'R'
            fire_time_c2 = -np.log(random.random()) / self.i2_to_r
            if fire_time_c1 < fire_time_c2:
                new_state = new_state_c1
                fire_time = fire_time_c1
            else:
                new_state = new_state_c2
                fire_time = fire_time_c2

        elif G.nodes[src_node]['state'] == 'I3Q':
            new_state_c1 = 'D'
            fire_time_c1 = -np.log(random.random()) / self.i3_to_d
            new_state_c2 = 'R'
            fire_time_c2 = -np.log(random.random()) / self.i3_to_r
            if fire_time_c1 < fire_time_c2:
                new_state = new_state_c1
                fire_time = fire_time_c1
            else:
                new_state = new_state_c2
                fire_time = fire_time_c2

        elif G.nodes[src_node]['state'] == 'R':
            new_state = 'R'
            fire_time = 10000000 + random.random()
        elif G.nodes[src_node]['state'] == 'D':
            new_state = 'D'
            fire_time = 10000000 + random.random()
        else:
            print('no matching state')
            assert (False)

        new_time = global_clock + fire_time

        # check q
        if new_time > G.nodes[src_node]['min_lifting_time']:
            if G.nodes[src_node]['state'] == 'R':
                new_time, new_state = G.nodes[src_node]['min_lifting_time'], G.nodes[src_node]['state'].replace('Q', '')
            else:
                G.nodes[src_node]['min_lifting_time'] = new_time + 1
        return new_time, new_state

    # ODE

    # has to be a vector in the order of models.states()
    def ode_init(self):
        init = [0.95, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
        init = [x * self.number_of_units for x in init]
        return init

    def ode_func(self, population_vector, t):
        s = population_vector[0]
        e = population_vector[1]
        i1 = population_vector[2]
        i2 = population_vector[3]
        i3 = population_vector[4]
        r = population_vector[5]
        d = population_vector[6]

        s_grad = -(
                self.s_to_e_dueto_i1 / self.number_of_units * i1 + self.s_to_e_dueto_i2 / self.number_of_units * i3 + self.s_to_e_dueto_i3 / self.number_of_units * i3) * s
        e_grad = (
                         self.s_to_e_dueto_i1 / self.number_of_units * i1 + self.s_to_e_dueto_i2 / self.number_of_units * i3 + self.s_to_e_dueto_i3 / self.number_of_units * i3) * s - self.e_to_i1 * e
        i1_grad = self.e_to_i1 * e - (self.i1_to_r + self.i1_to_i2) * i1
        i2_grad = self.i1_to_i2 * i1 - (self.i2_to_r + self.i2_to_i3) * i2
        i3_grad = self.i2_to_i3 * i2 - (self.i3_to_r + self.i3_to_d) * i3
        r_grad = self.i1_to_r * i1 + self.i2_to_r * i2 + self.i3_to_r * i3
        d_grad = self.i3_to_d * i3

        grad = [s_grad, e_grad, i1_grad, i2_grad, i3_grad, r_grad, d_grad]

        return grad
