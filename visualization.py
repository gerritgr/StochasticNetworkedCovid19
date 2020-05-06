import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import imageio
import glob
import random


########################################################
# Visualizaion code
########################################################

def viz_simulate(G, time_points, storage, model, node_pos=None, outpath='output_simulation_NUM.jpg'):
    os.system('mkdir output_gif')  # might be redundant #TODO fix paths
    rand_id = str(random.choice(range(1000000)))
    folder = 'output_gif/vis_out_1' + rand_id + '/'  # use random folder path to not overwrite
    os.system('mkdir {}'.format(folder))
    if node_pos is None:
        # pos=nx.spring_layout(G,seed=11)
        pos = nx.spectral_layout(G)
        # pos = nx.kamada_kawai_layout(G)
        node_pos = [(pos[i][0], pos[i][1]) for i in range(G.number_of_nodes())]

    color_palette = sns.color_palette("muted", len(model.states()))
    try:
        color_palette = [model.colors()[state] for state in model.states()]
    except:
        pass

    # storage = storage/np.max(storage)
    for i in range(storage.shape[1]):
        node_labels = [int(x) for x in storage[:, i]]
        plt.clf()

        # nodes
        for node in G.nodes():
            c = storage[node, i]
            c_rgb = color_palette[int(
                c + 0.000001)]  # pyplot: Please use a 2-D array with a single row if you really want to specify the same RGB
            s = 1.0 / len(node_pos) * 150 * 70
            plt.scatter([node_pos[node][0]], [node_pos[node][1]], s=s, alpha=0.8, zorder=15, c=[c_rgb],
                        edgecolors='none')

        # edges
        lw = min(3, 1.0 / len(G.edges()) * 600)
        for e in G.edges:
            pos_v1 = node_pos[e[0]]
            pos_v2 = node_pos[e[1]]
            plt.plot([pos_v1[0], pos_v2[0]], [pos_v1[1], pos_v2[1]], c='black', alpha=0.5, zorder=10, lw=lw)

        (plt.gca()).spines['top'].set_visible(False)
        (plt.gca()).spines['right'].set_visible(False)
        (plt.gca()).spines['bottom'].set_visible(False)
        (plt.gca()).spines['left'].set_visible(False)
        (plt.gca()).set_yticklabels([])
        (plt.gca()).set_xticklabels([])
        plt.xticks([], [])
        plt.yticks([], [])

        # legend
        state_len = np.max([len(state) for state in model.states()])
        count_len = len(str(G.number_of_nodes()))
        for color_i, state in enumerate(model.states()):
            count = str(node_labels.count(color_i))
            state_str = state
            while len(count) < count_len:
                count = '0' + count
            while len(state_str) < state_len:
                state_str = state_str + ' '
            label = state_str + ' (' + count + ')'
            plt.plot([0, 0.0001], [0, 0.0001], label=label, c=color_palette[color_i], zorder=1,
                     lw=3)  # todo make invsible (better)

        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=False)
        plt.plot([0, 0.0001], [0, 0.0001], label=state, c='white', zorder=2, lw=5)  # this is incredibly stupid

        plt.savefig(folder + outpath.replace('NUM', str(10000 + i)), bbox_inches='tight', dpi=300)

    # from https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python

    with imageio.get_writer('output_gif/' + outpath.replace('NUM', 'movie') + rand_id + '.gif', mode='I') as writer:
        for filename in sorted(glob.glob(folder + outpath.replace('NUM', '*'))):
            image = imageio.imread(filename)
            writer.append_data(image)
