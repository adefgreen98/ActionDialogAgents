import os

import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from tqdm import tqdm

name_map = {
        'linear': 'Action-Matrix',
        'linear-concat': 'Concat-linear',
        'fcn': 'Concat-multi-layer',
        'linear-infonce': 'Action-Matrix',
        'linear-concat-infonce': 'Concat-linear',
        'fcn-infonce': 'Concat-multi-layer',
        'baseline-random': 'baseline-random',
        'baseline-similarity': 'baseline-similarity'
    }



def old_training():
    df = pandas.read_csv('new-vect-results/statistics/hold-out/infonce/results.csv')
    df['vect_model'] = df['vect_model'].map(lambda name: name_map[name])

    seaborn.set(font_scale=1.5)

    g = seaborn.catplot(x='extractor_model', y='accuracy', hue='vect_model',
                        data=df[~df['vect_model'].isin({'baseline-random', 'baseline-similarity'})],
                        col="hold_out_procedure", kind="bar", height=10, aspect=.75)

    # random baseline

    df.loc[df['vect_model'] == 'baseline-random', 'accuracy'] = 0.25  # TODO remove correction

    [ax.axhline(
        df[(df['vect_model'] == 'baseline-random') &
                            (df['hold_out_procedure'] == ax.get_title().split()[-1])
                            ]['accuracy'].mean(),
        linestyle='--', color='lightgray', alpha=0.7, linewidth=2)
        for i, ax in enumerate(g.axes.flatten())]

    # similarity baseline
    [ax.axhline(
        df[(df['vect_model'] == 'baseline-similarity') &
                            (df['hold_out_procedure'] == ax.get_title().split()[-1])
                            ]['accuracy'].mean(),
        linestyle='--', color='lightcoral', alpha=0.7, linewidth=2)
        for i, ax in enumerate(g.axes.flatten())]

    for i, ax, title in zip(range(len(g.axes.flatten())), g.axes.flatten(), ['Samples', 'Objects', 'Scenes']):
        ax.set_title("Hold-out " + title)

    g.legend.remove()
    plt.legend(loc='upper right', facecolor='white')
    plt.show()


def training_appendix():
    df = pandas.read_csv('new-vect-results/statistics/hold-out/infonce/results.csv')
    df['training'] = 'contrastive'
    df.loc[df['vect_model'] == 'baseline-random', 'accuracy'] = 0.25  # TODO remove correction

    df2 = pandas.read_csv('new-vect-results/statistics/hold-out/l2/results.csv')
    df2['training'] = 'L2'
    df2.loc[df2['vect_model'] == 'baseline-random', 'accuracy'] = 0.25  # TODO remove correction

    baseline_df = [df2, df]

    full_df = pandas.concat(baseline_df)

    full_df['vect_model'] = full_df['vect_model'].map(lambda name: name_map[name])

    seaborn.set(font_scale=1.5)

    g = seaborn.catplot(x='extractor_model', y='accuracy', hue='vect_model', data=full_df[~full_df['vect_model'].isin({'baseline-random', 'baseline-similarity'})],
                        col="hold_out_procedure", row='training', kind="bar", height=10, aspect=.75)

    # random baseline
    [ax.axhline(
        baseline_df[i // 3][(baseline_df[i // 3]['vect_model'] == 'baseline-random') &
                            (baseline_df[i // 3]['hold_out_procedure'] == ax.get_title().split()[-1])
                            ]['accuracy'].mean(),
        linestyle='--', color='lightgray', alpha=0.7, linewidth=2)
        for i, ax in enumerate(g.axes.flatten())]

    # similarity baseline
    [ax.axhline(
        baseline_df[i // 3][(baseline_df[i // 3]['vect_model'] == 'baseline-similarity') &
                            (baseline_df[i // 3]['hold_out_procedure'] == ax.get_title().split()[-1])
                            ]['accuracy'].mean(),
        linestyle='--', color='lightcoral', alpha=0.7, linewidth=2)
        for i, ax in enumerate(g.axes.flatten())]

    for i, ax, title in zip(range(len(g.axes.flatten())), g.axes.flatten(), ['Samples', 'Objects', 'Scenes'] * 2):
        ax.set_title("Hold-out " + title + f" ({str('L2') if i // 3 == 0 else str('contrastive')})")

    g.legend.remove()
    plt.legend(loc='upper right', facecolor='white')
    plt.show()


def training(pth='cs-size-results/infonce-embedding/outputs.csv', show=True):
    full_df = pandas.read_csv(pth)
    full_df = full_df[lambda df: df['contrast_type'] == 'CS4']
    full_df.drop(columns=['contrast_type'], inplace=True)
    full_df['vect_model'] = full_df['vect_model'].map(lambda name: name.replace('-infonce', ''))

    indexer = ['extractor_model', 'vect_model', 'hold_out_procedure']

    full_df['accuracy'] = (full_df['gt_action'] == full_df['pred_action']).astype(int)
    acc = full_df.groupby(indexer).sum()['accuracy']

    acc = acc / full_df.groupby(indexer).count()['accuracy']

    acc_df = full_df.groupby(indexer).count()
    acc_df['accuracy'] = acc

    acc_df = acc_df.reset_index()[indexer + ['accuracy']]

    # add similarity baseline
    similarity_df = full_df[['extractor_model', 'hold_out_procedure', 'gt_action', 'visual_distractor_action']]
    similarity_df['accuracy'] = (similarity_df['gt_action'] == similarity_df['visual_distractor_action']).astype(int)
    similarity_df = similarity_df.groupby(['extractor_model', 'hold_out_procedure']).mean().reset_index()[['extractor_model', 'hold_out_procedure', 'accuracy']]

    similarity_df['vect_model'] = 'baseline-similarity'

    # after-image only baseline
    after_only_baseline = pandas.DataFrame({
        'extractor_model': ['moca-rn', 'clip-rn', 'moca-rn', 'clip-rn', 'moca-rn', 'clip-rn'],
        'hold_out_procedure': ['object_name', 'object_name', 'samples', 'samples', 'scene', 'scene'],
        'accuracy': [0.2676, 0.2841, 0.2556, 0.2624, 0.1981, 0.1348]
    })

    ### DROPPING SAMPLE SPLIT ###
    acc_df.drop(index=acc_df[lambda d: d['hold_out_procedure'] == 'samples'].index, inplace=True)
    after_only_baseline.drop(index=after_only_baseline[lambda d: d['hold_out_procedure'] == 'samples'].index, inplace=True)
    similarity_df.drop(index=similarity_df[lambda d: d['hold_out_procedure'] == 'samples'].index, inplace=True)
    ############################

    seaborn.set(font_scale=1.5)
    g = seaborn.catplot(x='extractor_model', y='accuracy', hue='vect_model',
                        data=acc_df, col="hold_out_procedure", ci=95,
                        kind="bar",
                        height=10, aspect=.75, sharex=False)

    g.set_titles("Holding out {col_name}")

    added_to_legend = False

    # GRAPHING PARAMS
    hline_thickness = 3
    vonly_color = 'coral'
    af_color = 'purple'
    rand_color = 'lightgray'
    human_color = 'red'
    alpha = .9

    plt.ylim((0,.9))

    # error bars
    sdf = full_df[['extractor_model', 'vect_model', 'hold_out_procedure', 'iteration', 'accuracy']]
    err_df = sdf.groupby(['extractor_model', 'vect_model', 'hold_out_procedure', 'iteration']).mean().reset_index().groupby(['extractor_model', 'vect_model', 'hold_out_procedure']).std()['accuracy']

    for i, ax in enumerate(g.axes.flatten()):
        title = ax.get_title()
        if 'object_name' in title:
            ax.set_title('Holding out objects')
        h_o_proc = title.split()[-1]
        print(h_o_proc)

        labels = [tx.get_text() for tx in ax.get_xticklabels()]
        pos = {name: i for i, name in enumerate(labels)}

        # split [0,1] to have correct positions for hlines
        k = 1 / len(labels)
        eps = 1e-3
        rngs = [(k * i + eps, k * (i + 1) - eps) for i in range(len(labels))]

        for name in labels:
            sim_val = similarity_df[
                lambda df: (df['extractor_model'] == name)
                           & (df['hold_out_procedure'] == h_o_proc)
            ]['accuracy'].item()

            af_val = after_only_baseline[
                lambda df: (df['extractor_model'] == name)
                           & (df['hold_out_procedure'] == h_o_proc)
            ]['accuracy'].item()

            if added_to_legend or i < (len(g.axes.flatten()) - 1):
                # similarity
                ax.axhline(sim_val, xmin=rngs[pos[name]][0], xmax=rngs[pos[name]][1],
                           linestyle='--', color=vonly_color, alpha=alpha, linewidth=hline_thickness)
                # after-image-only
                ax.axhline(af_val, xmin=rngs[pos[name]][0], xmax=rngs[pos[name]][1],
                           linestyle='--', color=af_color, alpha=alpha, linewidth=hline_thickness)

            else:
                # similarity
                ax.axhline(sim_val, xmin=rngs[pos[name]][0], xmax=rngs[pos[name]][1],
                           linestyle='--', color=vonly_color, alpha=alpha, linewidth=hline_thickness, label='visual-only')
                # after-image-only
                ax.axhline(af_val, xmin=rngs[pos[name]][0], xmax=rngs[pos[name]][1],
                           linestyle='--', color=af_color, alpha=alpha, linewidth=hline_thickness, label='action-name')
                added_to_legend = True
        ax.set_xlabel("")
        ax.set_xticklabels(["CLIP", "MOCA"])

        # random
        ax.axhline(0.25, linestyle='--', color=rand_color, alpha=alpha, linewidth=hline_thickness, label='random')

        # human
        ax.axhline(0.831, linestyle='-', color=human_color, alpha=0.5, linewidth=hline_thickness - 1, label='human')


    g.legend.remove()
    lg = plt.legend(loc='lower right', facecolor='white')
    if show:
        lg.set_draggable(True)
        plt.show()

    return g


def error_analysis(pth='cs-size-results/infonce_3/outputs.csv', show=True):
    seaborn.set(font_scale=1.5)

    full_df = pandas.read_csv(pth)
    full_df = full_df[lambda df: df['contrast_type'] == 'CS4'].drop(columns=['contrast_type'])
    full_df['vect_model'] = full_df['vect_model'].map(lambda el: el.replace('-infonce', ''))

    ### DROPPING SAMPLES SPLIT ###
    full_df.drop(index=full_df[lambda d: d['hold_out_procedure'] == 'samples'].index, inplace=True)
    ##############################

    indexer = ['extractor_model', 'vect_model', 'hold_out_procedure']

    ac_map = {name: i for i, name in enumerate(sorted(list(set(full_df['gt_action']))))}
    n = len(ac_map)

    def rework_conf_mat(cm):
        cm = cm / cm.sum(-1).reshape(n, 1)
        cm[numpy.isnan(cm)] = 0
        return cm

    def get_conf_mat(row):
        total_conf_mat = numpy.zeros((n, n))
        obj_conf_mat = {
            obj: numpy.zeros((n, n)) for obj in set(full_df['object_name'])
        }

        for row in row.to_dict('records'):
            total_conf_mat[ac_map[row['gt_action']], ac_map[row['pred_action']]] += 1
            obj_conf_mat[row['object_name']][ac_map[row['gt_action']], ac_map[row['pred_action']]] += 1

        total_conf_mat = rework_conf_mat(total_conf_mat)
        obj_conf_mat = {k: rework_conf_mat(obj_conf_mat[k]) for k in obj_conf_mat}
        return total_conf_mat, obj_conf_mat

    tb = full_df.groupby(indexer).apply(lambda r: get_conf_mat(r))

    # Computes action accuracies
    action_accuracies = tb.reset_index()
    action_accuracies[0] = action_accuracies[0].map(lambda el: {k: el[0][ac_map[k], ac_map[k]] for k in ac_map})
    for k in ac_map:
        action_accuracies[k] = action_accuracies[0].map(lambda d: d[k])
    action_accuracies.drop(columns=[0], inplace=True)
    action_accuracies.to_csv(Path(*(Path(pth).parts[:-1])) / 'action_accuracies.csv', index=False)
    action_accuracies.to_excel(Path(*(Path(pth).parts[:-1])) / 'action_accuracies.xlsx', index=False)

    # Plots
    cmap = {
        'Action-Matrix': 'Blues',
        'Concat-Multi': 'Greens',
        'Concat-Linear': 'Oranges'
    }

    rows = sorted(set(full_df['hold_out_procedure']))
    cols = sorted(set(full_df['vect_model']))

    col_idx = {k: i+1 for i, k in enumerate(cols)}
    row_idx = {k: i+1 for i, k in enumerate(rows)}

    fig = {
        k: plt.figure(figsize=(18, 12)) for k in set(full_df['extractor_model'])
    }

    for idx, el in zip(tb.index, tb):
        curr_r = row_idx[idx[-1]]
        curr_c = col_idx[idx[1]]

        print(idx, (curr_r, curr_c))

        ax = fig[idx[0]].add_subplot(len(rows), len(cols), (curr_r - 1) * len(cols) + curr_c)

        cm, obj_cm = el
        seaborn.heatmap(ax=ax, data=cm, cbar=True,
                        vmin=0.0, vmax=1.0,
                        cmap=cmap[idx[1]], square=True,
                        xticklabels=list(ac_map.keys()), yticklabels=list(ac_map.keys()))

        ax.set_title(f"{idx[1]}")

        if curr_c == 1:
            ax.set_ylabel(f"Holding out {idx[-1].split('_')[0]}")

    for k in fig:
        fig[k].suptitle("Confusion Matrices for " + k.split("-")[0].upper())
        fig[k].tight_layout()

    if show:
        plt.show()

    if not show:
        for k in fig:
            fig[k].savefig(Path(*(Path(pth).parts[:-1])) / (k + '-conf.png'))


    return fig


def plot_dataset(pth='new-dataset/data-improved-descriptions/dataset_cs_scene_object_nopt_augmented_recep.csv',
                 show=True):
    df = pandas.read_csv(pth, index_col=0)
    df.drop(index=df[(df['action_name'] == 'cook') | (df['distractor0_action_name'] == 'cook') | (
                df['distractor1_action_name'] == 'cook') | ((df['distractor2_action_name'] == 'cook'))].index,
            inplace=True)
    obj_indexer = ['object_split', 'object_name', 'action_name', 'after_image_path']
    obj_split_df = df[obj_indexer]
    obj_split_df.loc[lambda d: d['object_split'] != 'test', 'object_split'] = 'Seen'
    obj_split_df.loc[lambda d: d['object_split'] == 'test', 'object_split'] = 'Unseen'
    scene_indexer = ['scene_split', 'scene', 'action_name', 'after_image_path']
    scene_split_df = df[scene_indexer]
    scene_split_df.loc[lambda d: d['scene_split'] != 'test', 'scene_split'] = 'Seen'
    scene_split_df.loc[lambda d: d['scene_split'] == 'test', 'scene_split'] = 'Unseen'
    obj_split_df.rename(columns={'object_split': 'split'}, inplace=True)
    scene_split_df.rename(columns={'scene_split': 'split'}, inplace=True)
    obj_split_df['split_name'] = 'object'
    scene_split_df['split_name'] = 'scene'
    co_df = obj_split_df.groupby(['split_name', 'split', 'action_name']).count().reset_index()
    cs_df = scene_split_df.groupby(['split_name', 'split', 'action_name']).count().reset_index()
    split_df = pandas.concat([co_df, cs_df])
    split_df.rename(columns={'after_image_path': 'Samples'}, inplace=True)
    seaborn.set(font_scale=2)
    g = seaborn.catplot(data=split_df, x='split', y='Samples', row='split_name', hue='action_name', kind='bar');
    g.set_titles("{row_name} split");
    g.legend.set_draggable(True);
    g.legend.set_title('Action');

    if show:
        plt.show()


def action_accuracies(pth='cs-size-results/infonce-embedding/outputs.csv'):
    df = pandas.read_csv(pth)
    df['vect_model'] = df['vect_model'].map(lambda el: el.replace('-infonce', ''))
    resdf = df[['hold_out_procedure', 'extractor_model', 'vect_model', 'iteration', 'object_name', 'pred_action', 'gt_action']]
    resdf['accuracy'] = (resdf.pred_action == resdf.gt_action).astype(int)

    # TODO choose best model
    resdf = resdf[(resdf.extractor_model == 'moca-rn') & (resdf.vect_model == 'Concat-Multi')]
    tmp = resdf[lambda d: d['hold_out_procedure'] == 'object_name'].groupby(['gt_action', 'object_name', ]).mean()['accuracy']
    tmp = tmp.to_frame()
    tmp = tmp.reset_index().pivot(index='object_name', columns='gt_action')
    print(tmp)


def nearest_neighbor_analysis(neighbors=5, norm='cosine'):

    def wrap_nearest_neighbor_analysis(pth):
        import torch
        from multimodal.vect_models import LinearVectorTransform
        vec_size = 1024 if "clip" in str(pth) else 147
        model = LinearVectorTransform(set(range(12)), vec_size, device='cuda')
        model.load_state_dict(torch.load(pth))

        # retrieve actions dict
        outs = pandas.read_csv(Path(*Path(pth).parts[:-3], '0_@5', 'outputs.csv'))
        ac2id = dict(list({(row['gt_action'], row['gt_id']) for i, row in outs.iterrows()}))
        id2ac = {v: k for k, v in ac2id.items()}

        def normalize(t):
            return t / t.norm()

        similarities = {k: {} for k in ac2id}
        for curr_ac in ac2id:
            for i, other_ac in id2ac.items():
                if curr_ac != other_ac:
                    if norm == 'l2':
                        val = torch.sqrt(torch.sum((normalize(model.weights[ac2id[curr_ac]]) - normalize(model.weights[ac2id[other_ac]]))**2)).item()
                    elif norm == 'l1':
                        val = torch.sum(torch.abs(normalize(model.weights[ac2id[curr_ac]]) - normalize(model.weights[ac2id[other_ac]]))).item()
                    elif norm == 'cosine':
                        val = torch.cosine_similarity(
                            model.weights[ac2id[curr_ac]].flatten(),
                            model.weights[ac2id[other_ac]].flatten(),
                            dim=0
                        ).squeeze().item()
                    similarities[curr_ac][other_ac] = round(val, 4)
        similarities = {k: sorted(list(v.items()), key=lambda el: el[1], reverse=(norm == 'cosine')) for k, v in similarities.items()}
        for k in sorted(similarities):
            print(k, ":", similarities[k])

        return similarities

    sim = []
    for nr in {1, 4, 7, 10, 13}:
        print(f"============ {nr} ============")
        res = wrap_nearest_neighbor_analysis(f'saved-models-exp/models/Action-Matrix-infonce+moca-rn/object_name_{nr}_@5.pth')
        sim.append(res)
        print()

    print()
    for s, nr in zip(sim, {1, 4, 7, 10, 13}):
        print(f"============ {nr} ============")
        for k in sorted(s):
            print(k, ":", [el[0] for el in s[k]][:neighbors])


def wrap_nearest_samples(pth, k=4, n=100):
    df = pandas.read_csv(pth)

    res_pth = Path(Path(pth).parent, 'images')
    os.makedirs(res_pth, exist_ok=True)

    for gt, preds in tqdm(zip(df['pth'][:n], df['neighbors'][:n]), desc='Preparing neighbor images...', total=n):

        preds = [eval(el) for el in preds[1:-1].split(',')]

        fig, axes = plt.subplots(1, k+1)

        axes[0].imshow(Image.open(gt))
        axes[0].set_title('Ground truth')
        axes[0].set_axis_off()

        for prd, ax in zip(preds[:k], axes[1:]):
            ax.imshow(Image.open(prd))
            ax.set_axis_off()

        plt.tight_layout()

        imsave_pth = Path(res_pth, *(Path(gt).parts[-3:]))

        os.makedirs(imsave_pth.parent, exist_ok=True)
        plt.savefig(imsave_pth)
        plt.close()


def nearest_samples(pth, k=4, n=100):
    for r, d, fnames in os.walk(pth):
        for fname in fnames:
            if fname == 'ranking.csv':
                print(f"----- Neighbors for {Path(*[el.replace('_neighbors', '') for el in Path(r).parts[-2:]])} -----")
                wrap_nearest_samples(os.path.join(r, fname), k, n)


if __name__ == '__main__':

    # training('cs-size-results/infonce-embedding/outputs.csv')

    # error_analysis('cs-size-results/alternative-obj-split/outputs.csv')

    # plot_dataset('new-dataset/data-improved-descriptions/alternative_obj_split_dataset.csv')

    # nearest_neighbor_analysis(norm='l1')

    nearest_samples("experiment-nearest-neighbors/models")


    pass
