import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")

colors = sns.color_palette("hls")
color_hline = colors[0]
colors = colors[1:]


def bars(
    dataset2model2res,
    fname,
    label_datasets=True,
    ylim=(0.36, 0.5),
    plot_baseline=True,
):
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    n_datasets = len(dataset2model2res.keys())
    width = 0.8
    models = []
    baseline = None
    for model2res in dataset2model2res.values():
        models.extend(list(model2res.keys()))
        if "deep_cbow" in model2res:
            baseline = model2res["deep_cbow"][0]
    models = sorted(list(set(models)))
    linewidth = width / n_datasets
    i = 0

    if plot_baseline:
        ax.hlines(
            baseline,
            -0.5,
            len(models) - 0.5,
            colors=color_hline,
            label="deep_cbow baseline",
        )
        ax.margins(0, ax.margins()[1])

    for dsi, (dataset, model2res) in enumerate(dataset2model2res.items()):
        n = len(model2res.keys())
        current_models = sorted(list(model2res.keys()))
        for mi, model in enumerate((current_models)):
            res = model2res[model]
            if label_datasets:
                x = mi - width * 0.5 + (dsi + 0.5) * linewidth
            else:
                x = i
            label = dataset if mi == 0 and label_datasets else None
            ax.bar(
                x,
                res[0],
                yerr=res[1],
                width=linewidth,
                color=colors[dsi if label_datasets else 0],
                label=label,
            )
            ax.set_ylabel("test accuracy")
            i += 1
    ax.set_ylim(*ylim)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=13)
    plt.legend()
    plt.savefig(f"./plots/{fname}.png", bbox_inches="tight")
