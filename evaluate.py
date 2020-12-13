import run
import torch
import plot
import numpy as np
from collections import defaultdict

sent_len_label = (
    lambda i, ds: f"{run.SENT_LENGTH_BREAKPOINTS[i]} <= len < {run.SENT_LENGTH_BREAKPOINTS[i+1]}   (# samples: {len(ds[i])})"
)
sent_dataset_name2data = {
    sent_len_label(i, run.test_data_binned): (
        run.dev_data_binned[i],
        run.test_data_binned[i],
    )
    for i in range(len(run.dev_data_binned))
}


def eval_model(name, seed2model, test_data):
    test_accs = []
    for seed in ["0", "42", "420"]:
        torch.manual_seed(seed)
        criterion = torch.nn.CrossEntropyLoss()
        prep_fn = (
            run.prepare_treelstm_minibatch
            if "tree_lstm" in name
            else run.prepare_minibatch
        )
        model = seed2model[seed]
        test_loss, _, _, test_acc = run.evaluate(
            model,
            test_data,
            loss_fn=criterion,
            batch_size=run.BATCH_SIZE,
            batch_fn=run.get_minibatch,
            prep_fn=prep_fn,
        )
        test_accs.append(test_acc)
    return np.mean(test_accs), np.std(test_accs)


def load_model(init_fn, name, seed):
    results_fpath = str(run.MODELS_DIR / f"{name}_{seed}.pt")
    model = init_fn()
    model.load_state_dict(
        torch.load(results_fpath, map_location=run.device)["state_dict"]
    )
    return model


def load_seed2model(init_fn, name):
    seed2model = {
        seed: load_model(init_fn, name, seed) for seed in ["0", "42", "420"]
    }
    return seed2model


def get_dataset2model2res(
    dataset_name2data, model_whitelist=None, only_eval_own_dataset=False
):
    dataset2model2res = defaultdict(lambda: defaultdict(lambda: dict()))

    for (
        name,
        model_init_and_datasets,
    ) in run.name2model_init_and_datasets.items():
        if model_whitelist is not None:
            if name not in model_whitelist:
                continue
        print(f"evaluating {name}")
        model_init_fn, own_dataset_name = model_init_and_datasets
        seed2model = load_seed2model(model_init_fn, name)
        for dataset_name, data in run.dataset_name2data.items():
            if only_eval_own_dataset:
                if not dataset_name == own_dataset_name:
                    continue
            _, _, test_data = data
            test_acc, test_std = eval_model(name, seed2model, test_data)
            dataset2model2res[dataset_name][name] = (test_acc, test_std)
    return dataset2model2res


def plot_sent_lengths(dataset2model2res):
    model2dataset2res = defaultdict(lambda: defaultdict(lambda: dict()))
    for dataset, model2res in dataset2model2res.items():
        for model, res in model2res.items():
            model2dataset2res[model][dataset] = res
    plot.bars(
        model2dataset2res,
        "sentence_length_results",
        ylim=(0.3, 0.6),
        plot_baseline=False,
    )


"""
usage:
get evaluation results with

```
import run # make sure to set correct EXPERIMENT before importing
import utils
import evaluate
import plot
dataset_name2data = run.dataset_name2data # or evaluate.sent_dataset_name2data
fname = 'tree_lstm_intermittent_supervision_plot'
dataset2model2res = evaluate.get_dataset2model2res(dataset_name2data)
print(utils.defaultdict2dict(dataset2model2res))
plot.bars(dataset2model2res, fname)
```
"""
