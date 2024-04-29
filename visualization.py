import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def main():
    folder = "res/"
    runs = dict()

    mask_ratios = []
    for i, f in enumerate(os.listdir(folder)):
        if not f.endswith(".npz"):
            continue
    
        terms = f.replace(".npz", "").split("_")
        obj_func = terms[0]
        mask_ratio = terms[-1]
        rank = terms[-2]
        distr = "_".join(terms[1:-2])
        runID = f"run{i}"

        data = np.load(os.path.join(folder, f))
        loss_full, loss_mask = data["loss_full"], data["loss_mask"]
        mask_ratios.append(float(mask_ratio))
        runs[runID] = {
            "config": {
                "obj_func": obj_func,
                "distr": distr,
                "rank": int(rank) if obj_func == "cp" else ast.literal_eval(rank)[0],
                "mask_ratio": float(mask_ratio),
            },
            "metric": {
                "loss_full": loss_full,
                "loss_masked": loss_mask,
            }
        }

    colors = {
        "cp_bernoulli_logit": "darkred",
        "cp_normal": "red",
        "cp_poisson_log": "pink",
        "tt_bernoulli_logit": "lightblue",
        "tt_normal": "blue",
        "tt_poisson_log": "darkblue",
    }
    styles = {
        "cp_bernoulli_logit": "solid",
        "cp_normal": "solid",
        "cp_poisson_log": "solid",
        "tt_bernoulli_logit": "dashed",
        "tt_normal": "dashed",
        "tt_poisson_log": "dashed",
    }
    mask_ratios = np.unique(mask_ratios)
    for metric in ["loss_full", "loss_masked"]:
        for mask_ratio in mask_ratios:
            plt.cla()
            out = dict()
            for k, v in runs.items():
                if v["config"]["mask_ratio"] != mask_ratio:
                    continue
                label = "_".join([v["config"]["obj_func"], v["config"]["distr"]])
                if not label in out:
                    out[label] = {
                        "x": [],
                        "y": [],
                    }
                out[label]["x"].append(v["config"]["rank"])
                out[label]["y"].append(np.log(v["metric"][metric]))

            
            for k, v in out.items():
                x, y = np.array(v["x"]), np.array(v["y"])
                ind = np.argsort(x)
                x = x[ind]
                y = y[ind]
                title = f"{metric} performance over {mask_ratio}  masking ratio"
                plt.title(title)
                plt.xlabel("Rank")
                plt.ylabel(f"Log({metric})")
                plt.plot(x, y, label=k, color=colors[k], linestyle=styles[k])
                plt.scatter(x, y, color=colors[k])
        
            # input("stop...")
            plt.legend()
            plt.savefig(f"{mask_ratio}_{metric}.png", dpi=600)


if __name__ == "__main__":
    main()