from methabc.simulate import log_model_abc
from methabc.utils import write_config, repo_dir

import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default=os.path.join(repo_dir, "tmp", "synthetic"),
                        help="output path without extension, writes <output>.csv and <output>_params.csv")
    parser.add_argument("--meth_rate", type=float, default=np.log10(0.0022), help="log10")
    parser.add_argument("--demeth_rate", type=float, default=np.log10(0.0018), help="log10")
    parser.add_argument("--init_migration_rate", type=float, default=np.log10(0.009), help="log10")
    parser.add_argument("--mu_driver_birth", type=float, default=np.log10(0.0001), help="log10")
    parser.add_argument("--s_driver_birth", type=float, default=0.1)
    parser.add_argument("-l", "--left_demes", type=int, default=4,
                        help="number of sampled glands on side A")
    parser.add_argument("-r", "--right_demes", type=int, default=4,
                        help="number of sampled glands on side B")
    parser.add_argument("-K", "--carrying_capacity", type=int, default=100,
                        help="deme carrying capacity")
    parser.add_argument("-t", "--turnover", type=float, default=0,
                        help="turnover after expansion, as a fraction of expansion time")
    parser.add_argument("-s", "--seed", type=int, default=1)
    args = parser.parse_args()

    log_params = {
        'meth_rate': args.meth_rate,
        'demeth_rate': args.demeth_rate,
        'init_migration_rate': args.init_migration_rate,
        'mu_driver_birth': args.mu_driver_birth,
        's_driver_birth': args.s_driver_birth,
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    config_template_path = write_config(
        params={
            "deme_carrying_capacity": args.carrying_capacity,
            "turnover": args.turnover,
        },
        output_dir=out_dir,
    )

    np.random.seed(args.seed)
    print("Generating synthetic data...")
    df = log_model_abc(
        log_params,
        left_demes=args.left_demes,
        right_demes=args.right_demes,
        config_template_path=config_template_path,
    )["data"]
    os.remove(config_template_path)
    if df is None:
        print("Simulation failed, try another seed.")
        return

    # same gland order as in total_distance
    df = df.sort_values(by=['Side', 'OriginTime']).reset_index(drop=True)
    names = [f"SA{i + 1}" for i in range(args.left_demes)] + [f"SB{i + 1}" for i in range(args.right_demes)]
    data = pd.DataFrame(np.stack(df.AverageArray.values).T, columns=names)
    data.index = [f"locus{i}" for i in range(data.shape[0])]
    data.index.name = "Labels"

    data.to_csv(args.output + ".csv")
    pd.DataFrame([log_params]).to_csv(args.output + "_params.csv", index=False)
    print(f"Done! Saved {args.output}.csv and {args.output}_params.csv")


if __name__ == "__main__":
    main()
