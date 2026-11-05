import numpy as np
import pandas as pd
import random as rn

from torch import dtype
import Preprocessing as process_Data



def Ampute_data(missingness, scenario, num, seed, group):
    MS = missingness / 100
    data_orig = pd.read_csv(f"../Simulation/Data/{group}/Simulation_{group}_Seed_{seed}_{num}.txt", delimiter="\t", dtype=np.float32)  # Ground truth trait data
    data = data_orig.copy().to_numpy()
    num_traits = data.shape[1]

    resample = True

    while resample:
        if scenario == "MAR":
            xs = process_Data.introduce_missingness_fast_v2(dataset=data.copy(), missing_proportion=MS, mar_fraction=1.0)
        elif scenario == "MNAR":
            xs = process_Data.introduce_missingness_fast_v2(dataset=data.copy(), missing_proportion=MS, mar_fraction=0.0)
        elif scenario == "Mix":
            features = [i for i in range(0, num_traits, 1)]
            mar_features = set(rn.sample(features, rn.randrange(1, len(features), 1)))
            mnar_features = set(features) - mar_features
            mar_frac = rn.randrange(3, 7, 1) / 10
            xs = process_Data.introduce_missingness_fast_v2(dataset=data.copy(), missing_proportion=MS,
                                                         mar_fraction=mar_frac, mar_features=list(mar_features),
                                                         mnar_features=list(mnar_features))
        mask = np.where(np.isnan(xs[:, :num_traits]), 1.0, 0.0)  # Mask of missing trait positions

        actual_missing = sum(sum(mask)) / (mask.shape[0] * num_traits) * 100
        if missingness - 2 < actual_missing < missingness + 2:
            print(f"Data has {actual_missing:.2f} missing values and can continue.")
            resample = False
        else:
            print(f"Data has {actual_missing:.2f} % missing values and needs to be resampled. Needs {missingness} %.")

    np.savetxt(fname=f"{group}/{scenario}/Missing_{missingness}/Amputed_{group}_Seed_{seed}_{num}_{scenario}_missing_{missingness}.txt", X = xs, delimiter="\t", fmt='%s')

for group in ["Mammals", "Plants"]:
    for num in [200,1000]:
        for seed in range(1,11,1):
            for scenario in ["MAR", "Mix", "MNAR"]:
                for missing in range(5,85,5):
                    Ampute_data(missingness=missing, scenario=scenario, num=num, seed=seed, group=group)


