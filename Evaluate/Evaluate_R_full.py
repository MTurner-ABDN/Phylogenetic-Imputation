import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample


def get_rmse(y_true, y_pred, curr_amputed, missingness, n_traits):
    diff = y_pred - y_true
    mask = np.where(np.isnan(curr_amputed), 1, 0)  
    print("Getting_RMSE")
    if (abs((sum(sum(mask))/(len(mask)*n_traits))*100) - missingness) >= 2:
        print("Wrong missingness: ", (sum(sum(mask))/(len(mask)*n_traits))*100)
        raise ValueError("Missingness is not correct")
    print("Computed rmse")
    mask = mask.astype(bool)
    diff = diff[mask]
    rmse = np.sqrt(np.mean(diff ** 2))
    std = np.std(diff)


    return rmse, std

def evaluate():
    overlap_results = []
    running_errors = []
    for group in ["Mammals", "Plants"]:  # Each taxonomic group
        if group == "Mammals":
            n_traits = 6
        elif group == "Plants":
            n_traits = 7
        for species_num in [200, 1000]:  # Each number of simulated species
            for seed in range(1,11,1): #(1, 11, 1):  # Each simulation seed
                for scenario in ["MAR", "MNAR", "Mix"]:  # Each scenario
                    for missingness in range(5, 85, 5):  # Each missingness
                        try:
                            curr_imputed = np.genfromtxt(fname=f"../Imputation/Baselines/RPylo_outs/{group}/{scenario}/Missing_{missingness}/Full_Imputed_{species_num}_Seed_{seed}_{scenario}_Missing_{missingness}.txt",
                                delimiter="\t", skip_header=1)
                            curr_imputed = curr_imputed[:, 1:]

                            curr_GT = np.genfromtxt(fname=f"../Imputation/Baselines/RPylo_outs/{group}/{scenario}/Missing_{missingness}/Full_Ground_Truth_{species_num}_Seed_{seed}_{scenario}_Missing_{missingness}.txt",
                                delimiter="\t", skip_header=1)
                            curr_GT = curr_GT[:, 1:]

                            curr_amputed = np.genfromtxt(fname=f"../Imputation/Baselines/RPylo_outs/{group}/{scenario}/Missing_{missingness}/Full_Amputed_{species_num}_Seed_{seed}_{scenario}_Missing_{missingness}.txt",
                                delimiter="\t", skip_header=1)
                            curr_amputed = curr_amputed[:, 1:]

                            rmse, std = get_rmse(curr_GT, curr_imputed, curr_amputed, missingness, n_traits=n_traits)

                            running_errors.append({
                                "Group": group,
                                "species_num": species_num,
                                "seed": seed,
                                "scenario": scenario,
                                "missingness": missingness,
                                "rmse": rmse,
                                "std": std,
                                "error": "NA"
                            })

                        except Exception as e:
                            running_errors.append({
                                "Group": group,
                                "species_num": species_num,
                                "seed": seed,
                                "scenario": scenario,
                                "missingness": missingness,
                                "rmse": "NA",
                                "std": "NA",
                                "error": e
                            })

    results_df = pd.DataFrame(running_errors)
    results_df.to_csv(f"Results/RMSE_RPhylo.csv", index=False)


if __name__ == "__main__":
    evaluate()