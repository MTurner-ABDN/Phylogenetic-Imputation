# Import
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
import statsmodels.api as sm
from scipy.stats import norm


def get_rmse(y_true, y_pred, idx, n_traits, curr_amputed):
    diff = y_true - y_pred
    mask = np.where(np.isnan(curr_amputed), 1, 0)

    if idx == 0:
        print("Mask missingness: ", (sum(sum(mask))/(len(mask)*n_traits))*100)
    mask = mask.astype(bool)

    diff = diff[mask]
    rmse = np.sqrt(np.mean(diff ** 2))

    return rmse


def get_ground_truth(method, group, species, seed, scenario, missingness, repeat, n_traits):
    Ground_Truth = pd.read_csv(f"../Imputation/TabPFN_Benchmark/Results/{group}/{scenario}/Missing_{missingness}/Species_{species}_Seed_{seed}_Missingness_{missingness}_Ground_Truth_{repeat}.csv", delimiter="\t", header=None)  # Load current GT 1000 sample (needs the response variable)
    Ground_Truth = Ground_Truth.to_numpy()
    Ground_Truth = Ground_Truth[:, :n_traits]  # We only want trait data and response

    curr_GT = Ground_Truth[:, :n_traits].copy()
    GT_coefficients = []  # Empty list of coefficients

    response_GT = np.loadtxt(f"../Imputation/TabPFN_Benchmark/Results/{group}/{scenario}/Missing_{missingness}/Species_{species}_Seed_{seed}_Missingness_{missingness}_GT_response_{repeat}.txt", delimiter="\t")
    response = response_GT[:, 0]

    Ground_Truth = np.hstack([response[:, np.newaxis], Ground_Truth])  # Concat response onto GT trait data

    X_train = Ground_Truth[:, 1:]  # Xs are traits
    y_train = Ground_Truth[:, 0]  # Ys are response

    model = sm.OLS(y_train, X_train).fit()
    coef = model.params
    GT_coefficients.append(coef)

    GT_coefficients = np.array(GT_coefficients)

    return np.array(GT_coefficients), response, curr_GT


def get_imputed(method, group, species, seed, scenario, missingness, repeat, n_traits, response, curr_GT):
    curr_imputed_coefficients = []  # Empty list of coefficients
    curr_imputed_coeff_SEs = []  # Empty list of coefficients
    curr_imputed_intercept = []
    curr_rmses = []  # Get the 10 rmse's

    for i in range(10): 
        curr_imputed = pd.read_csv(f"../Imputation/TabPFN_Benchmark/Results/{group}/{scenario}/Missing_{missingness}/{group}_{seed}_{species}_{scenario}_Missing_{missingness}_repeat_{repeat}_imputation_{i}.csv",
                delimiter="\t", header=None)
        curr_imputed = curr_imputed.to_numpy()
        curr_amputed = pd.read_csv(f"../Imputation/TabPFN_Benchmark/Results/{group}/{scenario}/Missing_{missingness}/Species_{species}_Seed_{seed}_Missingness_{missingness}_amputed_{i+1}.csv",
                delimiter="\t", header=None) 
        curr_amputed = curr_amputed.to_numpy()
        curr_amputed = curr_amputed[:, :n_traits]
        curr_imputed = curr_imputed[:, :n_traits]  # We only want trait data and response
        temp_rmse = get_rmse(curr_GT, curr_imputed, i, n_traits, curr_amputed)
        curr_rmses.append(temp_rmse)  # For each of the 10 MI, append the RMSE.

        curr_imputed = np.hstack([response[:, np.newaxis], curr_imputed]) # concat response for these 10k to the curr_imputed data
        X_train = curr_imputed[:, 1:]  # Xs are trait data
        y_train = curr_imputed[:, 0]  # ys are corresponding response data

        model = sm.OLS(y_train, X_train).fit()
        coef = model.params
        se = model.bse
        curr_imputed_coefficients.append(coef)  # The n coefficients
        curr_imputed_coeff_SEs.append(se)  # And the n associated 

    # Pooling
    Q = np.vstack(curr_imputed_coefficients)
    SE = np.vstack(curr_imputed_coeff_SEs)
    m = Q.shape[0]
    Q_bar = Q.mean(axis=0)             # mean coeffiicient estimate
    W = (SE**2).mean(axis=0)           # within-imputation variance  - (average variance)
    B = Q.var(axis=0, ddof=1)          # between-imputation variance  - variance of the actual coefficient values
    T = W + (1 + 1/m) * B             # total variance
    SE_pool = np.sqrt(T)

    print("Within variance:", W)
    print("Between variance:", B)
    print("Total variance:", T)
    print("Pooled SE:", np.sqrt(T))

    mean_rmses = np.mean(curr_rmses)  # Mean RMSE across all 10 MI datasets
    std_rmses = np.std(curr_rmses)  # SD of RMSE across all 10 MI datasets

    return np.array(Q_bar), np.array(SE_pool), W, B, mean_rmses, std_rmses, curr_rmses


def compare_coefficients(Ground_truth, Curr_imputed, IMP_SE, W, B, m=10):
    n_coefs = Ground_truth.shape[1]
    covered = []
    biases = []
    confidence_intervals = []

    print("GT: ", Ground_truth)
    print("Curr_imputed: ", Curr_imputed)

    # Compaute coefficient-wise degrees of freedom for Rubin's rules

    with np.errstate(divide="ignore", invalid="ignore"):
        r = (1 + 1/m) * B / W
        df = (m - 1) * (1 + 1/r)**2
        df = np.where(B == 0, np.inf, df)

    t_critical = stats.t.ppf(0.975, df)

    print("Degrees of Freedom per coefficient: ", df)
    print("t-critical values per coefficient: ", t_critical)

    for i in range(n_coefs):
        curr_GT_coef = Ground_truth[:, i]
        curr_imp_coef = Curr_imputed[i]

        difference = curr_GT_coef - curr_imp_coef
        biases.append(difference)
        
        lower = curr_imp_coef - (t_critical[i] * IMP_SE[i])
        upper = curr_imp_coef + (t_critical[i] * IMP_SE[i])

        confidence_intervals.append(t_critical[i] * IMP_SE[i])

        covered.append(lower <= curr_GT_coef <= upper)

    print("Coverage: ", covered)
    print("biases: ", biases)


    return biases, covered, confidence_intervals


def evaluate():
    bias_data = []
    individual_errors = []
    curr_rmses = []
    for method in ["TabPFN"]:
        for group in ["Mammals", "Plants"]:
            if group == "Mammals":
                n_traits = 6
            elif group == "Plants":
                n_traits = 7
            for species_num in [200, 1000]:
                for seed in range(1,11,1):  # Each simulation seed
                    for scenario in ["MAR", "MNAR", "Mix"]:  # Each scenario
                        for missingness in range(5,85,5):  # Each missingness
                            for repeat in range(1,11,1):  # Each train/test split
                                try:
                                    # Get the current ground truth data
                                    GT_coefs, curr_response, curr_GT = get_ground_truth(method=method, group=group, species=species_num, seed=seed, scenario=scenario, missingness=missingness, repeat=repeat, n_traits=n_traits)
                                    # Get the current imputed data & RMSE
                                    IMP_coefs, IMP_SEs, within_var, between_var, mean_rmses, stds, individual_rmses = get_imputed(method=method, group=group, species=species_num, seed=seed, scenario=scenario, missingness=missingness, repeat=repeat, n_traits=n_traits, response=curr_response, curr_GT=curr_GT)
                                    # Compare coefficients.
                                    bias, coverage, CIs = compare_coefficients(GT_coefs, IMP_coefs, IMP_SEs, within_var, between_var)

                                    for MI_iteration in range(10):
                                        individual_errors.append({
                                            "method": method,
                                            "group": group,
                                            "species_num": species_num,
                                            "seed": seed,
                                            "scenario": scenario,
                                            "missingness": missingness,
                                            "repeat": repeat,
                                            "Iteration": MI_iteration,
                                            "RMSE": individual_rmses[MI_iteration],
                                            "Error": "NA"
                                        })

                                    for coef_idx in range(n_traits):
                                        bias_data.append({
                                            "method": method,
                                            "group": group,
                                            "species_num": species_num,
                                            "seed": seed,
                                            "scenario": scenario,
                                            "missingness": missingness,
                                            "repeat": repeat,
                                            "Coefficent_ID": coef_idx + 1,
                                            "Bias": bias[coef_idx],
                                            "Covered": coverage[coef_idx],
                                            "Imputed CI": CIs[coef_idx]
                                        })

                                except Exception as e:

                                    for MI_iteration in range(10):
                                        individual_errors.append({
                                            "method": method,
                                            "group": group,
                                            "species_num": species_num,
                                            "seed": seed,
                                            "scenario": scenario,
                                            "missingness": missingness,
                                            "repeat": repeat,  #
                                            "Iteration": MI_iteration,
                                            "RMSE": "NA",
                                            "Error": e
                                        })

                                    for coef_idx in range(n_traits):
                                        bias_data.append({
                                            "method": method,
                                            "group": group,
                                            "species_num": species_num,
                                            "seed": seed,
                                            "scenario": scenario,
                                            "missingness": missingness,
                                            "repeat": repeat,
                                            "Coefficent_ID": coef_idx + 1,
                                            "Bias": "NA",
                                            "Covered": "NA",
                                            "Imputed CI": "NA"
                                        })

    bias_results = pd.DataFrame(bias_data)
    bias_results.to_csv(f"Results/Coverage_TabPFN_bench.csv", index=False)
    individual_rmses = pd.DataFrame(individual_errors)
    individual_rmses.to_csv(f"Results/RMSE_TabPFN_bench.csv", index=False)



if __name__ == "__main__":
    evaluate()