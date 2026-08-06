import random
import numpy as np
import pandas as pd
import Preprocessing as pp

# Ensure missingness is correct
def check_missingness(amputed_subset, target_missingness, diff_threshold):
    missing_mask = np.isnan(amputed_subset) | (amputed_subset == 0)  # Get mask of amputed_subset
    indices_missingness = missing_mask.mean().mean() * 100  # compute actual missingness
    print("Achieved missingness: ", indices_missingness)

    # If the missingness is correct
    if abs(indices_missingness - target_missingness) <= diff_threshold:
        print("Achieved missingness: ", indices_missingness, "with target missing:", target_missingness)
        # print("difference missing: ", abs(indices_missingness - target_missingness), "with target missing:", target_missingness)
        return False  # Stop resampling
    else:
        print("difference missing: ", abs(indices_missingness - target_missingness))
        return True  # continue resampling

# Ensure no columns are fully missing
def fix_fully_nan_columns(amputed_df, original_df, n_restore=3, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    for col_idx in range(amputed_df.shape[1]):

        if amputed_df.iloc[:, col_idx].notna().sum() < 3:

            valid_idx = original_df.iloc[:, col_idx].dropna().index

            if len(valid_idx) == 0:
                continue

            k = min(n_restore, len(valid_idx))
            chosen = rng.choice(valid_idx, size=k, replace=False)

            col_name = amputed_df.columns[col_idx]

            amputed_df.loc[chosen, col_name] = (
                original_df.loc[chosen, col_name]
            )

    return amputed_df

# Introduce missingness into the virtual species data that is reflective of the missingness in the 'empirical' data
for group in ["Mammals", "Plants"]:
    for num in [200]:
        for seed in range(1,11,1):
            for scenario in ["MAR", "Mix", "MNAR"]:
                for missingness in range(5,85,5):
                    try:
                        print(f"Currently working on {group} {num} seed {seed}, for scenario {scenario} and missingness {missingness}%")
                        if group == "Mammals":
                            n = 7
                        else:
                            n = 8
                        full_VS = pd.read_csv(f"Simulated/with_eigens/{group}/{scenario}/Missing_{missingness}/Combined_Eigens_traits_VS_Seed_{seed}_{num}_{scenario}_Missing_{missingness}_trait_V2.txt", delimiter="\t")
                        empirical = full_VS.iloc[:num, :]
                        VS = full_VS.iloc[num:, :]
                        # Identify columns likely to be MAR or MNAR in the empirical amputed data
                        test = pp.detect_mar_mnar(empirical.iloc[:, 1:n])
                        mar_candidates = test[0]
                        print(mar_candidates)
                        mnar_candidates = test[1]
                        print(mnar_candidates)
                        mar_map = test[2]

                        # Introduce missingness into VS:
                        amputed_VS = VS.iloc[:, :n].copy()

                        redo = True
                        first_attempts_threshold = 100 # How many attempts to meet threshold 1 (thresholds refer to distance from ground truth missingness)
                        second_attempts_threshold = 100 # How many attempts to meet threshold 2
                        attempt = 0
                        while redo:
                            amputed_VS_temp = amputed_VS.copy()
                            if attempt <= first_attempts_threshold:
                                attempt = attempt + 1
                                for i in mar_candidates:
                                    # Define the candidates of MAR missingness and the conditions of MAR missingness
                                    rng = np.random.default_rng()
                                    cond = mar_map[i][0]
                                    miss_prop = test[3][i]
                                    direction = mar_map[i][1]
                                    threshold = mar_map[i][2]
                                    if np.isnan(threshold):
                                        threshold = random.uniform(0.3, 0.7)
                                    print(f"Current missing proportion for feature {i} is {miss_prop}")
                                    # Introduce MAR missingness
                                    curr_amputed_column = pp.reintroduce_mar(amputed_VS_temp.iloc[:, 1:n], i, cond, direction, threshold, missing_proportion=miss_prop, rng=rng)
                                    is_df = isinstance(curr_amputed_column, pd.DataFrame)
                                    amputed_VS_temp[i] = curr_amputed_column.values if is_df else curr_amputed_column.astype(float)


                                for i in mnar_candidates:  # Now add in estimated mnar
                                    # Define the candidates of MNAR missingness and the conditions of MNAR missingness
                                    miss_prop = test[3][i]
                                    curr_amputed_column = pp.introduce_mnar_target(amputed_VS_temp.iloc[:, 1:n], i, miss_prop)
                                    amputed_VS_temp[i] = curr_amputed_column.values
                                # Is the simulated missingness correct?
                                test_miss = check_missingness(amputed_VS_temp.iloc[:, 1:].to_numpy(), missingness, 2)
                                test_2 = pp.detect_mar_mnar(amputed_VS_temp.iloc[:, 1:n])
                                print(test_2[3])
                                print(test_miss)
                                redo = test_miss
                            # Repeat with a wider threshold
                            elif first_attempts_threshold < attempt <= second_attempts_threshold:
                                attempt = attempt + 1
                                print("Didn't ampute to the correct missingness proportion - using wider threshold")
                                for i in mar_candidates:
                                    rng = np.random.default_rng()
                                    cond = mar_map[i][0]
                                    direction = mar_map[i][1]
                                    threshold = mar_map[i][2]
                                    if np.isnan(threshold):
                                        threshold = random.uniform(0.3, 0.7)
                                    miss_prop = test[3][i]
                                    curr_amputed_column = pp.reintroduce_mar(amputed_VS_temp.iloc[:, 1:n], i, cond, direction, threshold, missing_proportion=miss_prop, rng=rng)
                                    is_df = isinstance(curr_amputed_column, pd.DataFrame)
                                    amputed_VS_temp[i] = curr_amputed_column.values if is_df else curr_amputed_column.astype(float)

                                for i in mnar_candidates:  # Now add in estimated mnar
                                    miss_prop = test[3][i]
                                    curr_amputed_column = pp.introduce_mnar_target(amputed_VS_temp.iloc[:, 1:n], i, miss_prop)
                                    amputed_VS_temp[i] = curr_amputed_column.values
                            
                                test_miss = check_missingness(amputed_VS_temp.iloc[:, 1:].to_numpy(), missingness, 4)
                                redo = test_miss
                            # Repeat with the widest threshold
                            else:
                                print("Didn't amputed to the correct missingness proportion with wider threshold - using next best")
                                for i in mar_candidates:
                                    rng = np.random.default_rng()
                                    cond = mar_map[i][0]
                                    direction = mar_map[i][1]
                                    threshold = mar_map[i][2]
                                    if np.isnan(threshold):
                                        threshold = random.uniform(0.3, 0.7)
                                    miss_prop = test[3][i]
                                    curr_amputed_column = pp.reintroduce_mar(amputed_VS_temp.iloc[:, 1:n], i, cond, direction, threshold, missing_proportion=miss_prop, rng=rng)
                                    is_df = isinstance(curr_amputed_column, pd.DataFrame)
                                    amputed_VS_temp[i] = curr_amputed_column.values if is_df else curr_amputed_column.astype(float)

                                for i in mnar_candidates:  # Now add in estimated mnar
                                    miss_prop = test[3][i]
                                    curr_amputed_column = pp.introduce_mnar_target(amputed_VS_temp.iloc[:, 1:n], i, miss_prop)
                                    amputed_VS_temp[i] = curr_amputed_column.values
                        
                                test_miss = check_missingness(amputed_VS_temp.iloc[:, 1:].to_numpy(), missingness, 4)
                                redo = False
                        amputed_VS = amputed_VS_temp.copy()
                        amputed_VS = fix_fully_nan_columns(amputed_VS, VS)
                        amputed_VS.to_csv(f"Simulated/with_eigens/{group}/{scenario}/Missing_{missingness}/Amputed_VS_{group}_Seed_{seed}_{num}_{scenario}_Missing_{missingness}.txt", sep="\t", na_rep=np.nan, index=False)
                    except Exception as e:
                         print("Error: ", e)