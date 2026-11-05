from sklearn.model_selection import train_test_split
import pandas as pd
import math

import numpy as np
import pickle
import torch
import gc
import os

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
import Preprocessing as process_Data


def check_missingness(amputed_subset, target_missingness):
    missing_mask = np.isnan(amputed_subset) | (amputed_subset == 0)  # Get mask of amputed_subset
    indices_missingness = missing_mask.mean().mean() * 100  # compute actual missingness

    # If the missingness is correct
    if abs(indices_missingness - target_missingness) <= 2:
        print("Achieved missingness: ", indices_missingness, "with target missing:", target_missingness)
        #print("difference missing: ", abs(indices_missingness - target_missingness), "with target missing:", target_missingness)
        return False  # Stop resampling
    else:
        print("difference missing: ", abs(indices_missingness - target_missingness))
        return True  # continue resampling

def get_CCs(amputed_data, trait):
    complete_cases = []

    # Get a mask of observed/not from amputed:
    curr_feature = amputed_data[:, trait]
    curr_feat_bool = np.isnan(curr_feature)

    for idx, missing in enumerate(curr_feat_bool):
        if not missing:
            complete_cases.append(amputed_data[idx, :])

    complete_cases = np.asarray(complete_cases)
    print(f"Training size for feature {trait} :  {len(complete_cases)}")

    return complete_cases


def TabPFN_Impute():
    def train_TabPFN_models(group, seed, species, missingness, scenario, repeats):
        if group == "Mammals":
            n = 6
        elif group == "Plants":
            n = 7

        # Get the baseline/amputed data as per usual
        num_test = 100
        resample = True

        # Get the corresponding round truth as usual
        Ground_Truth = np.genfromtxt(f"../Simulation/Data/{group}/Simulation_{group}_Seed_{seed}_{species}.txt", delimiter="\t", skip_header=True)  # Get ground truths for ys
        ys = Ground_Truth[:, 1:]  # Just the traits

        GT_response = np.genfromtxt(f"../Data/GT_response/{group}/Simulation_{group}_Seed_{seed}_{species}_trait_data_with_response.txt", delimiter="\t")  # Get ground truths for ys

        # Get train/test indices as usual
        checking_indices = np.genfromtxt(f"../Data/Traits_eigens/{group}/{scenario}/Missing_{missingness}/Combined_Eigens_traits_Seed_{seed}_{species}_{scenario}_Missing_{missingness}_trait_V2.txt",
            delimiter="\t", skip_header=True)  # Amputed dataset
        checking_indices = checking_indices[:, 1:]

        checking_indices = checking_indices[:, :n]

        while resample:  # This lets us get the train and test indices
            indices = np.random.permutation(species)  # Randomly permute number of species
            test_indices = indices[:num_test]  # Select the first 100 of these
            train_indices = indices[num_test:]  # remaining for training
            curr_subset = checking_indices[test_indices, :n]  # Get hypothetical test subset
            resample = check_missingness(curr_subset, missingness)  # If the missingness is consistent then stop resampling

        print("Outside resample")

        # Save corresponding GT and amputed datasets using these identified indices
        np.savetxt(fname=f"Results/{group}/{scenario}/Missing_{missingness}/Species_{species}_Seed_{seed}_Missingness_{missingness}_Ground_Truth_{repeats}.csv",
            X=ys[test_indices, :], delimiter="\t")  # current ground truth subset for test indices

        np.savetxt(fname=f"Results/{group}/{scenario}/Missing_{missingness}/Species_{species}_Seed_{seed}_Missingness_{missingness}_amputed_{repeats}.csv",
            X=checking_indices[test_indices, :], delimiter="\t")  # current ground truth subset for test indices

        np.savetxt(fname=f"Results/{group}/{scenario}/Missing_{missingness}/Species_{species}_Seed_{seed}_Missingness_{missingness}_GT_response_{repeats}.txt",
            X=GT_response[test_indices, :], delimiter="\t") 



        def train_test():
            ys = Ground_Truth[:, 1:]
            to_be_filled_test = curr_subset.copy()  # These are the test indices
            to_be_filled_train = checking_indices[train_indices, :n].copy()  # These are the test indices

            # Get missing features in test set
            print(f'to_be_filled: {to_be_filled_test[:10, :n]}')
            test_mask = process_Data.get_nan_mask_matrix(curr_subset[:, :n].copy())  # we get a mask from amputed data for computing RMSE
            missing_indexes_test = process_Data.get_missing_indexes_nan(curr_subset[:, :n].copy(), group=group)  # We get the missing indexes
            if group == "Mammals":
                f1, f2, f3, f4, f5, f6 = process_Data.split_indexes_by_feature(missing_indexes_test, group=group)  # Split by feature
                features_test = [f1, f2, f3, f4, f5, f6]
            elif group == "Plants":
                f1, f2, f3, f4, f5, f6, f7 = process_Data.split_indexes_by_feature(missing_indexes_test, group=group)  # Split by feature
                features_test = [f1, f2, f3, f4, f5, f6, f7]

            # Get missing features in train set
            train_mask = process_Data.get_nan_mask_matrix(checking_indices[train_indices, :n].copy())  # we get a mask from amputed data for computing RMSE
            missing_indexes_train = process_Data.get_missing_indexes_nan(checking_indices[train_indices, :n].copy(), group=group)  # We get the missing indexes
            if group == "Mammals":
                f1_tr, f2_tr, f3_tr, f4_tr, f5_tr, f6_tr = process_Data.split_indexes_by_feature(missing_indexes_train, group=group)  # Split by feature
                features_train = [f1_tr, f2_tr, f3_tr, f4_tr, f5_tr, f6_tr ]
            elif group == "Plants":
                f1_tr, f2_tr, f3_tr, f4_tr, f5_tr, f6_tr, f7_tr = process_Data.split_indexes_by_feature(missing_indexes_train, group=group)  # Split by feature
                features_train = [f1_tr, f2_tr, f3_tr, f4_tr, f5_tr, f6_tr, f7_tr]

            to_be_filled_dict = {}
            for t in range(num_quants):
                to_be_filled_dict[f"to_be_filled_{t}"] = to_be_filled_test.copy()

            # First train the first pass

            imputed_feat_list = []

            for feat in range(0, n, 1):
                print("Num traits: ", n, "feat: ", feat)
                imputed_feat_list.append(feat)

                # Get current xs - traits and mask are consistent in feature loop - eigenvectors differ
                xs = np.genfromtxt(f"../Data/Traits_eigens/{group}/{scenario}/Missing_{missingness}/Combined_Eigens_traits_Seed_{seed}_{species}_{scenario}_Missing_{missingness}_trait_V{feat + 2}.txt",
                delimiter="\t", skip_header=True)
                xs = xs[:, 1:]
                # Get mask for xs
                curr_traits = xs[:, :n]
                #curr_traits = np.nan_to_num(curr_traits, nan=0.0)
                curr_eigens = xs[:, n:]
            
                train_eigens = curr_eigens[train_indices, :]  # Eigen cols for train
                useable_columns = np.var(train_eigens, axis=0) > 1e-12  # Check variance is not 0
                curr_eigens = curr_eigens[:, useable_columns]  # Only use non 0 variance columns

                curr_mask = process_Data.get_nan_mask_matrix(curr_traits.copy())

                xs = np.concatenate((curr_traits, curr_mask, curr_eigens), axis=1)
                xs = xs[:, :500]

                X_Train = xs[train_indices, :]
                test_xs = xs[test_indices, :]

                if n != 0:  # Excluding the first pass
                    for so_far_imputed in imputed_feat_list:
                        test_xs[:, so_far_imputed] = to_be_filled_test[:, so_far_imputed]
                        X_Train[:, so_far_imputed] = to_be_filled_train[:, so_far_imputed]

                y_Train = ys[train_indices, :n]

                X_train = np.delete(X_Train.copy(), feat, 1)
                y_train = y_Train[:, feat].copy()  # This worked fine

                reg = TabPFNRegressor(device="cuda", n_estimators=30).create_default_for_version(ModelVersion.V2)
                reg.fit(X_train, y_train)

                # Train prediction and to_be_filled update
                train_predictions = reg.predict(X_train)
                for s, f in features_train[feat]:  # For each missing value from the given feature in X_test
                    to_be_filled_train[s][f] = train_predictions[s]  ######### Here we update test_xs instead so that the next iteration has access to these imputation


                # Then use this model during testing
                X_Test_copy = np.delete(test_xs.copy(), feat, 1)  # Remove focal feature
                print(f'Test shape: {np.shape(X_Test_copy)}, feature missing: {feat}')

                # Make predictions
                temp_predictions = reg.predict(X_Test_copy)  # Make predictions
                quantiles = np.random.beta(5, 5, num_quants)  # Define quantiles
                quantile_predictions = reg.predict(X_Test_copy, output_type="quantiles", quantiles=quantiles)  # Get quantile predictions
                quantile_predictions = np.asarray(quantile_predictions).T

                # Update dictionaries and to be filled
                quant_pred_dict = {}
                for t in range(num_quants):  # Increment t
                    quant_pred_dict[f"Quant_pred_{t}"] = quantile_predictions[:, t]  # Dictionary of quantile predictions for current feature.

                for t in range(num_quants):  # For dictionary slot t
                    for s, f in features_test[feat]:  # Get the position to be replaced in this focal feature
                        to_be_filled_dict[f"to_be_filled_{t}"][s][f] = quant_pred_dict[f"Quant_pred_{t}"][s]

                for s, f in features_test[feat]:  # For each missing value from the given feature in X_test
                    to_be_filled_test[s][f] = temp_predictions[s]  

                # This leaves us having trained the model once and got the imputed values for the same feature - we then update the training data (lines 85/86) and the test data (lines 103/104)
                # We then update the dictionary of the quantile predictions as well as the to be filled array - these will be updated for the next feature after its training/testing phase

            temp_full_predictions = to_be_filled_test[:, :n].copy()
            ys = ys[test_indices, :]
            diff_matrix = ys.copy() - temp_full_predictions
            test_mask = test_mask.astype(bool)  # Boolean mask
            masked_loss = diff_matrix[test_mask]  # Masked loss
            error = np.sqrt(np.mean(masked_loss ** 2))  # RMSE
            std_dev = np.std(masked_loss, ddof=1)  # STDev of residuals

            to_append = [missingness, error, std_dev]
            print(f'Model with missingness: {missingness} and training set size: {data_size - 100} has RMSE: {error}')

            return to_be_filled_dict, to_append

        filled_arrays_dictionary, curr_score = train_test()

        for t in range(num_quants):
            np.savetxt(fname=f"Results/{group}/{scenario}/Missing_{missingness}/{group}_{seed}_{species}_{scenario}_Missing_{missingness}_repeat_{repeats}_imputation_{t}.csv", X=np.asarray(filled_arrays_dictionary[f"to_be_filled_{t}"]), delimiter="\t")

        return curr_score

    scores = []

    for group in ["Mammals", "Plants"]:
        for scenario in ["MAR", "Mix", "MNAR"]:
            num_quants = 10
            for data_size in [200, 1000]:
                for seed in range(1, 11, 1):
                    for missing in range(5, 85, 5):
                        for repeats in range(1, 11, 1):  # Train/test repeats
                            print(f"Run for {group}, with {data_size} species seed {seed}. Missing conditions: {scenario}, {missing}%")
                            try:
                                curr_score = train_TabPFN_models(group=group, seed=seed, species=data_size,
                                                                 missingness=missing, scenario=scenario,
                                                                 repeats=repeats)
                                scores.append({
                                    "Group": group,
                                    "Scenario": scenario,
                                    "Dataset Size": data_size,
                                    "Seed": seed,
                                    "Missingness": missing,
                                    "Repeat": repeats,
                                    "Current Score": curr_score,
                                    "Error?": "NA"
                                })
                            except Exception as e:
                                scores.append({
                                    "Group": group,
                                    "Scenario": scenario,
                                    "Dataset Size": data_size,
                                    "Seed": seed,
                                    "Missingness": missing,
                                    "Repeat": repeats,
                                    "Current Score": "NA",
                                    "Error?": e
                                })

    scores = pd.DataFrame(scores)
    scores.to_csv(f"Results/{group}/Initial_scores_{scenario}_{data_size}.csv", index=False)

if __name__ == '__main__':
    TabPFN_Impute()