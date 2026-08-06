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


# This script is for training on complete cases with virtual species

def get_CCs(amputed_data, trait):
    complete_cases = []

    # Get a mask of observed/not from amputed:
    curr_feature = amputed_data[:, trait]
    curr_feat_bool = np.isnan(curr_feature)
    # Identify complete cases for the current trait
    for idx, missing in enumerate(curr_feat_bool):
        if not missing:
            complete_cases.append(amputed_data[idx, :])

    complete_cases = np.asarray(complete_cases)
    print(f"Training size for feature {trait} :  {len(complete_cases)}")

    return complete_cases


def TabPFN_Impute():
    def train_TabPFN_models(group, seed, species, missingness, scenario):
        # Defining number of traits
        if group == "Mammals":
            n = 6
        elif group == "Plants":
            n = 7

        # Get the baseline/amputed data
        Ground_Truth = np.genfromtxt(f"Simulate_VS/Simulated/with_eigens/{group}/{scenario}/Missing_{missingness}/Fully_concat_RPhylo_imp_VS_Seed_{seed}_{species}_{scenario}_Missing_{missingness}_trait_V2_YS.txt",delimiter="\t", skip_header=True)  # Get ground truths for ys
        ys = Ground_Truth[:, 1:n+1]  # Just the trait data for training

        GT_eval = np.genfromtxt(f"Data/Ground_Truth_data/{group}/Trait_data/Simulation_{group}_Seed_{seed}_{species}.txt", delimiter="\t", skip_header=True)  # Get ground truths for ys
        ys_eval = GT_eval[:, 1:n + 1]  # Just the trait data for evaluation

        Amputed_dataset = np.genfromtxt(f"Simulate_VS/Simulated/with_eigens/{group}/{scenario}/Missing_{missingness}/Fully_concat_RPhylo_imp_VS_Seed_{seed}_{species}_{scenario}_Missing_{missingness}_trait_V2_XS.txt", delimiter="\t", skip_header=True)
        Amputed_dataset = Amputed_dataset[:species, 1:n+1]  # Just the trait data


        def train_test():
            to_be_filled = Amputed_dataset[:, :n].copy()  # Amputed data copy to be filled with imputed values for each trait

            print(f'to_be_filled: {to_be_filled[:10, :n]}')
            test_mask = process_Data.get_nan_mask_matrix(Amputed_dataset[:, :n].copy())  # we get a mask from amputed data for computing RMSE

            # Identify locations of missing traits per feature (trait)
            missing_indexes = process_Data.get_missing_indexes_nan(Amputed_dataset[:, :n].copy(), group=group)  # We get the missing indexes
            if group == "Mammals":
                f1, f2, f3, f4, f5, f6 = process_Data.split_indexes_by_feature(missing_indexes, group=group)  # Split by feature
                features = [f1, f2, f3, f4, f5, f6]
            elif group == "Plants":
                f1, f2, f3, f4, f5, f6, f7 = process_Data.split_indexes_by_feature(missing_indexes, group=group)  # Split by feature
                features = [f1, f2, f3, f4, f5, f6, f7]

            to_be_filled_dict = {}
            for t in range(num_quants):
                to_be_filled_dict[f"to_be_filled_{t}"] = to_be_filled.copy()

            # First train pass
            imputed_feat_list = []
            for feat in range(0, n, 1):
                print("Num traits: ", n, "feat: ", feat)
                imputed_feat_list.append(feat)
                # Get current xs
                data = np.genfromtxt(f"Simulate_VS/Simulated/with_eigens/{group}/{scenario}/Missing_{missingness}/Fully_concat_RPhylo_imp_VS_Seed_{seed}_{species}_{scenario}_Missing_{missingness}_trait_V{feat+2}_XS.txt",
                    delimiter="\t", skip_header=True)
                data = data[:, 1:]

                # Removing RPhylopars
                traits = data[:, :n].copy()
                eigens = data[:, (2*n):].copy()
                data = np.concatenate((traits, eigens), axis = 1)

                # Generate mask
                curr_traits = data[:, :n]  # Traits for emp + VS
                curr_eigens = data[:, n:]  # Corresponding eigens
                curr_mask = process_Data.get_nan_mask_matrix(curr_traits.copy())  # Get mask
                data = np.concatenate((curr_traits, curr_mask, curr_eigens), axis=1)  # Bring together

                empirical_data = data[:species, :]  # Empirical data

                # Use previously imputed traits in future training
                if n != 0:
                    for so_far_imputed in imputed_feat_list:
                        empirical_data[:, so_far_imputed] = to_be_filled[:, so_far_imputed]

                virtual_species = data[species:, :]  # Virtual species - with missing values
  
                data = get_CCs(empirical_data, feat)  # Get complete cases for just the empirical data for current feature
                training_data = np.concatenate((data, virtual_species), axis=0)  # Training data is these plus the VS data
                X_train = np.delete(training_data.copy(), feat, 1)  # Current training data subset
                X_train = X_train[:, :500]
                y_train = np.concatenate((data[:, feat], ys[species:, feat].copy()), axis=0)

                # Define TabPFN model
                reg = TabPFNRegressor(device="cuda", n_estimators=30).create_default_for_version(ModelVersion.V2)
                # Fit to data
                reg.fit(X_train, y_train)

                # Then use this model during testing

                # Load up and preprocess the data with eigenvectors and virtual species
                test_xs = np.genfromtxt(f"Simulate_VS/Simulated/with_eigens/{group}/{scenario}/Missing_{missingness}/Fully_concat_RPhylo_imp_VS_Seed_{seed}_{species}_{scenario}_Missing_{missingness}_trait_V{feat+2}_XS.txt",
                                        delimiter="\t", skip_header=True)  # Load the amputed data and process
                test_xs = test_xs[:species, 1:]  # Just the empirical species for testing

                # Remove RPhylopars estimates
                traits_test = test_xs[:, :n].copy()  # Just the amputed traits
                eigens_test = test_xs[:, (2*n):].copy()  # And the eigenvectors
                test_xs = np.concatenate((traits_test, eigens_test), axis=1)

                # Generate missingness mask
                curr_traits = test_xs[:, :n]
                curr_eigens = test_xs[:, n:]
                curr_mask = process_Data.get_nan_mask_matrix(curr_traits.copy())
                test_xs = np.concatenate((curr_traits, curr_mask, curr_eigens), axis=1)

                # Use previously imputed traits in future testing
                if n != 0:
                    for so_far_imputed in imputed_feat_list:
                        test_xs[:, so_far_imputed] = to_be_filled[:, so_far_imputed]

                X_Test_copy = np.delete(test_xs.copy(), feat, 1)  # Remove focal feature
                print(f'Test shape: {np.shape(X_Test_copy)}, feature missing: {feat}')

                # Make predictions
                temp_predictions = reg.predict(X_Test_copy)
                quantiles = np.random.beta(5, 5, num_quants)  # Define quantiles for multiple imputation
                quantile_predictions = reg.predict(X_Test_copy, output_type="quantiles", quantiles=quantiles)  # Get quantile predictions
                quantile_predictions = np.asarray(quantile_predictions).T

                # Update dictionaries and to be filled
                quant_pred_dict = {}
                for t in range(num_quants):  # Across each quant
                    quant_pred_dict[f"Quant_pred_{t}"] = quantile_predictions[:, t]  # Save to the respective dictionary of quantile predictions for the current feature.

                for t in range(num_quants):  # For dictionary slot t
                    for s, f in features[feat]:  # Get the position to be replaced in this focal feature
                        to_be_filled_dict[f"to_be_filled_{t}"][s][f] = quant_pred_dict[f"Quant_pred_{t}"][s]

                for s, f in features[feat]:  # For each missing value from the given feature in X_test
                    to_be_filled[s][f] = temp_predictions[s]  # Used to updated xs

            temp_full_predictions = to_be_filled[:, :n].copy()
            print("temp_full_predictions: ", np.shape(temp_full_predictions))

            diff_matrix = ys_eval.copy() - temp_full_predictions
            test_mask = test_mask.astype(bool)  # Boolean mask
            masked_loss = diff_matrix[test_mask]  # Masked loss
            error = np.sqrt(np.mean(masked_loss ** 2))  # Initial RMSE estimate
            std_dev = np.std(masked_loss, ddof=1)  # STDev of residuals

            to_append = [missingness, error, std_dev]
            print(f'Model with missingness: {missingness} and training set size: {data_size - 100} has RMSE: {error}')

            return to_be_filled_dict, to_append

        filled_arrays_dictionary, curr_score = train_test()

        # Save multiply imputed datasets
        for t in range(num_quants):
            np.savetxt(fname="Results/{group}/{scenario}/Missing_{missingness}/Integrated_TT_CC_VS_{group}_{seed}_{species}_{scenario}_Missing_{missingness}_imputation_{t}.csv",
                X=np.asarray(filled_arrays_dictionary[f"to_be_filled_{t}"]), delimiter="\t")

        return curr_score  # Return temp RMSE and SD

    # Loop over datasets
    for group in ["Mammals", "Plants"]:
        for scenario in ["MAR", "Mix", "MNAR"]:
            num_quants = 10
            for data_size in [200, 1000]:
                scores = []
                for seed in range(1, 11, 1):
                    for missing in range(5, 85, 5):
                        print(f"Run for {group}, with {data_size} species seed {seed}. Missing conditions: {scenario}, {missing}%")
                        try:
                            curr_score = train_TabPFN_models(group=group, seed=seed, species=data_size,
                                                             missingness=missing, scenario=scenario)
                            scores.append({
                                "Group": group,
                                "Scenario": scenario,
                                "Dataset Size": data_size,
                                "Seed": seed,
                                "Missingness": missing,
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
                                "Current Score": "NA",
                                "Error?": e
                            })


if __name__ == '__main__':
    TabPFN_Impute()