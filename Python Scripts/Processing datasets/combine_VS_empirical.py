import random
import numpy as np
import pandas as pd
import Preprocessing as pp

for group in ["Mammals", "Plants"]:
    # Define number of traits and trait names
    if group == "Mammals":
        n = 6
        traits = ["V2", "V3", "V4", "V5", "V6", "V7"]
    else:
        n = 7
        traits = ["V2", "V3", "V4", "V5", "V6", "V7", "V8"]
    for num in [200]:
        for seed in range(1,10,1):
            for scenario in ["MAR", "Mix", "MNAR"]:
                for missingness in range(5,85,5):
                    print(f"Working on {group} {num} seed {seed}, {scenario} missingness {missingness}%")
                    for trait in traits:
                        # Loading 3 things:
                        try:
                            # Load virtual species, RPhylopars imputed empirical data, RPhylopars imputed virtual species, and original amputed data
                            full_VS = pd.read_csv(f"Simulated/with_eigens/{group}/{scenario}/Missing_{missingness}/Combined_Eigens_traits_VS_Seed_{seed}_{num}_{scenario}_Missing_{missingness}_trait_V2.txt", delimiter="\t")
                            full_VS.index = full_VS.iloc[:, 0]
                            full_VS = full_VS.iloc[:, 1:]
                            #print(full_VS.iloc[:10, :6])

                            imputed_empirical = pd.read_csv(f"RPhylopars/Data/Full/{group}/{scenario}/Missing_{missingness}/Full_Imputed_{num}_Seed_{seed}_{scenario}_Missing_{missingness}.txt", delimiter="\t")
                            imputed_VS = pd.read_csv(f"Simulated/{group}/{scenario}/Missing_{missingness}/Rphylo_Imputed_VS_{num}_Seed_{seed}_{scenario}_Missing_{missingness}.txt", delimiter="\t")

                            amputed_VS = pd.read_csv(f"Simulated/with_eigens/{group}/{scenario}/Missing_{missingness}/Amputed_VS_{group}_Seed_{seed}_{num}_{scenario}_Missing_{missingness}.txt", delimiter="\t")
                            amputed_VS.index = amputed_VS.iloc[:, 0]
                            amputed_VS = amputed_VS.iloc[:, 1:]


                            full_eigens = full_VS.iloc[:, n:]
                            amp_emp = full_VS.iloc[:num, :n]
                            imp_emp = imputed_empirical
                            orig_VS = full_VS.iloc[num:, :n]
                            amp_VS = amputed_VS
                            imp_VS = imputed_VS
                            # First combine the empirical amputed and RPhylopars imputations
                            emp_traits = pd.concat([amp_emp, imp_emp], axis=1)
                            # And combine virtual species amputed and imputed virtual species
                            VS_traits = pd.concat([amp_VS, imp_VS], axis=1)
                          
                            # Ensure column names are congruent
                            if n == 6:
                                emp_traits.columns.values[n:] = ['R_V2', 'R_V3', 'R_V4', 'R_V5', 'R_V6', 'R_V7']
                                VS_traits.columns.values[n:] = ['R_V2', 'R_V3', 'R_V4', 'R_V5', 'R_V6', 'R_V7']
                            else:
                                emp_traits.columns.values[n:] = ['R_V2', 'R_V3', 'R_V4', 'R_V5', 'R_V6', 'R_V7', 'R_V8']
                                VS_traits.columns.values[n:] = ['R_V2', 'R_V3', 'R_V4', 'R_V5', 'R_V6', 'R_V7', 'R_V8']

                            # Combine into xs and ys
                            full_traits = pd.concat([emp_traits, VS_traits], axis=0)

                            full_data_xs = pd.concat([full_traits, full_eigens], axis=1)
                            full_data_ys = pd.concat([emp_traits, orig_VS], axis=0)

                            full_data_xs.to_csv(f"Simulated/with_eigens/{group}/{scenario}/Missing_{missingness}/Fully_concat_RPhylo_imp_VS_Seed_{seed}_{num}_{scenario}_Missing_{missingness}_trait_{trait}_XS.txt", sep="\t", na_rep="NA")
                            full_data_ys.to_csv(f"Simulated/with_eigens/{group}/{scenario}/Missing_{missingness}/Fully_concat_RPhylo_imp_VS_Seed_{seed}_{num}_{scenario}_Missing_{missingness}_trait_{trait}_YS.txt", sep="\t", na_rep="NA")

                        except Exception as e:
                            print(f"There was an error:  {e}")
