import numpy as np
import pandas as pd


for group in ["Mammals", "Plants"]:
    if group == "Mammals":
        traits = ["V2", "V3", "V4", "V5", "V6", "V7"]
        n = 6
    else:
        traits = ["V2", "V3", "V4", "V5", "V6", "V7", "V8"]
        n = 7
    for species in [200,1000]:
        for seed in range(1,11,1):
            for scenario in ["MAR", "MNAR", "Mix"]:
                for missingness in range(5,85,5):
                    try:
                        Rphylo_imp = pd.read_csv(f"../Imputation/Baselines/RPhylo_outs/{group}/{scenario}/Missing_{missingness}/Full_Imputed_{species}_Seed_{seed}_{scenario}_Missing_{missingness}.txt", sep="\t")
                        Amputed = np.genfromtxt(f"../Imputation/Baselines/RPhylo_outs/{group}/{scenario}/Missing_{missingness}/Full_Amputed_{species}_Seed_{seed}_{scenario}_Missing_{missingness}.txt", delimiter="\t", skip_header=1, usecols=range(1,n+1,1))
                        positions = [i for i, x in enumerate(traits) if x not in list(Rphylo_imp)]  # Check if there is anywhere Rphylos didnt impute
                        tree_tips = list(Rphylo_imp.index)
                        Rphylo_imp = np.asarray(Rphylo_imp)
                        for position in positions:  # For each of these
                            temp_left = Rphylo_imp[:, :position]  # Take the columns before this position
                            temp_right = Rphylo_imp[:, position:]  # And the columns after this position
                            updated_Rphylo_imp = np.concatenate([temp_left, Amputed[:, position].reshape(species,1), temp_right], axis=1)  # And isnert the original amputed data
                            Rphylo_imp = updated_Rphylo_imp.copy()  # Then update RPhylo imp

                        tree_tips_col = np.array(tree_tips).reshape(-1, 1)  # Keep tree tips as a vector

                        for trait in traits:
                            full_eigens = np.genfromtxt(f"Traits_eigens/{group}/{scenario}/Missing_{missingness}/Combined_Eigens_traits_Seed_{seed}_{species}_{scenario}_Missing_{missingness}_trait_{trait}.txt", skip_header=1)
                            trait_data = np.asarray(full_eigens[:, 1:n+1])  # Get trait data
                            eigens = np.asarray(full_eigens[:, n+1:])  # Get eigencector data
                            concat_rphylo = np.concatenate([tree_tips_col,trait_data,Rphylo_imp,eigens], axis=1)  # concat the species names, trait data, updated rphylopars, and then eigens
                            concat_rphylo = np.array(concat_rphylo, dtype=object)
                            np.savetxt(f"Ensembled_RPhylo_amputed/{group}/{scenario}/Missing_{missingness}/Fully_Combined_Seed_{seed}_{species}_{scenario}_Missing_{missingness}_trait_{trait}.txt", concat_rphylo, delimiter="\t", fmt="%s")
                    except:
                        print("Skip")
