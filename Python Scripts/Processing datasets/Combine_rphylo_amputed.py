import numpy as np
import pandas as pd

# L
for group in ["Mammals", "Plants"]:
    # Trait names
    if group == "Mammals":
        traits = ["V2", "V3", "V4", "V5", "V6", "V7"]
        n = 6
    else:
        traits = ["V2", "V3", "V4", "V5", "V6", "V7", "V8"]
        n = 7
    for species in [200, 1000]:
        for seed in range(1,11,1):
            for scenario in ["MAR", "MNAR", "Mix"]:
                for missingness in range(5,85,5):
                    try:
                        # Load Rphylopars imputation and amputed dataset
                        Rphylo_imp = pd.read_csv(f"Baselines/RPhylopars/Data/Full/{group}/{scenario}/Missing_{missingness}/Full_Imputed_{species}_Seed_{seed}_{scenario}_Missing_{missingness}.txt", sep="\t")
                        Amputed = np.genfromtxt(f"Baselines/RPhylopars/Data/Full/{group}/{scenario}/Missing_{missingness}/Full_Amputed_{species}_Seed_{seed}_{scenario}_Missing_{missingness}.txt", delimiter="\t", skip_header=1, usecols=range(1,n+1,1))
                        # Identify traits imputed by RRhylopars - where it couldn't impute, replace with originally amputed data
                        positions = [i for i, x in enumerate(traits) if x not in list(Rphylo_imp)]
                        tree_tips = list(Rphylo_imp.index)
                        Rphylo_imp = np.asarray(Rphylo_imp)
                        for position in positions:
                            temp_left = Rphylo_imp[:, :position]
                            temp_right = Rphylo_imp[:, position:]
                            updated_Rphylo_imp = np.concatenate([temp_left, Amputed[:, position].reshape(species,1), temp_right], axis=1)
                            Rphylo_imp = updated_Rphylo_imp.copy()

                        tree_tips_col = np.array(tree_tips).reshape(-1, 1)

                        # combine the combined amputed/Rphyloaprs dataset with trait-specific eigenvector data
                        for trait in traits:
                            full_eigens = np.genfromtxt(f"Data/Full_eigens/{group}/{scenario}/Missing_{missingness}/Combined_Eigens_traits_Seed_{seed}_{species}_{scenario}_Missing_{missingness}_trait_{trait}.txt", skip_header=1)
                            trait_data = np.asarray(full_eigens[:, 1:n+1])
                            eigens = np.asarray(full_eigens[:, n+1:])
                            concat_rphylo = np.concatenate([tree_tips_col,trait_data,Rphylo_imp,eigens], axis=1)
                            concat_rphylo = np.array(concat_rphylo, dtype=object)
                            np.savetxt(f"Data/Combined_RPhylo_Amputed/{group}/{scenario}/Missing_{missingness}/Fully_Combined_Seed_{seed}_{species}_{scenario}_Missing_{missingness}_trait_{trait}.txt", concat_rphylo, delimiter="\t", fmt="%s")
                    except:
                        print("Skip")
