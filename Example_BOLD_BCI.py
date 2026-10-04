
import torch
import numpy as np
import pandas as pd
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.utils.sampling import draw_sobol_samples

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. PARAMETERS & CRRA ECONOMIC ENVIRONMENT (PyTorch Native)
# =====================================================================
P_BAD_WEATHER = 0.20
W_GOOD_ENDOWMENT = 100.0
W_BAD_ENDOWMENT = 30.0
GAMMA = 2.0  # CRRA risk aversion coefficient

def crra_utility(wealth, gamma=GAMMA):
    # Safe utility execution to prevent negative wealth errors during exploration
    return torch.where(wealth > 0, (wealth ** (1 - gamma)) / (1 - gamma), torch.tensor(-99.0))

# Calculate baseline utility for traditional farming without insurance/tech
u_baseline = (1 - P_BAD_WEATHER) * crra_utility(torch.tensor(W_GOOD_ENDOWMENT)) + \
             P_BAD_WEATHER * crra_utility(torch.tensor(W_BAD_ENDOWMENT))

def evaluate_bundled_contract(X):
    """
    Evaluates tensors of shape (N, 3) where columns are [Premium, Coverage, Tech_Subsidy]
    Returns an (N, 3) tensor of objectives: [Farmer Utility, Insurer Profit, Adoption Rate]
    """
    premium = X[:, 0]
    coverage = X[:, 1]
    tech_subsidy = X[:, 2]
    
    # AgTech shock mitigation function (Diminishing returns)
    tech_efficiency = tech_subsidy / (tech_subsidy + 8.0)
    w_good_tech = W_GOOD_ENDOWMENT + (3.0 * tech_efficiency)
    w_bad_tech = W_BAD_ENDOWMENT + (28.0 * tech_efficiency)
    
    # Final state-contingent wealth mappings
    w_good_final = w_good_tech - premium
    w_bad_final = w_bad_tech - premium + coverage
    
    # Objective 1: Maximize Farmer CRRA Utility
    u_farmer = (1 - P_BAD_WEATHER) * crra_utility(w_good_final) + \
               P_BAD_WEATHER * crra_utility(w_bad_final)
               
    # Objective 2: Maximize Insurer Expected Underwriting Profit
    expected_payout = P_BAD_WEATHER * coverage
    insurer_profit = premium - expected_payout - (0.40 * tech_subsidy)
    
    # Objective 3: Maximize Logistic Technology Adoption Rate
    welfare_gain = u_farmer - u_baseline
    adoption_rate = 1.0 / (1.0 + torch.exp(-2.5 * (welfare_gain + 0.15)))
    
    return torch.stack([u_farmer, insurer_profit, adoption_rate], dim=-1)

# =====================================================================
# 2. INITIALIZATION & SURROGATE FORMULATION
# =====================================================================
# Bounds for search space: [Premium (3-20), Coverage (10-70), Tech Subsidy (0-15)]
bounds = torch.tensor([[3.0, 10.0, 0.0], [20.0, 70.0, 15.0]], dtype=torch.double)

# Generate initial space-filling design using a Sobol Sequence (12 points)
train_X = draw_sobol_samples(bounds=bounds, n=12, q=1).squeeze(1)
train_Y = evaluate_bundled_contract(train_X)

# Define Reference Point for Hypervolume (worst acceptable values for objectives)
# [Farmer Utility Baseline, Insurer breaks even, Minimum acceptable adoption rate]
ref_point = torch.tensor([-0.035, -5.0, 0.10], dtype=torch.double)

# =====================================================================
# 3. MOBO OPTIMIZATION LOOP
# =====================================================================
NUM_RESTART_ITERATIONS = 10
BATCH_SIZE = 1  # Sequential evaluation (q=1)

print("Running BoTorch Multi-Objective Bayesian Optimization Loop...")

for iteration in range(NUM_RESTART_ITERATIONS):
    # Fit a multi-output SingleTask Gaussian Process to map the objectives
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    # Standardize partitioning of non-dominated space to calculate EHVI
    with torch.no_grad():
        # Filter points that beat the reference point
        feasible_mask = (train_Y > ref_point).all(dim=-1)
        feasible_Y = train_Y[feasible_mask]
        
        if len(feasible_Y) == 0:
            # Fallback if no initial points beat the reference point
            partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_Y)
        else:
            partitioning = NondominatedPartitioning(ref_point=ref_point, Y=feasible_Y)

    # Initialize the Expected Hypervolume Improvement acquisition function
    qEHVI = qExpectedHypervolumeImprovement(
        model=gp,
        ref_point=ref_point,
        partitioning=partitioning,
        sampler=None  # Automatically sets up a SobolQMCNormalSampler
    )
    
    # Optimize the acquisition function to find the next contract configuration to test
    new_X, _ = optimize_acqf(
        acq_function=qEHVI,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=512,
    )
    
    # Evaluate the proposed contract against the true economic system functions
    new_Y = evaluate_bundled_contract(new_X)
    
    # Update our historical training dataset
    train_X = torch.cat([train_X, new_X])
    train_Y = torch.cat([train_Y, new_Y])
    
    print(f"Iteration {iteration+1}/{NUM_RESTART_ITERATIONS} Completed. Extracted sample footprint size: {len(train_X)}")

# =====================================================================
# 4. RESULTS EXTRACTION & PARETO PROFILING
# =====================================================================
results = pd.DataFrame(train_X.numpy(), columns=['Premium', 'Coverage', 'TechSubsidy'])
results['Farmer_Utility'] = train_Y[:, 0].numpy()
results['Insurer_Profit'] = train_Y[:, 1].numpy()
results['Adoption_Rate'] = train_Y[:, 2].numpy()

# Identify final Pareto efficient configurations
is_pareto = np.ones(len(results), dtype=bool)
for i, row in results.iterrows():
    for j, other_row in results.iterrows():
        if i == j: continue
        if (other_row['Farmer_Utility'] >= row['Farmer_Utility'] and 
            other_row['Insurer_Profit'] >= row['Insurer_Profit'] and 
            other_row['Adoption_Rate'] >= row['Adoption_Rate']) and \
           (other_row['Farmer_Utility'] > row['Farmer_Utility'] or 
            other_row['Insurer_Profit'] > row['Insurer_Profit'] or 
            other_row['Adoption_Rate'] > row['Adoption_Rate']):
            is_pareto[i] = False
            break

pareto_contracts = results[is_pareto].sort_values(by='Adoption_Rate', ascending=False)
print("\n=== BOTORCH OPTIMIZED PARETO-EFFICIENT CONTRACT DESIGNS ===")
print(pareto_contracts.head(5).to_string(index=False))
