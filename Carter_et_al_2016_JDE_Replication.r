 
 
 
 # Carter et al. (2016) JDE Replication Script
 # Where and how index insurance can boost the adoption of improved agricultural technologies 
 # This script simulates technology adoption under different insurance and credit scenarios 
 
 # ===================== # 1. Load Required Libraries # ===================== 
 library(truncnorm)  # For truncated normal distributions 
 library(ggplot2)    # For plotting 
 library(reshape2)   # For data reshaping 

 # ===================== # 2. Set Seed for Reproducibility # ===================== 
 set.seed(123) 
 # ===================== # 3. Model Parameters # ===================== 
 # Technology parameters 
 
 gT <- 10                    # Output from traditional technology 
 gH <- 25                    # Output multiplier for high technology 
 K <- 10                     # Input requirement for high technology 
 # Risk parameters 
 theta_min <- 0 
 theta_max <- 2 
 sigma_c <- 0.38             # Coefficient of variation for covariant shock 
 sigma_s <- sqrt(1 - 0.8^2) * sigma_c  # Idiosyncratic shock (80% covariant) 
# Insurance parameters 
 strike_theta_c <- 1         
 # Insurance strike point 
 markup <- 0.3               # 30% markup on actuarially fair premium 
 # Financial parameters 
 pi <- 0.20                  # Risk-free opportunity cost of capital 
 collateral_levels <- seq(0, 1, by=0.1) # Collateral as % of loan value 
 
 # Preference parameters 
 CRRA <- function(x, psi) { 
    if (psi == 1) return(log(x)) 
    return((x^(1 - psi)) / (1 - psi)) } 
    
psi <- 2                    # Constant relative risk aversion 
# Population parameters 
M <- 1000                   # Number of agents 
wealth <- runif(M, 0.1 * gT, 1.5 * gT)  # Initial wealth 
risk_aversion <- runif(M, 0, 4)         # Risk aversion parameter 

# ===================== # 4. Simulation Functions # ===================== 

# Simulate shocks 
generate_shocks <- function(n) { 
    theta_c <- rtruncnorm(n, a=theta_min, b=theta_max, mean=1, sd=sigma_c) 
    theta_s <- rtruncnorm(n, a=theta_min, b=theta_max, mean=0, sd=sigma_s) 
    theta <- theta_c + theta_s 
    list(theta=theta, theta_c=theta_c,theta_s=theta_s)
} 
    
# Calculate actuarially fair premium for index insurance 
calc_premium <- function(gH, strike, sigma_c, markup) { 
    payout_prob <- pnorm(strike, mean=1, sd=sigma_c) 
    expected_payout <- gH * (strike - 1) * payout_prob 
    premium <- expected_payout * (1 + markup) 
    return(premium) } 


    # Household decision: expected utility for each scenario 
expected_utility <- function(wealth, psi, collateral, n_sim=1000) { 
        shocks <- generate_shocks(n_sim) 
        theta <- shocks$theta 
        theta_c <- shocks$theta_c 
        # Traditional technology 
        yT <- theta * gT 
        cons_T <- yT + wealth 
        EU_T <- mean(CRRA(cons_T, psi)) 
        
        # High technology without insurance 
        yH <- theta * gH repay <- (1 + pi) * K 
        cons_H <- ifelse(yH - repay >= 0, 
        yH - repay + wealth, wealth - collateral * K) 
        
        EU_H <- mean(CRRA(cons_H, psi)) # High technology with standalone 
        insurance premium <- calc_premium(gH, strike_theta_c, sigma_c, markup) 
        payout <- ifelse(theta_c < strike_theta_c, gH * (strike_theta_c - theta_c), 0) 
        yI <- yH + payout - premium 
        cons_I <- ifelse(yI - repay >= 0, yI - repay + wealth, wealth - collateral * K) 
        EU_I <- mean(CRRA(cons_I, psi)) # High technology with interlinked insurance (lower interest rate for low collateral) 
        pi_L <- ifelse(collateral < 0.5, pi * 0.5, pi) 
        repay_L <- (1 + pi_L) * K cons_L <- ifelse(yI - repay_L >= 0, yI - repay_L + wealth, 
        wealth - collateral * K) 
        EU_L <- mean(CRRA(cons_L, psi)) 
        return(c(EU_T=EU_T, EU_H=EU_H, EU_I=EU_I, EU_L=EU_L)) } 


        
        # ===================== # 5. Run Simulation for Representative Agent # ===================== 
        
        results <- data.frame() for (collateral in collateral_levels) { 
            EU <- expected_utility(wealth=0.5 * gT, psi=2, collateral=collateral) 
            results <- rbind(results, data.frame(collateral=collateral, EU_T=EU[1], EU_H=EU[2], EU_I=EU[3], EU_L=EU[4])) } 
            
        # ===================== # 6. Plot Certainty Equivalents # ===================== 
        results_long <- melt(results, id.vars="collateral", variable.name="Scenario", value.name="ExpectedUtility") 
        ggplot(results_long, aes(x=collateral, y=ExpectedUtility, color=Scenario))  + 
        geom_line(size=1.2) + labs(title="Expected Utility by Collateral and Scenario", x="Collateral (% of loan value)", y="Expected Utility (CRRA)") + 
        theme_minimal() 