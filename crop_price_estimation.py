import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample Data
prices = {
    "tomato": pd.DataFrame({'year': [2018, 2019, 2020, 2021],
                            'price': [25, 30, 28, 35]}),
    "potato": pd.DataFrame({'year': [2018, 2019, 2020, 2021],
                             'price': [18, 20, 22, 20]}),
    "onion": pd.DataFrame({'year': [2018, 2019, 2020, 2021],
                           'price': [15, 18, 12, 20]})
}

# Priors (Keeping it simple)
def log_prior(theta):
    return 0

# Simplified Likelihood (Assumes linear trend)
def log_likelihood(params, product, base_year, target_year):
    alpha, beta = params
    base_price = get_product_price(product, base_year)
    predicted_price = alpha + beta * (target_year - base_year)
    residual = predicted_price - get_product_price(product, target_year)
    return -0.5 * (np.log(2 * np.pi) + residual**2)

# Get Price (Handles missing years)
def get_product_price(product, year):
    try:
        return prices[product]["price"][prices[product]["year"] == year].values[0]
    except IndexError:
        print(f"Warning: Price data for year {year} not found. Extrapolating...")
        last_two_years = prices[product]['year'].tail(2)
        price_change = prices[product]['price'].diff().iloc[-1]
        extrapolated_price = prices[product]['price'].iloc[-1] + price_change * (year - last_two_years.iloc[-1])
        return extrapolated_price


# Simplified MCMC (Handles NaNs)
def metropolis_hastings(log_likelihood_fn, log_prior, product, base_year, target_year, n_samples):
    alpha_mean = get_product_price(product, base_year)
    beta_mean = 0
    alpha_std = 10
    beta_std = 2

    theta = np.array([alpha_mean, beta_mean])
    samples = np.zeros((n_samples, 2))
    for i in range(n_samples):
        theta_proposed = theta + np.random.normal(scale=[alpha_std, beta_std])
        likelihood = log_likelihood_fn(theta_proposed, product, base_year, target_year)
        if np.isnan(likelihood):  # Check for NaN
            continue  # Skip this sample
        acceptance_logprob = (likelihood + log_prior(theta_proposed)
                           - log_likelihood_fn(theta, product, base_year, target_year) - log_prior(theta))
        if np.log(np.random.rand()) < acceptance_logprob:
            theta = theta_proposed
        samples[i, :] = theta
    return samples

# User Input
product = input("Enter product name (tomato, potato, onion): ")
base_year = int(input("Enter base year: "))
target_years_str = input("Enter target years separated by commas (e.g., 2025, 2026, 2028): ")
target_years = [int(year) for year in target_years_str.split(',')]

# Generate samples
posterior_samples = metropolis_hastings(log_likelihood, log_prior, product, base_year, target_years[0], n_samples=1000)

# Predictions
for target_year in target_years:
    mean_alpha = np.mean(posterior_samples[:, 0])
    mean_beta = np.mean(posterior_samples[:, 1])
    predicted_price = mean_alpha + mean_beta * (target_year - base_year)
    print(f"Predicted Price for year {target_year}:", predicted_price)


# Visualization
historical_years = prices[product]['year']
historical_prices = prices[product]['price']

plt.figure(figsize=(8, 5))
plt.plot(historical_years, historical_prices, 'o-', label='Historical Prices')
plt.plot(target_years, [mean_alpha + mean_beta * (year - base_year) for year in target_years],
         '*-', label='Predictions')
plt.xlabel('Year')
plt.ylabel('Price')
plt.title(f'Price Predictions for {product}')
plt.legend()
plt.grid(True)
plt.show()
