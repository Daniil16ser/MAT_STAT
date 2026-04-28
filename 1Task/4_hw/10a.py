import numpy as np
from scipy import stats

original_sample = np.array([5, 8, 6, 12, 14, 18, 11, 6, 13, 7])
bootstrap_iterations = 50000
subsample_size = 100
alpha = 0.05

print("1. КРИТЕРИЙ ХИ-КВАДРАТ (РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ)")
a = np.min(original_sample)
b = np.max(original_sample)
unique_values = np.unique(original_sample)
k = len(unique_values)
obs_freq = np.array([np.sum(original_sample == val) for val in unique_values])
prob_per_value = 1 / (b - a + 1)
exp_freq = np.full(k, len(original_sample) * prob_per_value)
chi2_statistic = np.sum((obs_freq - exp_freq)**2 / exp_freq)

print(f"Формула: χ² = Σ (mi - npi)² / npi")
print(f"Параметры: a={a}, b={b}")
print(f"Вероятность: p = {prob_per_value:.4f}")
print(f"\n{'Значение':<10} {'mi':<12} {'npi':<12} {'(mi-npi)²/npi':<15}")
for i in range(k):
    term = (obs_freq[i] - exp_freq[i])**2 / exp_freq[i]
    print(f"{unique_values[i]:<10} {obs_freq[i]:<12} {exp_freq[i]:<12.4f} {term:<15.4f}")
print("-" * 50)
print(f"{'СУММА χ²':<35} {chi2_statistic:<15.4f}")

df = k - 3
p_value_chi2 = 1 - stats.chi2.cdf(chi2_statistic, df)
print(f"\nχ²_набл = {chi2_statistic:.4f}")
print(f"p-value = {p_value_chi2:.4f}")

print("\n2. КРИТЕРИЙ КОЛМОГОРОВА (БУТСТРАП 50000 ВЫБОРОК ПО 100 ЭЛЕМЕНТОВ)")
def compute_ks_stat_uniform(sample):
    a = np.min(sample)
    b = np.max(sample)
    sorted_sample = np.sort(sample)
    F_emp = np.arange(1, len(sample)+1)/len(sample)
    F_theor = (sorted_sample - a) / (b - a)
    return np.max(np.abs(F_emp - F_theor))*10

original_ks = compute_ks_stat_uniform(original_sample)
print(f"\nИсходная статистика Колмогорова: {original_ks:.4f}")

bootstrap_ks = np.zeros(bootstrap_iterations)
for i in range(bootstrap_iterations):
    subsample = np.random.choice(original_sample, size=subsample_size, replace=True)
    bootstrap_ks[i] = compute_ks_stat_uniform(subsample)

p_value_ks_bootstrap = np.mean(bootstrap_ks >= original_ks)
print(f"\nБутстрап-p-value: {p_value_ks_bootstrap:.4f}")

if p_value_chi2 > alpha:
    print(f"\nКритерий χ²: p-value ({p_value_chi2:.4f}) > α ({alpha}) → H₀ НЕ ОТВЕРГАЕТСЯ")
else:
    print(f"\nКритерий χ²: p-value ({p_value_chi2:.4f}) ≤ α ({alpha}) → H₀ ОТВЕРГАЕТСЯ")

if p_value_ks_bootstrap > alpha:
    print(f"Критерий Колмогорова: p-value ({p_value_ks_bootstrap:.4f}) > α ({alpha}) → H₀ НЕ ОТВЕРГАЕТСЯ")
else:
    print(f"Критерий Колмогорова: p-value ({p_value_ks_bootstrap:.4f}) ≤ α ({alpha}) → H₀ ОТВЕРГАЕТСЯ")