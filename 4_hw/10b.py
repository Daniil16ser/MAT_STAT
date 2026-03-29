import numpy as np
from scipy import stats

original_sample = np.array([5, 8, 6, 12, 14, 18, 11, 6, 13, 7])
n = len(original_sample)
bootstrap_iterations = 50000
alpha = 0.05

# 1. КРИТЕРИЙ ХИ-КВАДРАТ (РУЧНОЙ РАСЧЕТ ЧЕРЕЗ СУММУ)

print("1. КРИТЕРИЙ ХИ-КВАДРАТ (ФОРМУЛА СУММЫ)")


mu = np.mean(original_sample)
sigma = np.std(original_sample, ddof=1)

unique_values = np.unique(original_sample)
k = len(unique_values)

obs_freq = np.array([np.sum(original_sample == val) for val in unique_values])

exp_freq = []
for val in unique_values:
    prob = stats.norm.cdf(val + 0.5, mu, sigma) - stats.norm.cdf(val - 0.5, mu, sigma)
    exp_freq.append(n * prob)
exp_freq = np.array(exp_freq)

# Вычисляем статистику χ² через сумму по формуле: Σ (mi - npi)² / npi
chi2_statistic = np.sum((obs_freq - exp_freq)**2 / exp_freq)

for i in range(k):
    term = (obs_freq[i] - exp_freq[i])**2 / exp_freq[i]
    print(f"{unique_values[i]:<10} {obs_freq[i]:<12} {exp_freq[i]:<12.4f} {term:<15.4f}")
print("-" * 50)
print(f"{'СУММА χ²':<35} {chi2_statistic:<15.4f}")

# p-value через распределение хи-квадрат (df = k-3, так как оценили 2 параметра)
df = k - 3
p_value_chi2 = 1 - stats.chi2.cdf(chi2_statistic, df)

print(f"\nχ²_набл = {chi2_statistic:.4f}")
print(f"p-value = ∫[{chi2_statistic:.2f}, ∞] f_χ²(x, df={df})dx = {p_value_chi2:.4f}")


# 2. КРИТЕРИЙ КОЛМОГОРОВА С БУТСТРАПОМ (50 000 ИТЕРАЦИЙ)

print("\n" + "=" * 60)
print("2. КРИТЕРИЙ КОЛМОГОРОВА (БУТСТРАП)")
print("=" * 60)

def compute_ks_stat(sample):
    """Вычисляет статистику Колмогорова для проверки нормальности"""
    mu = np.mean(sample)
    sigma = np.std(sample, ddof=1)
    sorted_sample = np.sort(sample)
    F_emp = np.arange(1, len(sample)+1)/len(sample)
    F_theor = stats.norm.cdf(sorted_sample, mu, sigma)
    return np.max(np.abs(F_emp - F_theor))

# Статистика для исходной выборки
original_ks = compute_ks_stat(original_sample)
print(f"\nИсходная статистика Колмогорова: {original_ks:.4f}")

# Бутстрап-ресемплинг (50 000 раз)
bootstrap_ks = np.zeros(bootstrap_iterations)
for i in range(bootstrap_iterations):
    subsample = np.random.choice(original_sample, size=n, replace=True)
    bootstrap_ks[i] = compute_ks_stat(subsample)

# Бутстрап-p-value
p_value_ks_bootstrap = np.mean(bootstrap_ks >= original_ks)

print(f"\nБутстрап-p-value для Колмогорова: {p_value_ks_bootstrap:.4f}")

print("\n" + "=" * 60)
print("ВЫВОД ПО ОБАИМ КРИТЕРИЯМ")
print("=" * 60)

# Вывод по хи-квадрату
if p_value_chi2 > alpha:
    print(f"Критерий χ²: p-value ({p_value_chi2:.4f}) > α ({alpha}) → H₀ НЕ ОТВЕРГАЕТСЯ")
else:
    print(f"Критерий χ²: p-value ({p_value_chi2:.4f}) ≤ α ({alpha}) → H₀ ОТВЕРГАЕТСЯ")

# Вывод по Колмогорову
if p_value_ks_bootstrap > alpha:
    print(f"Критерий Колмогорова: p-value ({p_value_ks_bootstrap:.4f}) > α ({alpha}) → H₀ НЕ ОТВЕРГАЕТСЯ")
else:
    print(f"Критерий Колмогорова: p-value ({p_value_ks_bootstrap:.4f}) ≤ α ({alpha}) → H₀ ОТВЕРГАЕТСЯ")