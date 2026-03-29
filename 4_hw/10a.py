import numpy as np
from scipy import stats


original_sample = np.array([5, 8, 6, 12, 14, 18, 11, 6, 13, 7])
n = len(original_sample)
bootstrap_iterations = 50000
alpha = 0.05

# 1. КРИТЕРИЙ ХИ-КВАДРАТ
print("1. КРИТЕРИЙ ХИ-КВАДРАТ (РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ)")

# Для равномерного распределения: параметры a и b оцениваются из выборки
a = np.min(original_sample)
b = np.max(original_sample)

unique_values = np.unique(original_sample)
k = len(unique_values)

obs_freq = np.array([np.sum(original_sample == val) for val in unique_values])

prob_per_value = 1 / (b - a + 1)
exp_freq = np.full(k, n * prob_per_value)

# Вычисляем статистику χ² через сумму по формуле: Σ (mi - npi)² / npi
chi2_statistic = np.sum((obs_freq - exp_freq)**2 / exp_freq)

# Выводим детальный расчет каждого члена суммы
print(f"Формула: χ² = Σ (mi - npi)² / npi")
print(f"Параметры равномерного распределения: a={a}, b={b}")
print(f"Вероятность каждого значения: p = 1/({b}-{a}+1) = {prob_per_value:.4f}")
print(f"\n{'Значение':<10} {'mi':<12} {'npi':<12} {'(mi-npi)²/npi':<15}")

for i in range(k):
    term = (obs_freq[i] - exp_freq[i])**2 / exp_freq[i]
    print(f"{unique_values[i]:<10} {obs_freq[i]:<12} {exp_freq[i]:<12.4f} {term:<15.4f}")
print("-" * 50)
print(f"{'СУММА χ²':<35} {chi2_statistic:<15.4f}")

# p-value через распределение хи-квадрат (df = k-1-2 = k-3, так как оценили 2 параметра a и b)
df = k - 3
p_value_chi2 = 1 - stats.chi2.cdf(chi2_statistic, df)

print(f"\nχ²_набл = {chi2_statistic:.4f}")
print(f"p-value = ∫[{chi2_statistic:.2f}, ∞] f_χ²(x, df={df})dx = {p_value_chi2:.4f}")

# 2. КРИТЕРИЙ КОЛМОГОРОВА С БУТСТРАПОМ (50 000 ИТЕРАЦИЙ)
print("2. КРИТЕРИЙ КОЛМОГОРОВА (РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ)")

def compute_ks_stat_uniform(sample):
    """Вычисляет статистику Колмогорова для проверки равномерного распределения"""
    a = np.min(sample)
    b = np.max(sample)
    sorted_sample = np.sort(sample)
    F_emp = np.arange(1, len(sample)+1)/len(sample)
    # Теоретическая функция распределения для равномерного [a,b]: F(x) = (x-a)/(b-a)
    F_theor = (sorted_sample - a) / (b - a)
    return np.max(np.abs(F_emp - F_theor))

# Статистика для исходной выборки
original_ks = compute_ks_stat_uniform(original_sample)
print(f"\nИсходная статистика Колмогорова: {original_ks:.4f}")

# Бутстрап-ресемплинг (50 000 раз)
bootstrap_ks = np.zeros(bootstrap_iterations)
for i in range(bootstrap_iterations):
    subsample = np.random.choice(original_sample, size=n, replace=True)
    bootstrap_ks[i] = compute_ks_stat_uniform(subsample)

# Бутстрап-p-value
p_value_ks_bootstrap = np.mean(bootstrap_ks >= original_ks)

print(f"\nБутстрап-p-value для Колмогорова: {p_value_ks_bootstrap:.4f}")

# Вывод по хи-квадрату
if p_value_chi2 > alpha:
    print(f"Критерий χ²: p-value ({p_value_chi2:.4f}) > α ({alpha}) → H₀ НЕ ОТВЕРГАЕТСЯ")
    print(f"Выборка согласуется с равномерным распределением")
else:
    print(f"Критерий χ²: p-value ({p_value_chi2:.4f}) ≤ α ({alpha}) → H₀ ОТВЕРГАЕТСЯ")
    print(f"Выборка НЕ согласуется с равномерным распределением")

# Вывод по Колмогорову
if p_value_ks_bootstrap > alpha:
    print(f"\nКритерий Колмогорова: p-value ({p_value_ks_bootstrap:.4f}) > α ({alpha}) → H₀ НЕ ОТВЕРГАЕТСЯ")
    print(f"Выборка согласуется с равномерным распределением")
else:
    print(f"\nКритерий Колмогорова: p-value ({p_value_ks_bootstrap:.4f}) ≤ α ({alpha}) → H₀ ОТВЕРГАЕТСЯ")
    print(f"Выборка НЕ согласуется с равномерным распределением")