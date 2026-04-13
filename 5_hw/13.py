import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

# Параметры задачи
df1 = 999  # Европейцы
df2 = 138  # Египтяне
alpha = 0.05

# Критические значения F для двустороннего теста
F_crit_upper = f.ppf(1 - alpha/2, df1, df2)  # верхний хвост
F_crit_lower = f.ppf(alpha/2, df1, df2)      # нижний хвост

# Значения отношения дисперсий (theta = σ²₁ / σ²₂)
theta_range = np.linspace(0.8, 2.0, 500)

# Расчёт мощности для двустороннего теста
# Мощность = P(F > F_crit_upper | θ) + P(F < F_crit_lower | θ)
# При θ ≠ 1 наблюдаемая статистика F_obs ~ θ * F(df1, df2)
# Поэтому:
# P(F_obs > F_crit_upper) = P(θ * F > F_crit_upper) = P(F > F_crit_upper / θ)
# P(F_obs < F_crit_lower) = P(θ * F < F_crit_lower) = P(F < F_crit_lower / θ)
power = (1 - f.cdf(F_crit_upper / theta_range, df1, df2) + 
         f.cdf(F_crit_lower / theta_range, df1, df2))

# Точки для нашей задачи
theta_len = (6.161 / 5.722) ** 2
theta_wid = (5.055 / 4.612) ** 2

# Мощность в этих точках
power_len = (1 - f.cdf(F_crit_upper / theta_len, df1, df2) + 
             f.cdf(F_crit_lower / theta_len, df1, df2))
power_wid = (1 - f.cdf(F_crit_upper / theta_wid, df1, df2) + 
             f.cdf(F_crit_lower / theta_wid, df1, df2))

# Вывод численных результатов
print(f"Длина черепа:")
print(f"  θ = {theta_len:.4f}")
print(f"  F-статистика = {theta_len:.4f}")
print(f"  Критическое значение F (верхнее) = {F_crit_upper:.4f}")
print(f"  Мощность = {power_len:.4f}\n")

print(f"Ширина черепа:")
print(f"  θ = {theta_wid:.4f}")
print(f"  F-статистика = {theta_wid:.4f}")
print(f"  Критическое значение F (верхнее) = {F_crit_upper:.4f}")
print(f"  Мощность = {power_wid:.4f}")

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(theta_range, power, 'b-', linewidth=2, 
         label=f'Мощность критерия (df₁={df1}, df₂={df2})')
plt.axvline(x=1.0, color='gray', linestyle='--', label='H₀: θ = 1')
plt.axhline(y=alpha, color='red', linestyle=':', 
            label=f'Уровень значимости α = {alpha}')

# Отмечаем точки из задачи
plt.plot(theta_len, power_len, 'ro', markersize=8, 
         label=f'Длина (θ = {theta_len:.3f}, мощность = {power_len:.3f})')
plt.plot(theta_wid, power_wid, 'go', markersize=8, 
         label=f'Ширина (θ = {theta_wid:.3f}, мощность = {power_wid:.3f})')

# Подписываем значения мощности рядом с точками
plt.annotate(f'{power_len:.3f}', xy=(theta_len, power_len), 
             xytext=(theta_len+0.05, power_len+0.02), fontsize=10)
plt.annotate(f'{power_wid:.3f}', xy=(theta_wid, power_wid), 
             xytext=(theta_wid+0.05, power_wid-0.04), fontsize=10)

plt.xlabel('Отношение дисперсий θ = σ²₁ / σ²₂')
plt.ylabel('Мощность')
plt.title('График мощности F-критерия для сравнения дисперсий\nдлины и ширины черепа')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.ylim(0, 1.05)
plt.xlim(0.8, 2.0)
plt.tight_layout()
plt.savefig('мощность_критерия.png', dpi=300, bbox_inches='tight')   
plt.show()