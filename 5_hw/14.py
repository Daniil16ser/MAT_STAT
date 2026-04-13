import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Данные
x = np.array([-1.11, -6.10, 2.42])
y = np.array([-2.29, -2.91])

# Параметры
n = len(x)
m = len(y)
sigma_x_sq = 2
sigma_y_sq = 1
alpha = 0.05

# Выборочные средние
x_bar = np.mean(x)
y_bar = np.mean(y)
delta_obs = x_bar - y_bar

# Стандартная ошибка и статистика
se = np.sqrt(sigma_x_sq/n + sigma_y_sq/m)
z_stat = delta_obs / se

# Критические значения и p-value
z_crit = norm.ppf(1 - alpha)
p_value = 1 - norm.cdf(z_stat)

# Функция мощности
def power_delta(delta, se, z_crit):
    return 1 - norm.cdf(z_crit - delta / se)

power_obs = power_delta(delta_obs, se, z_crit)

# Δ для мощности 0.8
delta_80 = fsolve(lambda d: power_delta(d, se, z_crit) - 0.8, 2)[0]

# Построение графика
delta_range = np.linspace(-1, 4, 500)
power_values = power_delta(delta_range, se, z_crit)

plt.figure(figsize=(10, 6))
plt.plot(delta_range, power_values, 'b-', linewidth=2, 
         label=f'Мощность критерия (n={n}, m={m})')
plt.axvline(x=0, color='gray', linestyle='--', label='H₀: Δ = 0')
plt.axhline(y=alpha, color='red', linestyle=':', 
            label=f'Уровень значимости α = {alpha}')
plt.axhline(y=0.8, color='green', linestyle=':', alpha=0.5,
            label='Мощность = 0.8')
plt.plot(delta_obs, power_obs, 'ro', markersize=8, 
         label=f'Наблюдаемая Δ = {delta_obs:.3f}\nМощность = {power_obs:.3f}')
plt.axvline(x=delta_80, color='green', linestyle='--', alpha=0.5)
plt.plot(delta_80, 0.8, 'go', markersize=6)
plt.annotate(f'Δ = {delta_80:.2f}', xy=(delta_80, 0.8), 
             xytext=(delta_80+0.2, 0.75), fontsize=9)

plt.xlabel('Истинная разность средних Δ = a - b')
plt.ylabel('Мощность')
plt.title(f'Мощность Z-критерия (σ²_x={sigma_x_sq}, σ²_y={sigma_y_sq}, n={n}, m={m})')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.ylim(0, 1.05)
plt.xlim(-1, 4)
plt.tight_layout()

# Сохранение
plt.savefig('мощность_z_критерия.png', dpi=300, bbox_inches='tight')
plt.savefig('мощность_z_критерия.pdf', bbox_inches='tight')

# Вывод в файл
with open('результаты_z_тест.txt', 'w', encoding='utf-8') as f:
    f.write("Z-КРИТЕРИЙ ДЛЯ РАЗНОСТИ СРЕДНИХ\n")
    f.write("="*40 + "\n\n")
    f.write(f"Выборка X: {x.tolist()}, n = {n}, σ²_x = {sigma_x_sq}\n")
    f.write(f"Выборка Y: {y.tolist()}, m = {m}, σ²_y = {sigma_y_sq}\n\n")
    f.write(f"Среднее X: {x_bar:.6f}\n")
    f.write(f"Среднее Y: {y_bar:.6f}\n")
    f.write(f"Разность средних: {delta_obs:.6f}\n\n")
    f.write(f"Стандартная ошибка: {se:.6f}\n")
    f.write(f"Z-статистика: {z_stat:.6f}\n")
    f.write(f"Критическое Z (α={alpha}): {z_crit:.6f}\n")
    f.write(f"p-value: {p_value:.6f}\n\n")
    f.write(f"Мощность при наблюдаемой Δ: {power_obs:.6f}\n")
    f.write(f"Δ для мощности 0.8: {delta_80:.6f}\n\n")
    f.write(f"ВЫВОД: {['Отвергаем H₀', 'Нет оснований отвергнуть H₀'][p_value > alpha]}\n")

plt.show()

print(f"Результаты сохранены в файлы:")
print(f"  - мощность_z_критерия.png")
print(f"  - мощность_z_критерия.pdf")
print(f"  - результаты_z_тест.txt")