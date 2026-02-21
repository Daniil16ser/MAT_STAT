import numpy as np
import matplotlib.pyplot as plt 
import statistics
import scipy.stats as stats
from scipy.special import comb

np.random.seed(26)

arr1 = np.random.random(25)
xp = -np.log(1-arr1)

print("Пункт а)")
med = np.median(xp)
l = np.max(xp) - np.min(xp)
mod = statistics.multimode(xp)

print("Мода: ", mod)   
print("Медиана: ", med)
print("Размах", l)



m3 = stats.moment(xp, moment=3)

variance = np.var(xp)
sigma = np.sqrt(variance)

otkl = m3/sigma**3
print("Оценка коэффициента асимметрии", otkl)


print("Пункт b)")
# Империческая функция
x_ecdf = np.sort(xp.copy())
n = len(xp)
y_ecdf = np.arange(1, n + 1) / n 


plt.figure(figsize=(12, 5))
plt.step(x_ecdf, y_ecdf, where='post', linewidth=2, color='blue')
plt.scatter(x_ecdf, y_ecdf, color='red', s=50, zorder=5)
plt.xlabel('x')
plt.ylabel('~F(x)')
plt.title('Эмпирическая функция распределения (n=25)')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
median_xp = np.median(xp)
plt.axvline(median_xp, color='green', linestyle='--', label=f'Медиана = {median_xp:.3f}')
plt.axhline(0.5, color='gray', linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('empirical_distribution.png')
plt.show()  


# Построиение гистограммы 
k = int(1 + np.log2(n))

xmin = float(np.min(xp))
xmax = float(np.max(xp))
delta = (xmax - xmin) / k
bins_edges = np.linspace(xmin, xmax, k + 1)
plt.figure(figsize=(12, 5))

plt.hist(xp, bins=k, edgecolor='black', alpha=0.7, color='skyblue',density=True)
plt.xlabel('x')
plt.ylabel('Частота')
plt.title(f'Гистограмма частот (k={k})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
plt.show()


# Построение boxplot
plt.figure(figsize=(10, 4))
plt.boxplot(xp, vert=False)
plt.title('boxplot')
plt.xlabel('Значения')
plt.grid(True, alpha=0.3)
plt.savefig('boxplot_horizontal.png', dpi=300)
plt.show()


print("Пункт c)")
# Бутстрап
B = 1000
bootstrap_means = np.array([np.mean(np.random.choice(xp, size=n, replace=True)) 
                            for _ in range(B)])

# Параметры
mean_xp = np.mean(xp)
se_theor = 1 / np.sqrt(n)

# Построение графиков
plt.figure(figsize=(12, 5))

# Гистограмма бутстрапа
k = int(1 + np.log2(B))
plt.hist(bootstrap_means, bins=k, density=True, edgecolor='black', 
         alpha=0.7, color='skyblue', label='Бутстрап')

# Кривая ЦПТ
x = np.linspace(np.min(bootstrap_means), np.max(bootstrap_means), 100)
plt.plot(x, stats.norm.pdf(x, mean_xp, se_theor), 'r-', 
         linewidth=2, label='ЦПТ (теоретическая)')
plt.xlabel('Среднее')
plt.ylabel('Плотность')
plt.title('Сравнение бутстрапа и ЦПТ')
plt.legend()
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig('cpt_bootstrap.png')
plt.show()




print("Пункт d")
# Бутстрап оценка плотности распределения коэффициента асиметрии и оценка вероятности, что асиметрия < 1
def skew(x):
    return np.mean((x - np.mean(x))**3) / np.std(x, ddof=1)**3

boot = [skew(np.random.choice(xp, 25, True)) for _ in range(10000)]
prob = np.mean(np.array(boot) < 1)

plt.hist(boot, bins=30, color='skyblue', edgecolor='black')
plt.axvline(1, color='red', ls='--', linewidth=2)
plt.title(f'Бутстрап асимметрии\nP(<1) = {prob:.3f}')
plt.xlabel('Коэффициент асимметрии')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3)
plt.savefig('bootstrap_score.png')
plt.show()

print(f'Вероятность: {prob:.4f}')




print("Пукнт e)")
boot_medians = []
boot_means = []

for i in range(B):
    sample = np.random.choice(xp, size=25, replace=True)
    boot_medians.append(np.median(sample))
# Преобразуем x в массив (на случай скаляра)
    x = np.asarray(x)
    
def p_median(x: np.ndarray) -> np.ndarray:
    # Преобразуем входные данные в массив numpy (на случай, если передан скаляр)

    x  = np.asarray(x)
    

    # Вектор значений i от 13 до 25 включительно
    i_vals = np.arange(13, 26)          # shape = (13,)
    
    # Заранее вычисляем биномиальные коэффициенты C(25, i)
    comb_vals = comb(25, i_vals)        # shape = (13,)
    
    # Общие для всех i выражения
    exp_neg_x = np.exp(-x)              # e^{-x}
    one_minus_exp = 1 - exp_neg_x       # 1 - e^{-x}
    
    # Добавляем новую ось для i, чтобы использовать broadcasting
    # Форма x_expanded: (..., 1), one_minus_exp_expanded: (..., 1)
    x_expanded = x[..., np.newaxis]
    one_minus_exp_expanded = one_minus_exp[..., np.newaxis]
    
    # Первое слагаемое: i * (1-e^{-x})^{i-1} * e^{-(26-i)x}
    term1 = (i_vals *
            (one_minus_exp_expanded) ** (i_vals - 1) *
            np.exp(-(26 - i_vals) * x_expanded))
    
    # Второе слагаемое: (1-e^{-x})^i * (-(25-i)) * e^{-(25-i)x}
    term2 = ((one_minus_exp_expanded) ** i_vals *
            (-(25 - i_vals)) *
            np.exp(-(25 - i_vals) * x_expanded))
    
    # Сумма двух слагаемых, умноженная на биномиальные коэффициенты
    total = comb_vals * (term1 + term2)   # shape = (..., 13)
    
    # Суммируем по i (последняя ось) – получаем итоговое значение p(x)
    result = np.sum(total, axis=-1)
    
    return result

x = np.linspace(np.min(boot_medians), np.max(boot_medians), num=1000)
y = p_median(x)

plt.figure(figsize=(10, 6))
plt.hist(boot_medians, bins=30, alpha=0.5, color='skyblue', edgecolor='black', label='Медианы', density=True)
plt.plot(x,y,color = 'g', label = "p_madian")
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.title('Сравнение бутстрап-распределений медианы и среднего')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bootstrap_score_vs_med.png')
plt.show()