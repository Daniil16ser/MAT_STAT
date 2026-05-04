from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from itertools import combinations

class CategoricalRegression:
    def __init__(self, category: Dict[int, List]):
        """
        category: dict {category number: [values]}
        """
        self.categories = sorted(category.keys())
        self.n_categories = len(self.categories)
        self.n_total = sum(len(v) for v in category.values())

        # U (one-hot encoding)
        self.U = self._build_design_matrix(category)

        # F = U^T @ U
        self.F = self.U.T @ self.U

        self.Y = np.array([val for cat in self.categories for val in category[cat]])

        # Коэффициенты регрессии (средние по группам)
        self.coefficients = self._calculate_coefficients()

        # Остатки
        self.residuals = self.Y - self.U @ self.coefficients

        # RSS (Residual Sum of Squares)
        self.RSS = self.residuals.T @ self.residuals

        # Число степеней свободы
        self.df_residual = self.n_total - self.n_categories

        # MSE
        self.MSE = self.RSS / self.df_residual

    def _build_design_matrix(self, category: Dict[int, List]) -> np.ndarray:
        """U (one-hot encoding)"""
        U = np.zeros((self.n_total, self.n_categories))
        row_idx = 0
        for cat_idx, cat_key in enumerate(self.categories):
            n_samples = len(category[cat_key])
            U[row_idx:row_idx + n_samples, cat_idx] = 1
            row_idx += n_samples
        return U

    def _calculate_coefficients(self) -> np.ndarray:
        """b = F^{-1} @ U^T @ Y"""
        try:
            return np.linalg.inv(self.F) @ self.U.T @ self.Y
        except np.linalg.LinAlgError:
            return np.linalg.pinv(self.F) @ self.U.T @ self.Y

    def get_F(self) -> np.ndarray:
        """Fisher matrix"""
        return self.F

    def get_U(self) -> np.ndarray:
        """Return matrix of base functions"""
        return self.U

    def get_coefficients(self) -> np.ndarray:
        """Return coef regression"""
        return self.coefficients

    def predict(self, category_idx: int) -> float:
        """Prediction for category"""
        return self.coefficients[category_idx - 1]

    def get_RSS(self) -> float:
        """Residual Sum of Squares"""
        return self.RSS

    def get_MSE(self) -> float:
        """Mean Squared Error"""
        return self.MSE
    
    def get_counts(self) -> np.ndarray:
        """Количество наблюдений в каждой категории"""
        return np.diag(self.F)
    
    def _compute_pairwise_stats(self) -> List[dict]:
        """
        p-value
        """
        n = self.get_counts()
        results = []
        
        for i, j in combinations(range(self.n_categories), 2):
            diff = self.coefficients[i] - self.coefficients[j]
            
            se_diff = np.sqrt(self.MSE * (1/n[i] + 1/n[j]))
            
            t_stat = np.abs(diff) / se_diff
            
            # p-value 
            # t.cdf(t_stat, df) get P(T <= t_stat)
            # we need P(|T| >= t_stat) = 2 * (1 - P(T <= t_stat))
            p_value = 2 * (1 - stats.t.cdf(t_stat, self.df_residual))
            
            results.append({
                'cat1': self.categories[i],
                'cat2': self.categories[j],
                'diff': diff,
                't_stat': t_stat,
                'p_value': p_value,
                'se_diff': se_diff
            })
        
        return results
    
    def compare(self, alpha: float = 0.05) -> List[dict]:
        comparisons = self._compute_pairwise_stats()
        
        n_comparisons = len(comparisons)
        
        comparisons_sorted = sorted(comparisons, key=lambda x: x['p_value'])
        
        for rank, comp in enumerate(comparisons_sorted):
            adjusted_alpha = alpha / (n_comparisons - rank)
            comp['rank'] = rank + 1
            comp['adjusted_alpha'] = adjusted_alpha
            comp['significant'] = comp['p_value'] < adjusted_alpha
            
            if comp['significant']:
                comp['conclusion'] = (
                    f"Группы {comp['cat1']} и {comp['cat2']} "
                    f"СТАТИСТИЧЕСКИ РАЗЛИЧАЮТСЯ "
                    f"(p={comp['p_value']:.4f} < {adjusted_alpha:.4f})"
                )
            else:
                comp['conclusion'] = (
                    f"Группы {comp['cat1']} и {comp['cat2']} "
                    f"НЕ РАЗЛИЧАЮТСЯ "
                    f"(p={comp['p_value']:.4f} >= {adjusted_alpha:.4f})"
                )
        
        return comparisons_sorted


category = {
    1: [83, 85], 
    2: [84, 85, 85, 86, 86, 87], 
    3: [86, 87, 87, 87, 88, 88, 88, 88, 88, 89, 90], 
    4: [89, 90, 90, 91],
    5: [90, 92]
}

model = CategoricalRegression(category)

print("=" * 70)
print("КОЭФФИЦИЕНТЫ (СРЕДНИЕ ПО ГРУППАМ)")
print("=" * 70)
for i, (cat, coef) in enumerate(zip(model.categories, model.get_coefficients())):
    print(f'  Группа {cat}: {coef:.4f}')

print(f'\nRSS = {model.get_RSS():.4f}')
print(f'MSE = {model.get_MSE():.4f}')
print(f'df_residual = {model.df_residual}')

# Попарные сравнения с процедурой Бонферрони-Холма
print("\n" + "=" * 70)
print("ПОПАРНЫЕ СРАВНЕНИЯ (Бонферрони-Холм, α = 0.05)")
print("=" * 70)

results = model.compare(alpha=0.05)

print(f'\n{"Ранг":<6} {"Группы":<12} {"Разница":<10} {"t":<10} {"p-value":<10} {"α скорр.":<12} {"Значимо?":<12}')
print('-' * 70)
for r in results:
    print(f'{r["rank"]:<6} '
          f'{r["cat1"]} vs {r["cat2"]:<5} '
          f'{r["diff"]:<10.4f} '
          f'{r["t_stat"]:<10.4f} '
          f'{r["p_value"]:<10.4f} '
          f'{r["adjusted_alpha"]:<12.4f} '
          f'{"ДА" if r["significant"] else "НЕТ":<12}')

print('\n' + '=' * 70)
print('ВЫВОДЫ:')
print('=' * 70)
for r in results:
    print(f'  {r["conclusion"]}')