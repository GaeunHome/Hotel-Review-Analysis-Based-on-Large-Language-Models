"""
IPA priority statistical tests on aggregated dashboard statistics.
Input: ipa_all_hotels_statistics.xlsx (from ipa_dashboard.py)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class IPAPriorityStatistics:
    """Statistical analysis for IPA priority data."""

    def __init__(self, statistics_file: str):
        if statistics_file.endswith('.csv'):
            self.df = pd.read_csv(statistics_file, encoding='utf-8-sig')
        else:
            self.df = pd.read_excel(statistics_file)
        print(f"✓ {len(self.df)} 筆統計記錄")

    def analyze_priority_distribution(self):
        """Descriptive statistics of improvement priority."""
        p = self.df['改進優先度']
        result = {
            'mean': p.mean(), 'median': p.median(), 'std': p.std(),
            'min': p.min(), 'max': p.max(),
            'q1': p.quantile(0.25), 'q3': p.quantile(0.75),
            'skewness': stats.skew(p), 'kurtosis': stats.kurtosis(p)
        }
        print(f"\n{'='*60}\n改進優先度分布\n{'='*60}")
        for k, v in result.items():
            print(f"  {k}: {v:.4f}")
        return result

    def test_importance_performance_gap(self, alpha=0.05):
        """Paired t-test: importance vs performance."""
        imp, perf = self.df['平均重要度'], self.df['平均績效']
        diff = imp - perf
        t, p = stats.ttest_rel(imp, perf)
        d = diff.mean() / diff.std()
        ci = stats.t.interval(0.95, len(diff)-1, loc=diff.mean(), scale=stats.sem(diff))

        print(f"\n{'='*60}\n配對 t 檢定：重要度 vs 績效\n{'='*60}")
        print(f"  平均差異: {diff.mean():.4f}, t={t:.4f}, p={p:.4f}")
        print(f"  Cohen's d: {d:.4f}, 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"  結論: {'重要度顯著高於績效' if p < alpha and t > 0 else '無顯著差異'}")
        return {'t': t, 'p': p, 'mean_diff': diff.mean(), 'cohens_d': d, 'ci': ci}

    def correlation_importance_performance(self, alpha=0.05):
        """Pearson & Spearman correlation between importance and performance."""
        imp, perf = self.df['平均重要度'], self.df['平均績效']
        pr, pp = stats.pearsonr(imp, perf)
        sr, sp = stats.spearmanr(imp, perf)

        strength = "弱" if abs(pr) < 0.3 else "中等" if abs(pr) < 0.7 else "強"
        direction = "正" if pr > 0 else "負"

        print(f"\n{'='*60}\n相關分析\n{'='*60}")
        print(f"  Pearson: r={pr:.4f}, p={pp:.4f} ({'顯著' if pp < alpha else '不顯著'})")
        print(f"  Spearman: rho={sr:.4f}, p={sp:.4f}")
        print(f"  解釋: {strength}{direction}相關")
        return {'pearson_r': pr, 'pearson_p': pp, 'spearman_r': sr, 'spearman_p': sp}

    def identify_critical_attributes(self, threshold=None):
        """Identify high-priority attributes."""
        if threshold is None:
            threshold = self.df['改進優先度'].quantile(0.75)
        critical = self.df[self.df['改進優先度'] >= threshold].sort_values('改進優先度', ascending=False)

        print(f"\n{'='*60}\n關鍵改進屬性 (優先度 >= {threshold:.2f})\n{'='*60}")
        for _, r in critical.iterrows():
            print(f"  {r['屬性']}: 優先度 {r['改進優先度']:.2f} "
                  f"(重要度 {r['平均重要度']:.2f}, 績效 {r['平均績效']:.2f})")
        return critical

    def test_positive_rate(self, benchmark=0.7, alpha=0.05):
        """One-sample t-test: positive rate vs benchmark."""
        rate = self.df['正向率%'] / 100.0
        t, p = stats.ttest_1samp(rate, benchmark)
        print(f"\n{'='*60}\n正向率檢定 (基準: {benchmark*100:.0f}%)\n{'='*60}")
        print(f"  平均: {rate.mean()*100:.2f}%, t={t:.4f}, p={p:.4f}")
        print(f"  結論: {'顯著' if p < alpha else '不顯著'}{'高於' if t > 0 else '低於'}基準")
        return {'mean': rate.mean(), 't': t, 'p': p}

    def generate_visualization(self, output_dir: str):
        """Generate statistical charts."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. IPA scatter
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(self.df['平均績效'], self.df['平均重要度'],
                        s=self.df['提及次數'] * 2, c=self.df['改進優先度'],
                        cmap='RdYlGn_r', alpha=0.6, edgecolors='black')
        plt.colorbar(sc, label='改進優先度')
        ax.set_xlabel('平均績效'); ax.set_ylabel('平均重要度')
        ax.set_title('IPA 散點圖', fontsize=14, fontweight='bold')
        ax.axhline(y=self.df['平均重要度'].median(), color='gray', ls='--', alpha=0.5)
        ax.axvline(x=self.df['平均績效'].median(), color='gray', ls='--', alpha=0.5)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'IPA散點圖.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Priority distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.hist(self.df['改進優先度'], bins=20, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('改進優先度'); ax1.set_title('分布')
        ax2.boxplot(self.df['改進優先度']); ax2.set_title('箱型圖')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '改進優先度分布.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Importance vs Performance bars
        top = self.df.nlargest(10, '改進優先度')
        x = np.arange(len(top))
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - 0.175, top['平均重要度'], 0.35, label='重要度', alpha=0.8)
        ax.bar(x + 0.175, top['平均績效'], 0.35, label='績效', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(top['屬性'], rotation=45, ha='right')
        ax.set_title('Top 10 改進項目：重要度 vs 績效', fontweight='bold')
        ax.legend(); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '重要度績效對比.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Correlation heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = self.df[['平均重要度', '平均績效', '改進優先度', '正向率%']].corr()
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, ax=ax)
        ax.set_title('相關矩陣', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '相關矩陣.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 圖表: {output_dir}")

    def export_report(self, output_file: str):
        """Export full statistical report to Excel."""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            self.df.to_excel(writer, sheet_name='原始資料', index=False)
            self.df[['平均重要度', '平均績效', '改進優先度', '正向率%']].describe().to_excel(
                writer, sheet_name='描述性統計')
            self.identify_critical_attributes().to_excel(
                writer, sheet_name='關鍵改進項目', index=False)
        print(f"✅ 報告: {output_file}")


def main():
    input_file = "data/output/ipa_all_hotels_statistics.xlsx"
    if not os.path.exists(input_file):
        print(f"找不到 {input_file}，請先執行 ipa_dashboard")
        return

    print("="*80 + "\nIPA 優先度統計檢定\n" + "="*80)
    a = IPAPriorityStatistics(input_file)

    a.analyze_priority_distribution()
    a.test_importance_performance_gap()
    a.correlation_importance_performance()
    a.test_positive_rate()
    a.identify_critical_attributes()
    a.generate_visualization("data/output/statistical_charts")
    a.export_report("data/output/ipa_priority_statistics_report.xlsx")

    print("\n✅ 統計分析完成！")


if __name__ == "__main__":
    main()
