"""
Comprehensive statistical tests for IPA analysis.
Tests: descriptive stats, normality, paired t-test, correlation,
       ANOVA, non-parametric (Wilcoxon, Kruskal-Wallis), gap analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict
import json
import warnings
warnings.filterwarnings('ignore')


class IPAStatisticalTests:
    """Full statistical test suite for IPA data."""

    def __init__(self, data_file: str):
        if data_file.endswith('.csv'):
            self.df = pd.read_csv(data_file, encoding='utf-8-sig')
        else:
            self.df = pd.read_excel(data_file)
        self.results = {}

    def run_all_tests(self, alpha: float = 0.05) -> Dict:
        """Execute all statistical tests."""
        print("="*80 + "\n執行 IPA 統計檢定\n" + "="*80)
        self._prepare_data()

        tests = [
            ('descriptive', self.descriptive_statistics),
            ('normality', self.normality_tests),
            ('paired_ttest', self.paired_ttest),
            ('correlation', self.correlation_analysis),
            ('anova', self.anova_analysis),
            ('nonparametric', self.nonparametric_tests),
            ('gap_analysis', self.gap_analysis),
        ]
        for i, (name, fn) in enumerate(tests, 1):
            print(f"\n【{i}/{len(tests)}】{name}...")
            self.results[name] = fn(alpha)

        print("\n✅ 所有統計檢定完成！")
        return self.results

    def _prepare_data(self):
        """Extract statistics from raw review data if needed."""
        if '分析結果' not in self.df.columns:
            print("  ✓ 使用現有統計資料")
            return

        print("  提取原始評論統計...")
        records = []
        for _, row in self.df.iterrows():
            hotel = row.get('Hotel Name', 'Unknown')
            try:
                items = json.loads(str(row.get('分析結果', '[]')))
                for item in items:
                    records.append({
                        'Hotel Name': hotel,
                        'Category': item.get('key', ''),
                        'Sentiment': item.get('sentiment', ''),
                        'Score': item.get('score', 0),
                        'Importance': item.get('importance', 3)
                    })
            except Exception:
                continue

        raw = pd.DataFrame(records)
        agg = []
        for hotel in raw['Hotel Name'].unique():
            for cat in raw[raw['Hotel Name'] == hotel]['Category'].unique():
                subset = raw[(raw['Hotel Name'] == hotel) & (raw['Category'] == cat)]
                agg.append({
                    'Hotel Name': hotel, 'Category': cat,
                    'Importance': subset['Importance'].mean(),
                    'Performance': subset['Score'].mean(),
                    'Count': len(subset)
                })
        self.df = pd.DataFrame(agg)
        print(f"  ✓ {len(self.df)} 筆統計數據")

    def descriptive_statistics(self, alpha=0.05):
        """Descriptive statistics for importance, performance, and gap."""
        diff = self.df['Importance'] - self.df['Performance']
        result = pd.DataFrame({
            '變數': ['重要度', '績效', '差距'],
            '樣本數': [len(self.df)] * 3,
            '平均數': [self.df['Importance'].mean(), self.df['Performance'].mean(), diff.mean()],
            '標準差': [self.df['Importance'].std(), self.df['Performance'].std(), diff.std()],
            '中位數': [self.df['Importance'].median(), self.df['Performance'].median(), diff.median()],
            '最小值': [self.df['Importance'].min(), self.df['Performance'].min(), diff.min()],
            '最大值': [self.df['Importance'].max(), self.df['Performance'].max(), diff.max()]
        })
        print(f"\n{'='*60}\n描述性統計\n{'='*60}")
        print(result.to_string(index=False))
        return result

    def normality_tests(self, alpha=0.05):
        """Shapiro-Wilk and KS normality tests."""
        results = {}
        for name, data in [('Importance', self.df['Importance']),
                           ('Performance', self.df['Performance'])]:
            sw_stat, sw_p = stats.shapiro(data)
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            results[name] = {
                'Shapiro-Wilk': {'stat': sw_stat, 'p': sw_p, 'normal': sw_p > alpha},
                'KS': {'stat': ks_stat, 'p': ks_p, 'normal': ks_p > alpha}
            }
        print(f"\n{'='*60}\n常態性檢定\n{'='*60}")
        for v, r in results.items():
            for t, d in r.items():
                print(f"  {v} {t}: stat={d['stat']:.4f}, p={d['p']:.4f} "
                      f"({'常態' if d['normal'] else '非常態'})")
        return results

    def paired_ttest(self, alpha=0.05):
        """Paired t-test: importance vs performance."""
        imp, perf = self.df['Importance'], self.df['Performance']
        diff = imp - perf
        t, p = stats.ttest_rel(imp, perf)
        d = diff.mean() / diff.std()

        conclusion = ('重要度顯著高於績效' if t > 0 and p < alpha else
                      '績效顯著高於重要度' if t < 0 and p < alpha else
                      '無顯著差異')
        print(f"\n{'='*60}\n配對 t 檢定\n{'='*60}")
        print(f"  t={t:.4f}, p={p:.4f}, diff={diff.mean():.4f}, Cohen's d={d:.4f}")
        print(f"  結論: {conclusion}")
        return {'t': t, 'p': p, 'mean_diff': diff.mean(), 'cohens_d': d, 'conclusion': conclusion}

    def correlation_analysis(self, alpha=0.05):
        """Pearson and Spearman correlation."""
        imp, perf = self.df['Importance'], self.df['Performance']
        pr, pp = stats.pearsonr(imp, perf)
        sr, sp = stats.spearmanr(imp, perf)

        strength = "弱" if abs(pr) < 0.3 else "中等" if abs(pr) < 0.7 else "強"
        print(f"\n{'='*60}\n相關分析\n{'='*60}")
        print(f"  Pearson: r={pr:.4f}, p={pp:.4f}")
        print(f"  Spearman: rho={sr:.4f}, p={sp:.4f}")
        print(f"  解釋: {strength}{'正' if pr > 0 else '負'}相關")
        return {'Pearson': {'r': pr, 'p': pp}, 'Spearman': {'r': sr, 'p': sp}}

    def anova_analysis(self, alpha=0.05):
        """One-way ANOVA across hotels."""
        if 'Hotel Name' not in self.df.columns or len(self.df['Hotel Name'].unique()) < 2:
            return {'error': '需要至少兩家酒店'}

        hotels = self.df['Hotel Name'].unique()
        imp_groups = [self.df[self.df['Hotel Name'] == h]['Importance'].values for h in hotels]
        perf_groups = [self.df[self.df['Hotel Name'] == h]['Performance'].values for h in hotels]
        fi, pi = stats.f_oneway(*imp_groups)
        fp, pp = stats.f_oneway(*perf_groups)

        print(f"\n{'='*60}\nANOVA ({len(hotels)} 家酒店)\n{'='*60}")
        print(f"  重要度: F={fi:.4f}, p={pi:.4f}")
        print(f"  績效: F={fp:.4f}, p={pp:.4f}")
        return {'Importance': {'F': fi, 'p': pi}, 'Performance': {'F': fp, 'p': pp}}

    def nonparametric_tests(self, alpha=0.05):
        """Wilcoxon signed-rank and Kruskal-Wallis tests."""
        imp, perf = self.df['Importance'], self.df['Performance']
        w, wp = stats.wilcoxon(imp, perf)

        print(f"\n{'='*60}\n非參數檢定\n{'='*60}")
        print(f"  Wilcoxon: W={w:.4f}, p={wp:.4f}")
        result = {'Wilcoxon': {'stat': w, 'p': wp}}

        if 'Hotel Name' in self.df.columns and len(self.df['Hotel Name'].unique()) >= 2:
            hotels = self.df['Hotel Name'].unique()
            ki, kip = stats.kruskal(*[self.df[self.df['Hotel Name'] == h]['Importance'].values for h in hotels])
            kp, kpp = stats.kruskal(*[self.df[self.df['Hotel Name'] == h]['Performance'].values for h in hotels])
            result['Kruskal-Wallis'] = {'Importance': {'H': ki, 'p': kip}, 'Performance': {'H': kp, 'p': kpp}}
            print(f"  Kruskal-Wallis 重要度: H={ki:.4f}, p={kip:.4f}")
            print(f"  Kruskal-Wallis 績效: H={kp:.4f}, p={kpp:.4f}")
        return result

    def gap_analysis(self, alpha=0.05):
        """Gap analysis: importance - performance."""
        self.df['Gap'] = self.df['Importance'] - self.df['Performance']
        self.df['Gap_Category'] = pd.cut(
            self.df['Gap'], bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=['績效超過重要度', '平衡', '重要度超過績效'])

        dist = self.df['Gap_Category'].value_counts()
        t, p = stats.ttest_1samp(self.df['Gap'], 0)

        print(f"\n{'='*60}\n差距分析\n{'='*60}")
        print(f"  平均差距: {self.df['Gap'].mean():.4f}")
        for cat, count in dist.items():
            print(f"  {cat}: {count} ({count/len(self.df)*100:.1f}%)")
        print(f"  t={t:.4f}, p={p:.4f}")
        return {'mean_gap': self.df['Gap'].mean(), 't': t, 'p': p, 'distribution': dist.to_dict()}

    def export_results(self, output_file: str):
        """Export all test results to Excel."""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            if 'descriptive' in self.results:
                self.results['descriptive'].to_excel(writer, sheet_name='描述性統計', index=False)
            if 'paired_ttest' in self.results:
                pd.DataFrame([self.results['paired_ttest']]).to_excel(
                    writer, sheet_name='配對t檢定', index=False)
            if 'correlation' in self.results:
                rows = [{'方法': m, '係數': d['r'], 'p值': d['p']}
                        for m, d in self.results['correlation'].items()]
                pd.DataFrame(rows).to_excel(writer, sheet_name='相關分析', index=False)
            if 'gap_analysis' in self.results:
                pd.DataFrame([{k: v for k, v in self.results['gap_analysis'].items()
                              if k != 'distribution'}]).to_excel(
                    writer, sheet_name='差距分析', index=False)
        print(f"✅ 結果: {output_file}")


def main():
    import os
    input_file = "data/output/reviews_analysis_with_importance.xlsx"
    if not os.path.exists(input_file):
        print(f"找不到 {input_file}，請先執行 IPA 分析")
        return

    tester = IPAStatisticalTests(input_file)
    tester.run_all_tests(alpha=0.05)
    tester.export_results("data/output/statistical_test_results.xlsx")


if __name__ == "__main__":
    main()
