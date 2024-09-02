from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from sklearn.decomposition import PCA

def pca_analysis(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        # CSVファイルを読み込み
        csv_file = request.FILES['csv_file']
        df = pd.read_csv(csv_file)

        # 必要に応じてデータの前処理
        categories = df.iloc[:, 0]  # カテゴリ列
        dfs = df.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0)  # 標準化

        # PCAの実行
        pca = PCA()
        pca.fit(dfs)
        feature = pca.transform(dfs)

        # 寄与率と累積寄与率の計算
        contribution_ratios = pca.explained_variance_ratio_
        cumulative_contribution = np.cumsum(contribution_ratios)

        # 寄与率をプロット
        plt.figure(figsize=(8, 5))
        plt.bar(x=["PC{}".format(x + 1) for x in range(len(dfs.columns))],
                height=contribution_ratios)
        plt.xlabel("Principal Components")
        plt.ylabel("Contribution Rate")
        plt.title("Contribution Rate of Each Principal Component")
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        contribution_plot_image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        # 累積寄与率のプロット
        plt.figure(figsize=(8, 5))
        plt.plot([0] + list(cumulative_contribution), "-o")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Contribution Rate")
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        cumulative_plot_image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        # カテゴリデータの主成分散布図をプロット
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=categories.astype('category').cat.codes, cmap='viridis')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Scatter Plot with Categories")
        plt.colorbar(label='Categories')
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        scatter_plot_image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        # 列ラベルの散布図をプロット
        plt.figure(figsize=(8, 6))
        plt.scatter(pca.components_[0, :], pca.components_[1, :], alpha=0.8)

        # 主成分数に合わせたラベルの表示
        num_components = pca.components_.shape[1]
        for i in range(num_components):
            plt.text(pca.components_[0, i], pca.components_[1, i], df.columns[i], fontsize=10, ha='right')

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA Component Contribution")
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        scatter_with_labels_plot_image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close()

        context = {
            'contribution_ratios': contribution_ratios,
            'cumulative_contribution': cumulative_contribution,
            'contribution_plot_image': contribution_plot_image_base64,
            'cumulative_plot_image': cumulative_plot_image_base64,
            'scatter_plot_image': scatter_plot_image_base64,
            'scatter_with_labels_plot_image': scatter_with_labels_plot_image_base64,
        }

        return render(request, 'analysis/pca_result.html', context)

    return render(request, 'analysis/pca_analysis.html')
