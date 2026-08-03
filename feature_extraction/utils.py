from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_tracer_type(filename):
    filename = str(filename).lower()
    if 'fdg_' in filename:
        return 'FDG'
    if 'psma_' in filename:
        return 'PSMA'
    return 'Unknown'


def compute_pairwise_cosine_stats(embeddings, files):
    distances = squareform(pdist(embeddings, metric='cosine'))
    np.fill_diagonal(distances, np.inf)

    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    min_distance = distances[min_idx]
    closest_pair = (min_idx, min_distance, files[min_idx[0]], files[min_idx[1]])

    np.fill_diagonal(distances, -np.inf)
    max_idx = np.unravel_index(np.argmax(distances), distances.shape)
    max_distance = distances[max_idx]
    furthest_pair = (max_idx, max_distance, files[max_idx[0]], files[max_idx[1]])

    full_distances = squareform(pdist(embeddings, metric='cosine'))
    return {
        'pairwise_distances': full_distances,
        'closest': closest_pair,
        'furthest': furthest_pair,
    }


def compute_unlabelled_distance_to_labelled(unlabelled_embeddings, labelled_embeddings):
    if len(labelled_embeddings) == 0:
        return np.zeros(len(unlabelled_embeddings))
    return np.min(cdist(unlabelled_embeddings, labelled_embeddings, metric='cosine'), axis=1)


def build_distance_dataframe(unlabelled_results):
    rows = []
    for tracer, res in unlabelled_results.items():
        for idx, (filename, distance) in enumerate(zip(res['files_unlabelled'], res['min_dist_to_labelled'])):
            rows.append({
                'tracer': tracer,
                'filename': filename,
                'dist_to_labelled': float(distance),
                'index': idx,
            })
    if not rows:
        return pd.DataFrame(columns=['tracer', 'filename', 'dist_to_labelled', 'index'])
    return pd.DataFrame(rows).sort_values('dist_to_labelled', ascending=False).reset_index(drop=True)


def plot_distance_distribution(distance_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(distance_df['dist_to_labelled'], bins=30, color='steelblue', alpha=0.75, edgecolor='black')
    axes[0].set_title('Distance Distribution - All Unlabelled Data', fontweight='bold')
    axes[0].set_xlabel('Cosine Distance to Labelled Set')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.25)

    for tracer in sorted(distance_df['tracer'].unique()):
        tracer_data = distance_df.loc[distance_df['tracer'] == tracer, 'dist_to_labelled']
        axes[1].hist(tracer_data, bins=20, alpha=0.65, label=tracer, edgecolor='black')
    axes[1].set_title('Distance Distribution - By Tracer', fontweight='bold')
    axes[1].set_xlabel('Cosine Distance to Labelled Set')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


def get_top_percent_farthest(distance_df, top_percent=0.05):
    if distance_df.empty:
        return distance_df.copy()
    top_n = max(1, int(np.ceil(len(distance_df) * top_percent)))
    return distance_df.head(top_n).copy()


def get_top_percent_farthest_by_tracer(distance_df, top_percent=0.05):
    if distance_df.empty:
        return distance_df.copy()
    top_frames = []
    for tracer, tracer_df in distance_df.groupby('tracer', sort=False):
        tracer_df = tracer_df.sort_values('dist_to_labelled', ascending=False).reset_index(drop=True)
        top_n = max(1, int(np.ceil(len(tracer_df) * top_percent)))
        top_frames.append(tracer_df.head(top_n).copy())
    if not top_frames:
        return distance_df.head(0).copy()
    return pd.concat(top_frames, ignore_index=True).sort_values(['tracer', 'dist_to_labelled'], ascending=[True, False]).reset_index(drop=True)


def rank_top_subset_by_diversity(unlabelled_results, top_subset_df):
    ranked_rows = []
    for tracer in sorted(top_subset_df['tracer'].unique()):
        tracer_subset = top_subset_df[top_subset_df['tracer'] == tracer].copy()
        if tracer_subset.empty:
            continue

        res = unlabelled_results[tracer]
        orig_indices = tracer_subset['index'].to_numpy()
        subset_embeddings = res['embeddings_unlabelled'][orig_indices]

        if len(subset_embeddings) == 1:
            tracer_subset['avg_dist_to_top_subset'] = float('nan')
            tracer_subset['diversity_rank'] = 1
            ranked_rows.append(tracer_subset)
            continue

        intra_distances = squareform(pdist(subset_embeddings, metric='cosine'))
        avg_distances = intra_distances.mean(axis=1)
        diversity_order = np.argsort(avg_distances)[::-1]

        tracer_subset = tracer_subset.iloc[diversity_order].copy()
        tracer_subset['avg_dist_to_top_subset'] = avg_distances[diversity_order]
        tracer_subset['diversity_rank'] = np.arange(1, len(tracer_subset) + 1)
        ranked_rows.append(tracer_subset)

    if not ranked_rows:
        return pd.DataFrame(columns=list(top_subset_df.columns) + ['avg_dist_to_top_subset', 'diversity_rank'])
    return pd.concat(ranked_rows, ignore_index=True)


def plot_kernel_pca_with_top_subset(unlabelled_results, labelled_results, top_subset_df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for idx, tracer in enumerate(['FDG', 'PSMA']):
        if tracer not in unlabelled_results:
            continue

        result = unlabelled_results[tracer]
        u_pca = result['kpca_coords_unlabelled']
        l_pca = result['kpca_coords_labelled']
        closest_idx = result['closest'][0]
        furthest_idx = result['furthest'][0]
        min_dist_to_lab = result['min_dist_to_labelled']
        ax = axes[idx]

        tracer_top = top_subset_df[top_subset_df['tracer'] == tracer]
        tracer_top_n = len(tracer_top)

        scatter = ax.scatter(u_pca[:, 0], u_pca[:, 1],
                             c=min_dist_to_lab, cmap='YlOrRd', s=60,
                             edgecolors='gray', linewidth=0.3, alpha=0.7,
                             label=f'Unlabelled (n={len(u_pca)})', zorder=2)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Cosine dist to nearest labelled', fontsize=9)

        ax.scatter(l_pca[:, 0], l_pca[:, 1],
                   c='#2471A3', s=70, marker='x', alpha=1.0,
                   label=f'Labelled (n={len(l_pca)})', zorder=10)

        if not tracer_top.empty:
            top_idx = tracer_top['index'].to_numpy()
            ax.scatter(u_pca[top_idx, 0], u_pca[top_idx, 1],
                       facecolors='none', s=260, marker='o', edgecolors='black', linewidth=1.2,
                       label=f'Top 5% furthest (n={tracer_top_n})', zorder=11)
            ax.scatter(u_pca[top_idx, 0], u_pca[top_idx, 1],
                       facecolors='none', s=360, marker='o', edgecolors='#F4D03F', linewidth=2.2,
                       zorder=10)

        ax.scatter(u_pca[closest_idx[0], 0], u_pca[closest_idx[0], 1],
                   color='#E67E22', s=280, marker='s', edgecolors='white', linewidth=2, zorder=8)
        ax.scatter(u_pca[closest_idx[1], 0], u_pca[closest_idx[1], 1],
                   color='#E67E22', s=280, marker='s', edgecolors='white', linewidth=2,
                   label=f'Closest pair (d={result["closest"][1]:.4f})', zorder=8)
        ax.plot([u_pca[closest_idx[0], 0], u_pca[closest_idx[1], 0]],
                [u_pca[closest_idx[0], 1], u_pca[closest_idx[1], 1]],
                color='#E67E22', linestyle='--', linewidth=2.5, alpha=0.7, zorder=3)

        ax.scatter(u_pca[furthest_idx[0], 0], u_pca[furthest_idx[0], 1],
                   color='#1ABC9C', s=280, marker='^', edgecolors='white', linewidth=2, zorder=8)
        ax.scatter(u_pca[furthest_idx[1], 0], u_pca[furthest_idx[1], 1],
                   color='#1ABC9C', s=280, marker='^', edgecolors='white', linewidth=2,
                   label=f'Furthest pair (d={result["furthest"][1]:.4f})', zorder=8)
        ax.plot([u_pca[furthest_idx[0], 0], u_pca[furthest_idx[1], 0]],
                [u_pca[furthest_idx[0], 1], u_pca[furthest_idx[1], 1]],
                color='#1ABC9C', linestyle='--', linewidth=2.5, alpha=0.7, zorder=3)

        ax.set_xlabel('KPC1', fontsize=11)
        ax.set_ylabel('KPC2', fontsize=11)
        ax.set_title(f'Unlabelled + Labelled — {tracer}\nCosine Kernel PCA | Distances in full embedding space',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=8, framealpha=0.95, markerscale=0.6,
                  edgecolor='gray', fancybox=True)
        ax.grid(True, alpha=0.2, linestyle='-')
        ax.set_facecolor('#FAFAFA')

    plt.tight_layout()
    plt.show()
