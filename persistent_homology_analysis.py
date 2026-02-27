"""
persistent_homology_analysis.py

論文 arXiv:2204.11139v1 "Musical Stylistic Analysis: A Study of
Intervallic Transition Graphs via Persistent Homology" の手法を
sanjuanito / tobas の MIDI ファイルに適用する。

手順:
  1. MIDI ファイルからノートを読み込み、12×12 の Intervallic Transition
     Matrix (ITM) を構築する。
  2. ITM をヒートマップとして可視化する。
  3. ITM から距離行列を構築する（補助ノード＋最短パス距離）。
  4. Vietoris-Rips 複体により Persistent Homology (H0, H1) を計算する。
  5. Persistence Diagram / Barcode を描画し、統計量を算出する。
  6. ジャンル間の比較を行う。

出力先: persistent_homology/<ジャンル>/<曲名>/
"""

import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pretty_midi
import networkx as nx
from ripser import ripser
from persim import plot_diagrams
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIDI_DIR = os.path.join(SCRIPT_DIR, "midi")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "persistent_homology")
GENRES = ["sanjuanito", "tobas"]
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------------------
# 1. Intervallic Transition Matrix の構築
# ---------------------------------------------------------------------------
def build_itm(midi_path: str) -> np.ndarray:
    """
    MIDI ファイルからピッチクラス間の 12×12 Intervallic Transition Matrix
    (ITM) を構築する。

    各遷移 (n_k, n_{k+1}) の重みは d_k * d_{k+1} (持続時間の積)。
    行列は全体の合計で正規化し、ジョイント確率分布にする。
    """
    pm = pretty_midi.PrettyMIDI(midi_path)

    # 全ノートを開始時間順にソートして1列のリストにする
    all_notes = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            all_notes.append(note)

    if len(all_notes) < 2:
        print(f"  [WARN] ノート数が不足: {midi_path}")
        return np.zeros((12, 12))

    # 開始時間でソート
    all_notes.sort(key=lambda n: (n.start, n.pitch))

    itm = np.zeros((12, 12))
    total_weight = 0.0

    for k in range(len(all_notes) - 1):
        n_k = all_notes[k]
        n_k1 = all_notes[k + 1]

        pc_i = n_k.pitch % 12
        pc_j = n_k1.pitch % 12

        d_k = n_k.end - n_k.start   # 持続時間
        d_k1 = n_k1.end - n_k1.start

        w = d_k * d_k1
        itm[pc_i, pc_j] += w
        total_weight += w

    # 正規化
    if total_weight > 0:
        itm /= total_weight

    return itm


# ---------------------------------------------------------------------------
# 2. ITM のヒートマップ描画
# ---------------------------------------------------------------------------
def plot_itm_heatmap(itm: np.ndarray, save_path: str, title: str):
    """12×12 ITM をヒートマップとして保存する。"""
    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(itm, cmap="YlOrRd", aspect="equal", origin="lower")

    ax.set_xticks(range(12))
    ax.set_xticklabels(PITCH_CLASSES, fontsize=9)
    ax.set_yticks(range(12))
    ax.set_yticklabels(PITCH_CLASSES, fontsize=9)
    ax.set_xlabel("To pitch class", fontsize=11)
    ax.set_ylabel("From pitch class", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized weight", fontsize=10)

    # セル内に値を表示
    for i in range(12):
        for j in range(12):
            val = itm[i, j]
            if val > 0.001:
                text_color = "white" if val > itm.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=6, color=text_color)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. ITM → 距離行列 (論文 Section 3 の手法)
# ---------------------------------------------------------------------------
def itm_to_distance_matrix(itm: np.ndarray) -> tuple:
    """
    ITM から距離行列を構築する。

    論文の手法:
      - 12 個のピッチクラスノード + 遷移ごとの補助ノード v_{ij} を作る。
      - 各遷移 (i→j, i≠j) に対し、辺 {i, v_{ij}} と {v_{ij}, j} を
        重み w' = m_{ij}/2 で追加（辺の長さ = 1/w'）。
      - 自己ループ (i→i) に対し、辺 {i, v_{ii}} を重み w' = m_{ii} で追加。
      - 全ノード間の最短パス距離を距離行列とする。

    Returns
    -------
    dist_matrix : np.ndarray
        全ノード間の距離行列
    node_labels : list
        各ノードのラベル (ピッチクラス名 or 補助ノード名)
    pitch_class_indices : list
        距離行列内でのピッチクラスノードのインデックス
    """
    G = nx.Graph()

    # ピッチクラスノードを追加
    for i in range(12):
        G.add_node(f"PC_{i}", label=PITCH_CLASSES[i])

    # 補助ノードと辺を追加
    for i in range(12):
        for j in range(12):
            if itm[i, j] > 1e-12:
                aux_node = f"aux_{i}_{j}"
                G.add_node(aux_node, label=f"v({PITCH_CLASSES[i]},{PITCH_CLASSES[j]})")

                if i == j:
                    # 自己ループ: 辺 {i, v_{ii}}, 重み = m_{ii}
                    w_prime = itm[i, i]
                    edge_length = 1.0 / w_prime
                    G.add_edge(f"PC_{i}", aux_node, weight=edge_length)
                else:
                    # 遷移 (i→j): 辺 {i, v_{ij}}, {v_{ij}, j}
                    # 重み w' = m_{ij}/2
                    w_prime = itm[i, j] / 2.0
                    edge_length = 1.0 / w_prime
                    G.add_edge(f"PC_{i}", aux_node, weight=edge_length)
                    G.add_edge(aux_node, f"PC_{j}", weight=edge_length)

    # 全ノード間の最短パス距離を計算
    nodes = list(G.nodes())
    n = len(nodes)
    node_index = {node: idx for idx, node in enumerate(nodes)}

    # 距離行列を初期化 (∞)
    dist_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(dist_matrix, 0.0)

    # NetworkX で最短パス距離を計算
    for source in nodes:
        lengths = nx.single_source_dijkstra_path_length(G, source, weight="weight")
        for target, dist in lengths.items():
            idx_s = node_index[source]
            idx_t = node_index[target]
            dist_matrix[idx_s, idx_t] = dist

    # ノードラベルを取得
    node_labels = [G.nodes[n_].get("label", n_) for n_ in nodes]

    # ピッチクラスノードのインデックス
    pitch_class_indices = [node_index[f"PC_{i}"] for i in range(12)]

    return dist_matrix, node_labels, pitch_class_indices


# ---------------------------------------------------------------------------
# 4. Persistent Homology の計算
# ---------------------------------------------------------------------------
def compute_persistent_homology(dist_matrix: np.ndarray, maxdim: int = 1):
    """
    距離行列から Vietoris-Rips persistent homology を計算する。

    Parameters
    ----------
    dist_matrix : np.ndarray
        距離行列 (対称)
    maxdim : int
        PH の最大次元 (default: 1 → H0, H1)

    Returns
    -------
    result : dict
        ripser の出力 (dgms キーに persistence diagrams がある)
    """
    # ∞ を有限の大きな値に置き換える（ripser は ∞ を扱えない）
    finite_max = dist_matrix[np.isfinite(dist_matrix)].max()
    dm = np.copy(dist_matrix)
    dm[np.isinf(dm)] = finite_max * 10

    result = ripser(dm, maxdim=maxdim, distance_matrix=True)
    return result


# ---------------------------------------------------------------------------
# 5. Persistence Diagram / Barcode の描画
# ---------------------------------------------------------------------------
def plot_persistence_diagram(result, save_path: str, title: str):
    """Persistence Diagram を保存する。"""
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_diagrams(result["dgms"], ax=ax, show=False)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_barcode(result, save_path: str, title: str):
    """Persistence Barcode を保存する。"""
    dgms = result["dgms"]
    fig, axes = plt.subplots(len(dgms), 1, figsize=(10, 4 * len(dgms)),
                              squeeze=False)

    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for dim_idx, dgm in enumerate(dgms):
        ax = axes[dim_idx, 0]
        if len(dgm) == 0:
            ax.set_title(f"H{dim_idx} — (empty)", fontsize=11)
            continue

        # ∞ を有限の最大死亡時間 + 少し余裕に置換
        finite_deaths = dgm[:, 1][np.isfinite(dgm[:, 1])]
        max_death = finite_deaths.max() if len(finite_deaths) > 0 else 1.0
        inf_replacement = max_death * 1.2

        bars = []
        for birth, death in dgm:
            d = death if np.isfinite(death) else inf_replacement
            bars.append((birth, d - birth))

        # 長さでソート（長い順）
        bars.sort(key=lambda x: x[1], reverse=True)

        for bar_idx, (birth, length) in enumerate(bars):
            color = colors[dim_idx % len(colors)]
            ax.barh(bar_idx, length, left=birth, height=0.7,
                    color=color, alpha=0.8, edgecolor="black", linewidth=0.3)

        ax.set_ylabel("Feature index", fontsize=10)
        ax.set_xlabel("Filtration value", fontsize=10)
        ax.set_title(f"H{dim_idx} Barcode", fontsize=11, fontweight="bold")
        ax.invert_yaxis()

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. 統計量の計算
# ---------------------------------------------------------------------------
def compute_ph_statistics(result) -> dict:
    """
    Persistence Diagram から統計量を算出する。

    - persistent mean (m_p)
    - persistent std dev (sd_p)
    - persistent entropy (e_p)
    - 各次元のバーの数
    """
    stats = {}

    for dim_idx, dgm in enumerate(result["dgms"]):
        dim_key = f"H{dim_idx}"

        if len(dgm) == 0:
            stats[dim_key] = {
                "n_bars": 0,
                "n_finite": 0,
                "n_infinite": 0,
                "persistent_mean": 0.0,
                "persistent_std": 0.0,
                "persistent_entropy": 0.0,
            }
            continue

        # 有限・無限バーの分離
        finite_mask = np.isfinite(dgm[:, 1])
        finite_bars = dgm[finite_mask]
        infinite_bars = dgm[~finite_mask]

        # 有限バーの長さ
        if len(finite_bars) > 0:
            finite_lengths = finite_bars[:, 1] - finite_bars[:, 0]
        else:
            finite_lengths = np.array([])

        # Persistent Mean, Std
        if len(finite_lengths) > 0:
            p_mean = float(np.mean(finite_lengths))
            p_std = float(np.std(finite_lengths))
        else:
            p_mean = 0.0
            p_std = 0.0

        # Persistent Entropy
        # 無限バーは m+1 に置き換え (m = max finite death)
        finite_deaths = dgm[:, 1][finite_mask]
        m = float(finite_deaths.max()) if len(finite_deaths) > 0 else 0.0

        all_lengths = []
        for birth, death in dgm:
            if np.isfinite(death):
                all_lengths.append(death - birth)
            else:
                all_lengths.append((m + 1) - birth)

        all_lengths = np.array(all_lengths)
        total = all_lengths.sum()

        if total > 0:
            probs = all_lengths / total
            probs = probs[probs > 0]
            p_entropy = float(-np.sum(probs * np.log(probs)))
        else:
            p_entropy = 0.0

        stats[dim_key] = {
            "n_bars": len(dgm),
            "n_finite": int(finite_mask.sum()),
            "n_infinite": int((~finite_mask).sum()),
            "persistent_mean": round(p_mean, 6),
            "persistent_std": round(p_std, 6),
            "persistent_entropy": round(p_entropy, 6),
        }

    return stats


# ---------------------------------------------------------------------------
# 7. ジャンル間比較のプロット
# ---------------------------------------------------------------------------
def plot_genre_comparison(all_stats: dict, save_dir: str):
    """
    ジャンルごとの scatter plot (Entropy vs Mean) を描画する。
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    genre_colors = {"sanjuanito": "#1976D2", "tobas": "#D32F2F"}
    genre_markers = {"sanjuanito": "o", "tobas": "s"}

    for dim_idx, dim_key in enumerate(["H0", "H1"]):
        ax = axes[dim_idx]

        for genre in GENRES:
            if genre not in all_stats:
                continue

            means = []
            entropies = []
            labels = []

            for song_name, stats in all_stats[genre].items():
                if dim_key in stats:
                    means.append(stats[dim_key]["persistent_mean"])
                    entropies.append(stats[dim_key]["persistent_entropy"])
                    labels.append(song_name[:20])  # 短縮

            if means:
                ax.scatter(
                    means, entropies,
                    c=genre_colors.get(genre, "gray"),
                    marker=genre_markers.get(genre, "o"),
                    s=100, alpha=0.8, edgecolors="black", linewidths=0.5,
                    label=genre,
                    zorder=5,
                )
                for lbl, mx, ey in zip(labels, means, entropies):
                    ax.annotate(
                        lbl, (mx, ey),
                        fontsize=6, alpha=0.7,
                        textcoords="offset points", xytext=(5, 5),
                    )

        ax.set_xlabel("Persistent Mean", fontsize=12)
        ax.set_ylabel("Persistent Entropy", fontsize=12)
        ax.set_title(f"{dim_key}: Persistent Mean vs Entropy", fontsize=13,
                     fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Sanjuanito vs Tobas — Topological Footprint Comparison",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    save_path = os.path.join(save_dir, "genre_comparison_entropy_vs_mean.png")
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")

    # Std vs Mean のプロットも作成
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

    for dim_idx, dim_key in enumerate(["H0", "H1"]):
        ax = axes2[dim_idx]

        for genre in GENRES:
            if genre not in all_stats:
                continue

            means = []
            stds = []
            labels = []

            for song_name, stats in all_stats[genre].items():
                if dim_key in stats:
                    means.append(stats[dim_key]["persistent_mean"])
                    stds.append(stats[dim_key]["persistent_std"])
                    labels.append(song_name[:20])

            if means:
                ax.scatter(
                    means, stds,
                    c=genre_colors.get(genre, "gray"),
                    marker=genre_markers.get(genre, "o"),
                    s=100, alpha=0.8, edgecolors="black", linewidths=0.5,
                    label=genre,
                    zorder=5,
                )
                for lbl, mx, sy in zip(labels, means, stds):
                    ax.annotate(
                        lbl, (mx, sy),
                        fontsize=6, alpha=0.7,
                        textcoords="offset points", xytext=(5, 5),
                    )

        ax.set_xlabel("Persistent Mean", fontsize=12)
        ax.set_ylabel("Persistent Std Dev", fontsize=12)
        ax.set_title(f"{dim_key}: Persistent Mean vs Std Dev", fontsize=13,
                     fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    fig2.suptitle("Sanjuanito vs Tobas — Mean vs Std Dev Comparison",
                  fontsize=15, fontweight="bold", y=1.02)
    fig2.tight_layout()

    save_path2 = os.path.join(save_dir, "genre_comparison_std_vs_mean.png")
    fig2.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [SAVED] {save_path2}")


# ---------------------------------------------------------------------------
# 8. PCA 分析
# ---------------------------------------------------------------------------
def run_pca_analysis(all_stats: dict, all_itms: dict, save_dir: str):
    """
    論文の手法に基づく PCA 分析。

    論文では各曲が4パートを持ち、各パートから6統計量 (m_H0, sd_H0, e_H0,
    m_H1, sd_H1, e_H1) を取得して 4×6=24 次元ベクトルを構成するが、
    本データは各MIDIが1パートのため、以下の3種類のPCAを実行する:

    1. 6D PH PCA: 1パート × 6統計量
    2. 24D ITM PCA: ITM行周辺分布(12) + 列周辺分布(12)
    3. 30D Combined PCA: 6D PH + 24D ITM
    """
    genre_colors = {"sanjuanito": "#1976D2", "tobas": "#D32F2F"}
    genre_markers = {"sanjuanito": "o", "tobas": "s"}

    # --- 特徴量ベクトルの構築 ---
    labels = []      # 曲名
    genres = []      # ジャンル
    ph_vectors = []  # 6D PH 統計量
    itm_vectors = [] # 24D ITM 周辺分布

    for genre in GENRES:
        if genre not in all_stats:
            continue
        for song_name in all_stats[genre]:
            s = all_stats[genre][song_name]
            # 6D: (m_H0, sd_H0, e_H0, m_H1, sd_H1, e_H1)
            ph_vec = [
                s["H0"]["persistent_mean"],
                s["H0"]["persistent_std"],
                s["H0"]["persistent_entropy"],
                s["H1"]["persistent_mean"],
                s["H1"]["persistent_std"],
                s["H1"]["persistent_entropy"],
            ]
            ph_vectors.append(ph_vec)

            # 24D: ITM 行周辺分布(12) + 列周辺分布(12)
            itm = all_itms[genre][song_name]
            row_marginals = itm.sum(axis=1)  # shape (12,)
            col_marginals = itm.sum(axis=0)  # shape (12,)
            itm_vec = list(row_marginals) + list(col_marginals)
            itm_vectors.append(itm_vec)

            labels.append(song_name)
            genres.append(genre)

    ph_matrix = np.array(ph_vectors)    # (N, 6)
    itm_matrix = np.array(itm_vectors)  # (N, 24)
    combined_matrix = np.hstack([ph_matrix, itm_matrix])  # (N, 30)

    # --- PCA 実行 ---
    pca_configs = [
        ("6D PH Statistics", ph_matrix, "pca_6d_ph"),
        ("24D ITM Marginals", itm_matrix, "pca_24d_itm"),
        ("30D Combined (PH + ITM)", combined_matrix, "pca_30d_combined"),
    ]

    for title_base, data, filename_prefix in pca_configs:
        # StandardScaler で正規化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # PCA (2D)
        pca = PCA(n_components=min(2, data_scaled.shape[1]))
        coords_2d = pca.fit_transform(data_scaled)
        ev_ratio = pca.explained_variance_ratio_

        # --- 2D Scatter Plot ---
        fig, ax = plt.subplots(figsize=(10, 8))

        for genre in GENRES:
            mask = [g == genre for g in genres]
            idx = [i for i, m in enumerate(mask) if m]
            if not idx:
                continue
            x = coords_2d[idx, 0]
            y = coords_2d[idx, 1]
            ax.scatter(
                x, y,
                c=genre_colors.get(genre, "gray"),
                marker=genre_markers.get(genre, "o"),
                s=120, alpha=0.85, edgecolors="black", linewidths=0.5,
                label=genre, zorder=5,
            )
            for i_idx in idx:
                short_name = labels[i_idx].split("_100")[0]  # 短縮
                ax.annotate(
                    short_name, (coords_2d[i_idx, 0], coords_2d[i_idx, 1]),
                    fontsize=7, alpha=0.75,
                    textcoords="offset points", xytext=(6, 6),
                )

        ax.set_xlabel(f"PC1 ({ev_ratio[0]*100:.1f}% variance)", fontsize=12)
        ax.set_ylabel(f"PC2 ({ev_ratio[1]*100:.1f}% variance)", fontsize=12)
        ax.set_title(f"PCA — {title_base}\nSanjuanito vs Tobas",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=12, loc="best")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.5)
        fig.tight_layout()

        save_path = os.path.join(save_dir, f"{filename_prefix}_2d.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  [SAVED] {save_path}")

        # --- PCA Loadings (成分負荷量) ---
        if data.shape[1] <= 30:
            _plot_pca_loadings(pca, data, title_base, filename_prefix,
                               save_dir, ph_matrix.shape[1] if data.shape[1] == 6 else None)

        # --- 寄与率をテキスト出力 ---
        print(f"    {title_base}:")
        print(f"      PC1: {ev_ratio[0]*100:.1f}%, PC2: {ev_ratio[1]*100:.1f}%")
        print(f"      Cumulative: {sum(ev_ratio)*100:.1f}%")

    # --- PCA 結果をテキストに保存 ---
    pca_summary_path = os.path.join(save_dir, "pca_summary.txt")
    with open(pca_summary_path, "w", encoding="utf-8") as f:
        f.write("PCA Analysis Summary\n")
        f.write(f"{'='*60}\n\n")

        for title_base, data, filename_prefix in pca_configs:
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            n_comp = min(data_scaled.shape[1], data_scaled.shape[0])
            pca_full = PCA(n_components=n_comp)
            pca_full.fit(data_scaled)

            f.write(f"{title_base} (dim={data.shape[1]}):\n")
            f.write(f"{'-'*40}\n")
            cumsum = 0
            for i, (ev, evr) in enumerate(
                zip(pca_full.explained_variance_, pca_full.explained_variance_ratio_)
            ):
                cumsum += evr
                f.write(f"  PC{i+1}: variance={ev:.4f}, "
                        f"ratio={evr*100:.1f}%, cumulative={cumsum*100:.1f}%\n")
            f.write("\n")

            # 特徴量ベクトルの値
            f.write(f"  Feature vectors:\n")
            for i, (lbl, g) in enumerate(zip(labels, genres)):
                vec_str = ", ".join(f"{v:.4f}" for v in data_scaled[i][:6])
                if data.shape[1] > 6:
                    vec_str += ", ..."
                f.write(f"    [{g}] {lbl}: [{vec_str}]\n")
            f.write("\n\n")

    print(f"  [SAVED] {pca_summary_path}")


def _plot_pca_loadings(pca, data, title_base, filename_prefix, save_dir,
                       n_ph_features=None):
    """PCA の成分負荷量をバープロットで描画する。"""
    n_features = data.shape[1]

    # 特徴量名
    if n_features == 6:
        feat_names = ["m_H0", "sd_H0", "e_H0", "m_H1", "sd_H1", "e_H1"]
    elif n_features == 24:
        feat_names = ([f"row_{pc}" for pc in PITCH_CLASSES]
                      + [f"col_{pc}" for pc in PITCH_CLASSES])
    elif n_features == 30:
        feat_names = (["m_H0", "sd_H0", "e_H0", "m_H1", "sd_H1", "e_H1"]
                      + [f"row_{pc}" for pc in PITCH_CLASSES]
                      + [f"col_{pc}" for pc in PITCH_CLASSES])
    else:
        feat_names = [f"f{i}" for i in range(n_features)]

    n_components = min(2, len(pca.components_))
    fig, axes = plt.subplots(n_components, 1,
                              figsize=(max(12, n_features * 0.5), 4 * n_components),
                              squeeze=False)

    for pc_idx in range(n_components):
        ax = axes[pc_idx, 0]
        loadings = pca.components_[pc_idx]
        colors = ["#1976D2" if v >= 0 else "#D32F2F" for v in loadings]
        ax.bar(range(n_features), loadings, color=colors, alpha=0.8,
               edgecolor="black", linewidth=0.3)
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Loading", fontsize=10)
        ax.set_title(f"PC{pc_idx+1} Loadings "
                     f"({pca.explained_variance_ratio_[pc_idx]*100:.1f}%)",
                     fontsize=11, fontweight="bold")
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(f"PCA Loadings — {title_base}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(save_dir, f"{filename_prefix}_loadings.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {save_path}")


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------
def main():
    all_stats = {}
    all_itms = {}   # PCA 分析用に ITM を保存

    for genre in GENRES:
        midi_genre_dir = os.path.join(MIDI_DIR, genre)

        if not os.path.isdir(midi_genre_dir):
            print(f"[WARN] ディレクトリが見つかりません: {midi_genre_dir}")
            continue

        midi_files = sorted(
            f for f in os.listdir(midi_genre_dir)
            if f.lower().endswith(".mid")
        )

        # 転調版（ファイル名末尾に _+N or _-N があるもの）をスキップ
        # 例: ponchito_100-7500hz_midi_1990hz_+3.mid → スキップ
        # 例: ponchito_100-7500hz_midi_1990hz.mid   → 処理対象
        transposition_pattern = re.compile(r'_[+-]\d+\.mid$', re.IGNORECASE)
        midi_files = [
            f for f in midi_files
            if not transposition_pattern.search(f)
        ]

        print(f"\n{'='*60}")
        print(f"  Genre: {genre}  ({len(midi_files)} files)")
        print(f"{'='*60}")

        all_stats[genre] = {}
        all_itms[genre] = {}

        for midi_file in midi_files:
            midi_path = os.path.join(midi_genre_dir, midi_file)
            base_name = os.path.splitext(midi_file)[0]

            # 出力ディレクトリ（ファイル名をそのまま使用）
            song_dir = os.path.join(OUTPUT_DIR, genre, base_name)
            os.makedirs(song_dir, exist_ok=True)

            print(f"\n--- {midi_file} ---")

            # 1. ITM 構築
            print("  [1] Building Intervallic Transition Matrix...")
            itm = build_itm(midi_path)

            # 2. ITM ヒートマップ
            heatmap_path = os.path.join(song_dir, "itm_heatmap.png")
            plot_itm_heatmap(
                itm, heatmap_path,
                title=f"Intervallic Transition Matrix\n{base_name}"
            )
            print(f"  [SAVED] {heatmap_path}")

            # 3. 距離行列の構築
            print("  [2] Computing distance matrix...")
            dist_matrix, node_labels, pc_indices = itm_to_distance_matrix(itm)

            # 距離行列（ピッチクラスのみ）のヒートマップも保存
            pc_dist = dist_matrix[np.ix_(pc_indices, pc_indices)]
            fig_dist, ax_dist = plt.subplots(figsize=(8, 7))
            # ∞ 表示のため有限最大値で置換して表示
            pc_dist_display = np.copy(pc_dist)
            finite_vals = pc_dist_display[np.isfinite(pc_dist_display)]
            if len(finite_vals) > 0:
                max_finite = finite_vals.max()
                pc_dist_display[np.isinf(pc_dist_display)] = max_finite * 1.5
            else:
                pc_dist_display[np.isinf(pc_dist_display)] = 100

            im_dist = ax_dist.imshow(pc_dist_display, cmap="viridis",
                                      aspect="equal", origin="lower")
            ax_dist.set_xticks(range(12))
            ax_dist.set_xticklabels(PITCH_CLASSES, fontsize=9)
            ax_dist.set_yticks(range(12))
            ax_dist.set_yticklabels(PITCH_CLASSES, fontsize=9)
            ax_dist.set_title(f"Pitch Class Distance Matrix\n{base_name}",
                              fontsize=13, fontweight="bold")
            fig_dist.colorbar(im_dist, ax=ax_dist, fraction=0.046, pad=0.04,
                              label="Shortest path distance")
            fig_dist.tight_layout()
            dist_heatmap_path = os.path.join(song_dir, "distance_matrix_heatmap.png")
            fig_dist.savefig(dist_heatmap_path, dpi=150)
            plt.close(fig_dist)
            print(f"  [SAVED] {dist_heatmap_path}")

            # 4. Persistent Homology の計算
            print("  [3] Computing Persistent Homology...")
            ph_result = compute_persistent_homology(dist_matrix, maxdim=1)

            # 5. Persistence Diagram
            pd_path = os.path.join(song_dir, "persistence_diagram.png")
            plot_persistence_diagram(
                ph_result, pd_path,
                title=f"Persistence Diagram\n{base_name}"
            )
            print(f"  [SAVED] {pd_path}")

            # 6. Barcode
            bc_path = os.path.join(song_dir, "barcode.png")
            plot_barcode(
                ph_result, bc_path,
                title=f"Persistence Barcode — {base_name}"
            )
            print(f"  [SAVED] {bc_path}")

            # 7. 統計量
            stats = compute_ph_statistics(ph_result)
            all_stats[genre][base_name] = stats
            all_itms[genre][base_name] = itm

            # 統計量をテキストファイルに書き出し
            stats_path = os.path.join(song_dir, "analysis_results.txt")
            with open(stats_path, "w", encoding="utf-8") as f:
                f.write(f"Persistent Homology Analysis Results\n")
                f.write(f"Song: {base_name}\n")
                f.write(f"Genre: {genre}\n")
                f.write(f"{'='*50}\n\n")

                f.write(f"Intervallic Transition Matrix (ITM):\n")
                f.write(f"  Non-zero entries: "
                        f"{np.count_nonzero(itm)} / 144\n")
                f.write(f"  ITM entropy (Shannon): "
                        f"{_shannon_entropy(itm):.6f}\n\n")

                for dim_key, dim_stats in stats.items():
                    f.write(f"{dim_key}:\n")
                    for k, v in dim_stats.items():
                        f.write(f"  {k}: {v}\n")
                    f.write("\n")

                # Persistence Diagram の raw データも書き出し
                f.write(f"\nRaw Persistence Diagrams:\n")
                f.write(f"{'-'*50}\n")
                for dim_idx, dgm in enumerate(ph_result["dgms"]):
                    f.write(f"H{dim_idx} ({len(dgm)} bars):\n")
                    for birth, death in dgm:
                        death_str = f"{death:.6f}" if np.isfinite(death) else "inf"
                        f.write(f"  [{birth:.6f}, {death_str})\n")
                    f.write("\n")

            print(f"  [SAVED] {stats_path}")

            # 統計量を表示
            for dim_key, dim_stats in stats.items():
                print(f"  {dim_key}: mean={dim_stats['persistent_mean']:.4f}, "
                      f"std={dim_stats['persistent_std']:.4f}, "
                      f"entropy={dim_stats['persistent_entropy']:.4f}, "
                      f"bars={dim_stats['n_bars']} "
                      f"(finite={dim_stats['n_finite']}, "
                      f"inf={dim_stats['n_infinite']})")

    # ジャンル間比較
    print(f"\n{'='*60}")
    print(f"  Genre Comparison")
    print(f"{'='*60}")
    plot_genre_comparison(all_stats, OUTPUT_DIR)

    # PCA 分析
    print(f"\n{'='*60}")
    print(f"  PCA Analysis")
    print(f"{'='*60}")
    run_pca_analysis(all_stats, all_itms, OUTPUT_DIR)

    # 全体の統計量を JSON で保存
    json_path = os.path.join(OUTPUT_DIR, "all_statistics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] {json_path}")

    # サマリーテキスト
    summary_path = os.path.join(OUTPUT_DIR, "comparison_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Persistent Homology: Sanjuanito vs Tobas Summary\n")
        f.write(f"{'='*60}\n\n")

        for genre in GENRES:
            if genre not in all_stats:
                continue
            f.write(f"Genre: {genre}\n")
            f.write(f"{'-'*40}\n")

            for song, stats in all_stats[genre].items():
                f.write(f"\n  {song}:\n")
                for dim_key, dim_stats in stats.items():
                    f.write(f"    {dim_key}: "
                            f"mean={dim_stats['persistent_mean']:.6f}, "
                            f"std={dim_stats['persistent_std']:.6f}, "
                            f"entropy={dim_stats['persistent_entropy']:.6f}\n")

            f.write("\n")

        # ジャンル平均を計算
        f.write(f"\n{'='*60}\n")
        f.write(f"Genre Averages\n")
        f.write(f"{'='*60}\n\n")

        for genre in GENRES:
            if genre not in all_stats:
                continue
            f.write(f"Genre: {genre}\n")

            for dim_key in ["H0", "H1"]:
                means = [s[dim_key]["persistent_mean"]
                         for s in all_stats[genre].values()
                         if dim_key in s]
                stds = [s[dim_key]["persistent_std"]
                        for s in all_stats[genre].values()
                        if dim_key in s]
                ents = [s[dim_key]["persistent_entropy"]
                        for s in all_stats[genre].values()
                        if dim_key in s]

                if means:
                    f.write(f"  {dim_key}:\n")
                    f.write(f"    avg persistent_mean:    "
                            f"{np.mean(means):.6f} ± {np.std(means):.6f}\n")
                    f.write(f"    avg persistent_std:     "
                            f"{np.mean(stds):.6f} ± {np.std(stds):.6f}\n")
                    f.write(f"    avg persistent_entropy: "
                            f"{np.mean(ents):.6f} ± {np.std(ents):.6f}\n")

            f.write("\n")

    print(f"  [SAVED] {summary_path}")
    print("\n完了しました。")


def _shannon_entropy(matrix: np.ndarray) -> float:
    """行列のシャノンエントロピーを計算する。"""
    flat = matrix.flatten()
    flat = flat[flat > 0]
    if len(flat) == 0:
        return 0.0
    return float(-np.sum(flat * np.log(flat)))


if __name__ == "__main__":
    main()
