"""
generate_chroma_images.py

midi/ フォルダ内の sanjuanito と tobas の MIDI ファイルから
オクターブをつぶした symbolic chroma の画像を生成するスクリプト。

各 MIDI ファイルについて:
  - オリジナルの chroma 画像
  - 半音を +1 ~ +6 上げたバージョン
  - 半音を -1 ~ -6 下げたバージョン
を chroma/(sanjuanito or tobas)/ に保存する。
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI なしで画像生成するため
import matplotlib.pyplot as plt
import pretty_midi


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIDI_DIR = os.path.join(SCRIPT_DIR, "midi")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "chroma")
GENRES = ["sanjuanito", "tobas"]
SEMITONE_SHIFTS = list(range(-6, 7))  # -6, -5, ..., 0, ..., +5, +6
FS = 100  # サンプリング周波数 (Hz)


# ---------------------------------------------------------------------------
# コア関数 (notebook: octave_invariant_midi.ipynb より)
# ---------------------------------------------------------------------------
def compute_midi_chroma(midi_path: str, fs: int = FS, semitone_shift: int = 0):
    """
    MIDI ファイルからシンボリック・クロマグラム（オクターブ不変）を計算する。

    Parameters
    ----------
    midi_path : str
        MIDI ファイルのパス
    fs : int
        ピアノロールのサンプリング周波数 (default 100 Hz = 10 ms 精度)
    semitone_shift : int
        半音でのトランスポーズ量。正で上げ、負で下げる。

    Returns
    -------
    chroma : np.ndarray (12, T) or None
        正規化済みクロマグラム（エラー時は None）
    fs : int
        使用したサンプリング周波数
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"  [ERROR] MIDI 読み込み失敗: {e}")
        return None, None

    # 半音シフトを適用 ― ノート番号を直接ずらす
    if semitone_shift != 0:
        for instrument in pm.instruments:
            for note in instrument.notes:
                note.pitch = note.pitch + semitone_shift
                # MIDI ノート番号は 0–127 の範囲に収める
                note.pitch = max(0, min(127, note.pitch))

    # ピアノロール取得 (128 pitches × time)
    piano_roll = pm.get_piano_roll(fs=fs)

    n_pitch = piano_roll.shape[0]
    n_time = piano_roll.shape[1]
    chroma = np.zeros((12, n_time))

    for p in range(n_pitch):
        if np.sum(piano_roll[p, :]) > 0:
            chroma_idx = p % 12
            chroma[chroma_idx, :] += piano_roll[p, :]

    # 正規化 (0–1)
    if np.max(chroma) > 0:
        chroma = chroma / np.max(chroma)

    return chroma, fs


def save_chroma_image(
    chroma: np.ndarray,
    fs: int,
    save_path: str,
    title: str = "Symbolic Chroma (Octave Invariant)",
):
    """
    クロマグラムを画像ファイルとして保存する。

    Parameters
    ----------
    chroma : np.ndarray (12, T)
    fs : int
    save_path : str
        保存先のファイルパス (.png)
    title : str
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    duration = chroma.shape[1] / fs
    im = ax.imshow(
        chroma,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        extent=[0, duration, 0, 12],
    )

    ax.set_yticks(np.arange(0.5, 12.5))
    ax.set_yticklabels(
        ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch Class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Velocity / Intensity")
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------
def main():
    for genre in GENRES:
        midi_genre_dir = os.path.join(MIDI_DIR, genre)
        output_genre_dir = os.path.join(OUTPUT_DIR, genre)

        if not os.path.isdir(midi_genre_dir):
            print(f"[WARN] ディレクトリが見つかりません: {midi_genre_dir}")
            continue

        midi_files = sorted(
            f for f in os.listdir(midi_genre_dir) if f.lower().endswith(".mid")
        )
        print(f"\n=== {genre} ({len(midi_files)} files) ===")

        for midi_file in midi_files:
            midi_path = os.path.join(midi_genre_dir, midi_file)
            base_name = os.path.splitext(midi_file)[0]  # 拡張子なしのファイル名

            for shift in SEMITONE_SHIFTS:
                # ファイル名を決定
                if shift == 0:
                    png_name = f"{base_name}.png"
                    title_suffix = "(original)"
                elif shift > 0:
                    png_name = f"{base_name}_+{shift}.png"
                    title_suffix = f"(+{shift} semitones)"
                else:
                    png_name = f"{base_name}_{shift}.png"
                    title_suffix = f"({shift} semitones)"

                save_path = os.path.join(output_genre_dir, png_name)

                title = f"Symbolic Chroma: {midi_file} {title_suffix}"

                chroma, fs_out = compute_midi_chroma(
                    midi_path, fs=FS, semitone_shift=shift
                )
                if chroma is None:
                    print(f"  [SKIP] {midi_file} shift={shift}")
                    continue

                save_chroma_image(chroma, fs_out, save_path, title=title)
                print(f"  saved: {save_path}")

    print("\n完了しました。")


if __name__ == "__main__":
    main()
