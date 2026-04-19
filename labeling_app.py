import streamlit as st
import pandas as pd
from pathlib import Path

# CSVファイルのパス（sanjuanitoとtobasを統合）
SANJUANITO_CSV = Path('data_collection/sanjuanito_20260418_2309_filtered_country_remix_composer.csv')
TOBAS_CSV = Path('data_collection/tobas_20260418_2311_filtered_country_remix_composer.csv')

# ラベル付け結果の保存先
OUTPUT_PATH = Path('data_collection/labeling_results.csv')

st.title("音楽ジャンルラベル付けアプリ")
st.write("各曲を聴いて、ジャンルをラベル付けしてください。研究にご協力ありがとうございます！")

# アノテーターの名前入力
annotator_name = st.text_input("あなたの名前を入力してください（例: Taro Yamada）", "")

if not annotator_name:
    st.warning("名前を入力してください。")
    st.stop()

# CSV統合（ランダムに10曲ずつサンプリング）
dfs = []
if SANJUANITO_CSV.exists():
    df_s = pd.read_csv(SANJUANITO_CSV, dtype=str).fillna('')
    df_s = df_s.sample(n=min(10, len(df_s)), random_state=42)  # ランダム10曲
    df_s['expected_genre'] = 'Sanjuanito'
    dfs.append(df_s)
if TOBAS_CSV.exists():
    df_t = pd.read_csv(TOBAS_CSV, dtype=str).fillna('')
    df_t = df_t.sample(n=min(10, len(df_t)), random_state=42)  # ランダム10曲
    df_t['expected_genre'] = 'Tobas'
    dfs.append(df_t)

if not dfs:
    st.error("CSVファイルが見つかりません。")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# セッション状態で現在のインデックスを管理
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# 現在の曲を取得
if st.session_state.current_index < len(df):
    row = df.iloc[st.session_state.current_index]
    st.subheader(f"曲 {st.session_state.current_index + 1} / {len(df)}")
    st.write(f"**タイトル:** {row['title']}")
    st.write(f"**アーティスト:** {row['artist']}")
    st.write(f"**期待ジャンル:** {row['expected_genre']}")

    # YouTube埋め込み
    url = row['url']
    if 'youtube.com' in url or 'youtu.be' in url:
        video_id = url.split('v=')[1].split('&')[0] if 'v=' in url else url.split('/')[-1]
        st.video(f"https://www.youtube.com/embed/{video_id}")
    else:
        st.write("再生できません。")

    # ラベル選択
    label = st.radio("ジャンルを選択:", ["Sanjuanito", "Tobas", "不明"], key=f"label_{st.session_state.current_index}")

    # ボタン
    col1, col2 = st.columns(2)
    with col1:
        if st.button("保存して次へ"):
            # 結果を保存
            result = {
                'annotator': annotator_name,
                'index': st.session_state.current_index,
                'title': row['title'],
                'artist': row['artist'],
                'expected_genre': row['expected_genre'],
                'url': row['url'],
                'label': label
            }
            try:
                if OUTPUT_PATH.exists():
                    existing_df = pd.read_csv(OUTPUT_PATH, dtype=str).fillna('')
                    new_df = pd.DataFrame([result])
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined_df = pd.DataFrame([result])
                combined_df.to_csv(OUTPUT_PATH, index=False)
                st.session_state.current_index += 1
                st.rerun()
            except Exception as e:
                st.error(f"保存エラー: {e}")

    with col2:
        if st.button("スキップ"):
            st.session_state.current_index += 1
            st.rerun()

else:
    st.write("すべての曲のラベル付けが完了しました！")
    if OUTPUT_PATH.exists():
        st.download_button("結果をダウンロード", OUTPUT_PATH.read_bytes(), file_name="labeling_results.csv")

st.write("---")
st.write("ご協力ありがとうございました_(._.)_")