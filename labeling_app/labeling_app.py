import streamlit as st
import pandas as pd
from pathlib import Path

# テスト用のCSV
TEST_CSV = Path('test_tunes.csv')

# CSVファイルのパス（sanjuanitoとtobasを統合）
SANJUANITO_CSV = Path('sanjuanito_20260418_2309_filtered_country_remix_composer.csv')
TOBAS_CSV = Path('tobas_20260418_2311_filtered_country_remix_composer.csv')

st.title("音楽ジャンルラベル付けアプリ")
st.write("各曲を聴いて、ジャンルをラベル付けしてください。研究にご協力ありがとうございます！")

# アノテーターの名前入力
annotator_name = st.text_input("あなたの名前を入力してください（例: Taro Yamada）", "")

# ラベル付け結果の保存先
OUTPUT_PATH = Path(f'{annotator_name}_labeling_results.csv')

if not annotator_name:
    st.warning("名前を入力してください。")
    st.stop()

# CSV統合
dfs = []
if SANJUANITO_CSV.exists():
    df_s = pd.read_csv(SANJUANITO_CSV, dtype=str).fillna('')
    #df_s = df_s.sample(n=min(10, len(df_s)), random_state=42)  # ランダム10曲
    df_s['expected_genre'] = 'Sanjuanito'
    dfs.append(df_s)
if TOBAS_CSV.exists():
    df_t = pd.read_csv(TOBAS_CSV, dtype=str).fillna('')
    #df_t = df_t.sample(n=min(10, len(df_t)), random_state=42)  # ランダム10曲
    df_t['expected_genre'] = 'Tobas'
    dfs.append(df_t)

if not dfs:
    st.error("CSVファイルが見つかりません。")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# セッション状態で現在のインデックスを管理
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# テストを突破したらラベリングをやる
# 突破できなかったら "残念！全問正解しないとラベリングできません" と表示して終了する

# テスト用データを読み込む
if TEST_CSV.exists():
    test_df = pd.read_csv(TEST_CSV, dtype=str).fillna('')
else:
    test_df = pd.DataFrame()

# セッション状態でテスト状態を管理
if 'test_passed' not in st.session_state:
    st.session_state.test_passed = False
if 'test_answers' not in st.session_state:
    st.session_state.test_answers = {}
if 'test_question_index' not in st.session_state:
    st.session_state.test_question_index = 0

# テストフェーズ
if not st.session_state.test_passed and len(test_df) > 0:
    st.header("📝 ジャンル判定テスト")
    st.write("ラベル付けを開始する前に、簡単なテストを受けてください。各曲のジャンルを正しく判定できるか確認します。")
    
    # 現在の問題を表示
    if st.session_state.test_question_index < len(test_df):
        test_row = test_df.iloc[st.session_state.test_question_index]
        st.subheader(f"問題 {st.session_state.test_question_index + 1} / {len(test_df)}")
        st.write(f"**タイトル:** {test_row['title']}")
        st.write(f"**アーティスト:** {test_row['artist']}")
        
        # YouTube埋め込み（真ん中から下を表示）
        url = test_row['url']
        if 'youtube.com' in url or 'youtu.be' in url:
            video_id = url.split('v=')[1].split('&')[0] if 'v=' in url else url.split('/')[-1]
            # CSSでクロップして真ん中から下だけ表示
            st.markdown(f"""
            <div style="width: 300px; height: 160px; overflow: hidden; margin: 0; padding: 0;">
                <iframe width="300" height="320" src="https://www.youtube.com/embed/{video_id}?controls=1" 
                frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen style="margin-top: -160px;"></iframe>
            </div>
            """, unsafe_allow_html=True)
            st.info("🔊 表示されている画面で曲を聴いてください。")
        else:
            st.write("再生できません。")
        
        # ラベル選択
        answer = st.radio(
            "このジャンルは？",
            ["Sanjuanito", "Tobas", "不明"],
            key=f"test_answer_{st.session_state.test_question_index}"
        )
        
        st.session_state.test_answers[st.session_state.test_question_index] = answer
        
        # 次へボタン
        col1, col2 = st.columns(2)
        with col1:
            if st.button("回答して次へ"):
                st.session_state.test_question_index += 1
                st.rerun()
        
        with col2:
            if st.session_state.test_question_index > 0:
                if st.button("戻る"):
                    st.session_state.test_question_index -= 1
                    st.rerun()
    
    # 全問答えた後に確認
    if st.session_state.test_question_index >= len(test_df):
        st.subheader("テスト結果の確認")
        st.write("すべての問題に回答しました。確認ボタンを押してください。")
        
        if st.button("回答を提出"):
            # 全問正解かチェック
            all_correct = True
            for idx, test_row in test_df.iterrows():
                correct_answer = test_row['category'].capitalize()
                if st.session_state.test_answers.get(idx) != correct_answer:
                    all_correct = False
                    break
            
            if all_correct:
                st.session_state.test_passed = True
                st.success("✅ 全問正解！ラベル付けを開始します。")
                st.rerun()
            else:
                st.error("❌ 残念！全問正解しないとラベリングできません。")
                st.stop()

# ラベリングフェーズ（テスト突破後）
if st.session_state.test_passed:
    st.header("🎵 ジャンルラベル付け")

    # 現在の曲を取得
    if st.session_state.current_index < len(df):
        row = df.iloc[st.session_state.current_index]
        st.subheader(f"曲 {st.session_state.current_index + 1} / {len(df)}")
        st.write(f"**タイトル:** {row['title']}")
        st.write(f"**アーティスト:** {row['artist']}")
        st.write(f"**期待ジャンル:** {row['expected_genre']}")

        # YouTube埋め込み（最小サイズで再生ボタンのみ表示）
        url = row['url']
        if 'youtube.com' in url or 'youtu.be' in url:
            video_id = url.split('v=')[1].split('&')[0] if 'v=' in url else url.split('/')[-1]
            # CSSでクロップして真ん中から下だけ表示
            st.markdown(f"""
            <div style="width: 300px; height: 160px; overflow: hidden; margin: 0; padding: 0;">
                <iframe width="300" height="320" src="https://www.youtube.com/embed/{video_id}?controls=1" 
                frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen style="margin-top: -160px;"></iframe>
            </div>
            """, unsafe_allow_html=True)
            st.info("🔊 表示されている画面で曲を聴いてください。")
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