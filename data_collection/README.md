# 環境設定ファイルの説明
## client_secret_tv.json

このファイルは、YouTube API などを使うための OAuth 2.0 クライアントID 情報を保存します。

### 作成手順
1. Google Cloud Console にアクセスします。
2. 左側メニューから「API とサービス」→「認証情報」を開きます。
3. 「認証情報を作成」をクリックします。
4. 「OAuth クライアント ID」を選択します。
5. 「アプリケーションの種類」で「テレビと入力が限られたデバイス」を選びます。
6. 名前を入力します（例: `Music TDA Tobas Sanjuanito TV`）。
7. 作成すると、JSON 形式のクライアント情報が表示されます。
8. 表示された内容をそのまま `data_collection/client_secret_tv.json` に保存します。

### JSON の中身の例
```json
{
  "installed": {
    "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
    "project_id": "your-project-id",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_secret": "YOUR_CLIENT_SECRET",
    "redirect_uris": [
      "urn:ietf:wg:oauth:2.0:oob",
      "http://localhost"
    ]
  }
}
```

> `client_secret_tv.json` は、アプリが Google の認証画面を開いてユーザー認証を行うときに必要です。保存場所とファイル名を変えないようにしてください。

## browser.json

このファイルは、ブラウザでログインした YouTube Music セッションを再利用するための認証情報を保存します。

### 取得手順
1. Firefox で YouTube Music にログインします。
2. F12 で開発者ツールを開き、ネットワークタブを表示します。
3. `ライブラリ` や `ホーム` をクリックして、通信を発生させます。
4. ネットワークフィルターで `browse` を入力し、`browse?...` の POST リクエストを表示します。
5. そのリクエストを右クリックし、「リクエストヘッダーをコピー」を選びます。

### raw を保存した後の処理
1. `POST` 行やヘッダー行を含まない、純粋なリクエストヘッダーだけを保存します。
2. 端末で次を実行します。

   ```powershell
   ytmusicapi browser
   ```

3. 画面の指示に従い、コピーしたリクエストヘッダーを貼り付けます。
4. `browser.json` が作成されます。

### 手動で `browser.json` を作成する場合
`browser.json` に次のようなヘッダー情報を保存します。`Cookie` の値はコピーした認証 cookie で置き換えてください。

```json
{
  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0",
  "Accept": "*/*",
  "Accept-Language": "en-US,en;q=0.5",
  "Content-Type": "application/json",
  "X-Goog-AuthUser": "0",
  "x-origin": "https://music.youtube.com",
  "Cookie": "PASTE_COOKIE"
}
```

### Python から使う方法
`browser.json` を作成したら、プロジェクト内で次のように読み込みます。

```python
from ytmusicapi import YTMusic

ytmusic = YTMusic('data_collection/browser.json')
```

> これにより、ブラウザセッションの認証情報が再利用され、YouTube Music の認証済みリクエストを実行できます。
