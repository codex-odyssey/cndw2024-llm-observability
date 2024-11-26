# 気まぐれ LLM のふるまいを暴け！ - OpenLLMetry を通して見る世界 -

CloudNative Days Winter 2024 で実施した「気まぐれ LLM のふるまいを暴け！ - OpenLLMetry を通して見る世界 -」のデモコードの保管リポジトリです。

## 実行方法

envを設定

```
cp .env.sample .env
```

環境の構築

```
docker compose up -d
```

ブラウザで`localhost:8501`に接続する

### テスト

テストデータの生成

```
export TRACELOOP_LOGGING_ENABLED=TRUE
export TRACELOOP_BASE_URL=http://localhost:4318
python3 app/generate_data.py
```

## 諸注意

このリポジトリ内のアプリケーションはCNDW2024のデモの為に作成しています。
そのため、実装したアプリケーションや各 OSS の設定などは、推奨される設定と異なる場合があります。
ご注意のうえ、ご参照ください。
