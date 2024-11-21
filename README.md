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

テストデータの生成

```
TRACELOOP_BASE_URL: http://localhost:4318
python3 app/generate_data.py
```
