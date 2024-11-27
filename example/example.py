
from traceloop.sdk import Traceloop

Traceloop.init(
    # デモ用なので、batch processorではなく即時でトレースデータを送る
    disable_batch=True,
    # アプリケーションの名前
    app_name="Example",
    # 独自属性の追加
    resource_attributes={"env": "dev", "version": "1.0.0"},
    traceloop_sync_enabled=True,
)
