import os
import datetime
import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
import gradio as gr

# 擴充股票與熱門 ETF 對照表（支援中文與代號雙向搜尋）
STOCK_MAP = {
    # 熱門權值股
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
    "2382": "廣達",
    "2308": "台達電",
    "2603": "長榮",
    "2881": "富邦金",
    "2882": "國泰金",
    "2412": "中華電",
    "2002": "中鋼",
    "2891": "中信金",
    "3037": "欣興",
    "2357": "華碩",
    "2379": "瑞昱",
    "2609": "陽明",
    # 熱門高股息與市值型 ETF
    "0050": "元大台灣50",
    "0056": "元大高股息",
    "00878": "國泰永續高股息",
    "00919": "群益台灣精選高息",
    "00929": "復華台灣科技優息",
    "00713": "元大台灣高息低波",
    "006208": "富邦台50"
}


def resolve_pure_id(query: str) -> str:
    query = query.strip()
    # 支援輸入中文名稱直接反查代號
    for code, name in STOCK_MAP.items():
        if query == name or query in name or query == code:
            return code

    # 提取純數字代號（支援 4 位數或以上 ETF 代號）
    pure_id = "".join(filter(str.isdigit, query))
    if len(pure_id) >= 4:
        return pure_id

    return query.replace(".TW", "").replace(".TWO", "")


def fetch_finmind_stock_data(query: str):
    pure_id = resolve_pure_id(query)
    if not pure_id or len(pure_id) < 4:
        raise ValueError(f"無法識別的股票或 ETF 代號/名稱：{query}")

    # 透過 FinMind API v4 取得日成交資訊（股票與 ETF 通用）
    url = "https://api.finmindtrade.com/api/v4/data"
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    parameters = {
        "dataset": "TaiwanStockPrice",
        "data_id": pure_id,
        "start_date": start_date
    }

    response = requests.get(url, params=parameters, timeout=10)
    result = response.json()

    if "data" not in result or len(result["data"]) == 0:
        raise ValueError(f"透過 FinMind 找不到代號 {pure_id} 的資料，請確認代號是否正確。")

    df = pd.DataFrame(result["data"])

    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "Trading_Volume": "Volume"
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close"])
    return df, pure_id


def predict_stock(query: str):
    if not query:
        return "⚠️ 請輸入股票/ETF 代號或名稱（例如：0050 或 台積電）"
    try:
        df, pure_id = fetch_finmind_stock_data(query)

        # 特徵工程
        data = df.copy()
        data["return_1d"] = data["Close"].pct_change(1, fill_method=None)
        data["return_5d"] = data["Close"].pct_change(5, fill_method=None)
        data["intraday_range"] = (data["High"] - data["Low"]) / data["Open"]
        data["upper_shadow"] = (data["High"] - data[["Open", "Close"]].max(axis=1)) / data["Close"]
        data["lower_shadow"] = (data[["Open", "Close"]].min(axis=1) - data["Low"]) / data["Close"]

        for window in [5, 10, 20, 60]:
            ma = data["Close"].rolling(window).mean()
            data[f"bias_{window}"] = (data["Close"] - ma) / ma

        delta = data["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        data["rsi_14"] = 100 - (100 / (1 + rs))

        data["vol_ratio_5"] = data["Volume"] / (data["Volume"].rolling(5).mean() + 1e-9)
        data["vol_ratio_20"] = data["Volume"] / (data["Volume"].rolling(20).mean() + 1e-9)

        feature_cols = [
            "return_1d", "return_5d", "intraday_range", "upper_shadow", "lower_shadow",
            "bias_5", "bias_10", "bias_20", "bias_60", "rsi_14", "vol_ratio_5", "vol_ratio_20"
        ]

        data["target_next_close"] = data["Close"].shift(-1)
        data["target_next_return"] = data["target_next_close"] / data["Close"] - 1
        data["target_up"] = (data["target_next_return"] > 0).astype(int)

        valid_feat_df = data.dropna(subset=feature_cols)
        latest_row = valid_feat_df.iloc[[-1]][feature_cols]
        latest_date = valid_feat_df.index[-1].strftime("%Y-%m-%d")
        latest_close = valid_feat_df["Close"].iloc[-1]

        dataset = data.dropna().copy()
        X = dataset[feature_cols]
        y_cls = dataset["target_up"]
        y_reg = dataset["target_next_return"]

        split_idx = int(len(dataset) * 0.8)
        X_train = X.iloc[:split_idx]
        y_cls_train = y_cls.iloc[:split_idx]
        y_reg_train = y_reg.iloc[:split_idx]

        # 訓練 LightGBM 模型
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, max_depth=4, verbose=-1)
        clf.fit(X_train, y_cls_train)

        reg = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.03, max_depth=4, verbose=-1)
        reg.fit(X_train, y_reg_train)

        up_prob = clf.predict_proba(latest_row)[0][1]
        next_return_pred = reg.predict(latest_row)[0]
        next_price_pred = latest_close * (1 + next_return_pred)

        target_name = STOCK_MAP.get(pure_id, f"台股標的 {pure_id}")

        result_md = f"""
        ### 📈 【{target_name} ({pure_id})】AI 分析報告
        * **最新收盤日期**：`{latest_date}`
        * **最新收盤價**：`{latest_close:.2f} 元`
        ---
        ### 🔮 下一個交易日 AI 預測
        * **預估上漲機率**：`{up_prob:.1%}`
        * **預估收盤價**：`{next_price_pred:.2f} 元` (`{next_return_pred:+.2%}`)
        ---
        *⚠️ 免責聲明：本系統由 AI 模型依據歷史量價自動運算，僅供技術展示與學術研究參考，不構成任何投資建議。*
        """
        return result_md
    except Exception as e:
        return f"❌ 發生錯誤：{str(e)}"


# 建立券商 App 風格介面
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📱 台股與 ETF AI 預測中心")
    gr.Markdown("支援台股個股與 **0050、0056、00878、00919** 等熱門 ETF！點擊下方快捷按鈕或直接搜尋。")

    with gr.Row():
        search_input = gr.Textbox(
            label="🔍 搜尋股票/ETF代號或名稱",
            placeholder="例如: 0050, 00878, 台積電, 2330",
            scale=3
        )
        search_btn = gr.Button("查詢預測", variant="primary", scale=1)

    gr.Markdown("### ⭐ 熱門權值股快捷面板")
    with gr.Row():
        btn_tsmc = gr.Button("🔥 台積電 (2330)")
        btn_honhai = gr.Button("📦 鴻海 (2317)")
        btn_mediatek = gr.Button("💻 聯發科 (2454)")
        btn_quanta = gr.Button("🤖 廣達 (2382)")

    gr.Markdown("### 📊 熱門 ETF 快捷面板")
    with gr.Row():
        btn_0050 = gr.Button("藍 0050 元大台灣50")
        btn_0056 = gr.Button("金 0056 元大高股息")
        btn_00878 = gr.Button("綠 00878 國泰永續")
        btn_00919 = gr.Button("橘 00919 群益精選")

    output_md = gr.Markdown(label="預測結果", value="👉 請點選上方快捷按鈕或輸入代號查詢")

    search_btn.click(fn=predict_stock, inputs=search_input, outputs=output_md)
    search_input.submit(fn=predict_stock, inputs=search_input, outputs=output_md)

    btn_tsmc.click(fn=lambda: predict_stock("2330"), inputs=[], outputs=output_md)
    btn_honhai.click(fn=lambda: predict_stock("2317"), inputs=[], outputs=output_md)
    btn_mediatek.click(fn=lambda: predict_stock("2454"), inputs=[], outputs=output_md)
    btn_quanta.click(fn=lambda: predict_stock("2382"), inputs=[], outputs=output_md)

    btn_0050.click(fn=lambda: predict_stock("0050"), inputs=[], outputs=output_md)
    btn_0056.click(fn=lambda: predict_stock("0056"), inputs=[], outputs=output_md)
    btn_00878.click(fn=lambda: predict_stock("00878"), inputs=[], outputs=output_md)
    btn_00919.click(fn=lambda: predict_stock("00919"), inputs=[], outputs=output_md)

if __name__ == "__main__":
    # 自動抓取雲端平台分配的 Port，並綁定 0.0.0.0 讓外部網路可以連線
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)