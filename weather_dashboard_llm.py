import requests
import streamlit as st
import pandas as pd
import os
import json
import time # 用於 API 重試/等待

# ----------------- 設定與金鑰 -----------------
# 必須在 Streamlit Secrets 中設定這兩個金鑰
CWA_API_KEY = os.environ.get("CWA_API_KEY") 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 

DATASTORE_ID = "F-C0032-001" 
LLM_MODEL = "gemini-2.5-flash-preview-09-2025"
LLM_API_URL_BASE = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"

LOCATIONS = ["臺北市", "臺中市", "高雄市", "新北市", "桃園市", "臺南市", "基隆市", "新竹縣", "苗栗縣", "彰化縣", "南投縣", "雲林縣", "嘉義縣", "屏東縣", "宜蘭縣", "花蓮縣", "臺東縣", "澎湖縣", "金門縣", "連江縣"]

# ----------------- LLM 系統指令 (設定角色) -----------------
# 這是作業要求中「溫和問候的語氣」的設定
SYSTEM_INSTRUCTION = {
    "parts": [{
        "text": "你是一位親切、溫和、且體貼的天氣顧問。請根據提供的數據，用傳統中文撰寫一個簡短、禮貌的問候語，總結未來的天氣狀況，並給予一到兩條實用的穿著或活動建議。請保持語氣友善和關心，不要使用標題或項目符號。"
    }]
}

def call_gemini_api(prompt):
    """呼叫 Gemini API 進行數據處理和文字生成"""
    if not GEMINI_API_KEY:
        return "Gemini API 金鑰未設定。無法生成 LLM 結果。"

    url = f"{LLM_API_URL_BASE}?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": SYSTEM_INSTRUCTION
    }

    # 實作指數退避 (Exponential Backoff) 處理 API 節流
    max_retries = 3
    delay = 2
    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
            res.raise_for_status() # 檢查 HTTP 錯誤
            
            data = res.json()
            
            # 提取 LLM 生成的文字
            text = data.get('candidates')[0]['content']['parts'][0]['text']
            return text

        except requests.exceptions.RequestException as e:
            st.error(f"LLM API 連線錯誤 (嘗試 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            return "❌ LLM 服務連線失敗或超時。"
        except Exception as e:
            st.error(f"LLM 響應處理錯誤: {e}")
            return "❌ LLM 響應處理失敗。"
    return "❌ LLM 服務錯誤。"

def extract_cwa_data_for_prompt(location):
    """從 CWA 數據中提取關鍵資訊，用於生成 LLM 的 Prompt"""
    if not CWA_API_KEY:
        st.error("❌ CWA API 金鑰未設定。請在 Secrets 中設定 CWA_API_KEY。")
        return None, None
    
    url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/{DATASTORE_ID}?Authorization={CWA_API_KEY}&locationName={location}"

    try:
        # 解決 PaaS 環境中的 SSL 憑證錯誤：verify=False
        res = requests.get(url, verify=False)
        data = res.json()

        if res.status_code == 200 and data.get("success") == "true":
            location_data = data.get("records", {}).get("location", [{}])[0]
            elements = location_data.get("weatherElement", [])

            # 提取關鍵要素
            key_elements = {}
            for e in elements:
                name = e["elementName"]
                if name in ["MinT", "MaxT", "PoP", "CI"]: # MinT(最低溫), MaxT(最高溫), PoP(降雨機率), CI(舒適度)
                    # 抓取第一筆 (即未來 12 小時或第一段預報)
                    value = e["time"][0]["parameter"]["parameterName"]
                    key_elements[name] = value

            # 整理成 LLM Prompt 文字
            prompt_text = (
                f"以下是 {location} 未來的 36 小時天氣預報關鍵數據 (取第一時段): "
                f"最低溫度 (MinT): {key_elements.get('MinT', '無')} 度, "
                f"最高溫度 (MaxT): {key_elements.get('MaxT', '無')} 度, "
                f"降雨機率 (PoP): {key_elements.get('PoP', '無')} %, "
                f"舒適度 (CI): {key_elements.get('CI', '無')}。"
            )
            
            # 額外提取繪圖所需的 MaxT/MinT 數據 (用於視覺化展示)
            chart_data = []
            min_t_times = next((e["time"] for e in elements if e["elementName"] == "MinT"), [])
            max_t_times = next((e["time"] for e in elements if e["elementName"] == "MaxT"), [])
            
            for min_t, max_t in zip(min_t_times, max_t_times):
                time_point = pd.to_datetime(min_t["startTime"]).strftime("%H:%M")
                chart_data.append({
                    "時間": time_point,
                    "最低溫 (MinT)": int(min_t["parameter"]["parameterName"]),
                    "最高溫 (MaxT)": int(max_t["parameter"]["parameterName"])
                })
            
            df_chart = pd.DataFrame(chart_data).set_index("時間") if chart_data else None
            
            return prompt_text, df_chart

        else:
            st.error(f"CWA API 請求失敗: {data.get('message') or '未知錯誤'}")
            return None, None

    except Exception as e:
        st.error(f"CWA 數據處理錯誤: {e}")
        return None, None


def main():
    st.set_page_config(layout="wide")
    st.title("☁️ 多雲整合服務：LLM 天氣顧問")
    st.caption("結合 CWA API 數據和 Gemini LLM 處理")
    st.markdown("---")

    selected_location = st.selectbox("選擇城市", LOCATIONS)
    
    # 檢查所有必要的金鑰
    if not (CWA_API_KEY and GEMINI_API_KEY):
        st.error("請檢查 Streamlit Secrets：您必須設定 CWA_API_KEY 和 GEMINI_API_KEY。")
        return

    # 1. 從 CWA 雲端 API 獲取資料
    prompt_source, df_chart = extract_cwa_data_for_prompt(selected_location)
    
    if not prompt_source:
        return

    st.info(f"✅ 已從 CWA 取得 {selected_location} 數據。")
    
    # 2. 將資料丟給 LLM 處理 (作業步驟 2)
    st.subheader("🤖 LLM 天氣顧問的溫和問候與建議")
    
    # 使用 Spinner 顯示處理中的狀態
    with st.spinner('正在呼叫 Gemini LLM 進行語氣處理...'):
        llm_response = call_gemini_api(prompt_source)

    # 3. 使用介面將結果回傳 (作業步驟 3)
    # 使用 Markdown 區塊來美化 LLM 的輸出
    st.markdown(
        f"""
        <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 5px solid #4682b4;">
            <p style="font-size: 1.1em; margin: 0; line-height: 1.6;">{llm_response}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # 顯示原始數據圖表 (次要展示)
    if df_chart is not None and not df_chart.empty:
        st.subheader(f"📊 {selected_location} 36小時溫度趨勢 (原始數據)")
        st.line_chart(df_chart)


if __name__ == "__main__":
    main()