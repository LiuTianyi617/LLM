import requests
import streamlit as st
import pandas as pd
import os
import json
import time 

# ----------------- 設定與金鑰 -----------------
CWA_API_KEY = os.environ.get("CWA_API_KEY") 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") 

DATASTORE_ID = "F-C0032-001" 
LLM_MODEL = "gemini-2.5-flash-preview-09-2025"
LLM_API_URL_BASE = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"

LOCATIONS = ["臺北市", "臺中市", "高雄市", "新北市", "桃園市", "臺南市", "基隆市", "新竹縣", "苗栗縣", "彰化縣", "南投縣", "雲林縣", "嘉義縣", "屏東縣", "宜蘭縣", "花蓮縣", "臺東縣", "澎湖縣", "金門縣", "連江縣"]

SYSTEM_INSTRUCTION = {
    "parts": [{
        "text": "你是一位親切、溫和、且體貼的天氣顧問。請根據提供的數據，用傳統中文撰寫一個簡短、禮貌的問候語，總結未來的天氣狀況，並給予一到兩條實用的穿著或活動建議。請保持語氣友善和關心，不要使用標題或項目符號。"
    }]
}

def call_gemini_api(prompt):
    """呼叫 Gemini API 進行數據處理和文字生成"""
    if not GEMINI_API_KEY:
        return "Gemini API 金鑰未設定。無法生成 LLM 結果。"
    
    # 這裡不進行快取，因為 LLM 呼叫是整個服務的核心價值和實作要求
    url = f"{LLM_API_URL_BASE}?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": SYSTEM_INSTRUCTION
    }

    max_retries = 3
    delay = 2
    for attempt in range(max_retries):
        try:
            res = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
            res.raise_for_status()
            
            data = res.json()
            
            text = data.get('candidates')[0]['content']['parts'][0]['text']
            return text

        except requests.exceptions.RequestException as e:
            st.error(f"LLM API 連線錯誤 (嘗試 {attempt + 1}/{max_retries})。")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            return "❌ LLM 服務連線失敗或超時。"
        except Exception as e:
            return "❌ LLM 響應處理失敗。"
    return "❌ LLM 服務錯誤。"

@st.cache_data(ttl=3600) # <--- 核心優化：將數據快取 1 小時 (3600秒)
def extract_cwa_data_for_prompt(location):
    """從 CWA 數據中提取關鍵資訊，用於生成 LLM 的 Prompt"""
    if not CWA_API_KEY:
        st.error("❌ CWA API 金鑰未設定。請在 Secrets 中設定 CWA_API_KEY。")
        return None, None
    
    url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/{DATASTORE_ID}?Authorization={CWA_API_KEY}&locationName={location}"

    try:
        # 解決 PaaS 環境中的 SSL 憑證錯誤：verify=False
        res = requests.get(url, verify=False)
        res.raise_for_status() # 檢查 HTTP 狀態碼
        data = res.json()

        if data.get("success") == "true":
            location_data = data.get("records", {}).get("location", [{}])[0]
            elements = location_data.get("weatherElement", [])

            # 提取 LLM 關鍵資訊
            key_elements = {}
            for e in elements:
                name = e["elementName"]
                if name in ["MinT", "MaxT", "PoP", "CI"]: 
                    value = e["time"][0]["parameter"]["parameterName"]
                    key_elements[name] = value

            prompt_text = (
                f"以下是 {location} 未來的 36 小時天氣預報關鍵數據 (取第一時段): "
                f"最低溫度 (MinT): {key_elements.get('MinT', '無')} 度, "
                f"最高溫度 (MaxT): {key_elements.get('MaxT', '無')} 度, "
                f"降雨機率 (PoP): {key_elements.get('PoP', '無')} %, "
                f"舒適度 (CI): {key_elements.get('CI', '無')}。"
            )
            
            # 提取繪圖數據
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

    except requests.exceptions.RequestException as e:
        st.error(f"CWA 連線錯誤 (可能是網路或 SSL 問題)。")
        return None, None
    except Exception as e:
        st.error(f"CWA 數據處理錯誤。")
        return None, None


def main():
    st.set_page_config(layout="wide")
    st.title("☁️ 多雲整合服務：LLM 天氣顧問")
    st.caption("結合 CWA API 數據和 Gemini LLM 處理 (數據快取優化)")
    st.markdown("---")

    selected_location = st.selectbox("選擇城市", LOCATIONS)
    
    if not (CWA_API_KEY and GEMINI_API_KEY):
        st.error("請檢查 Streamlit Secrets：您必須設定 CWA_API_KEY 和 GEMINI_API_KEY。")
        return

    # 1. 從 CWA 雲端 API 獲取資料 (使用快取)
    prompt_source, df_chart = extract_cwa_data_for_prompt(selected_location)
    
    if not prompt_source:
        return

    # 2. 將資料丟給 LLM 處理
    st.subheader("🤖 LLM 天氣顧問的溫和問候與建議")
    
    with st.spinner('正在呼叫 Gemini LLM 進行語氣處理...'):
        llm_response = call_gemini_api(prompt_source)

    # 3. 使用介面將結果回傳 (優化項目 B: 使用 st.info)
    st.info(llm_response)

    st.markdown("---")

    # 顯示原始數據圖表
    if df_chart is not None and not df_chart.empty:
        st.subheader(f"📊 {selected_location} 36小時溫度趨勢 (原始數據)")
        st.line_chart(df_chart)


if __name__ == "__main__":
    main()
