import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.cm as cm
import io
import boto3  # 🌟 雲端升級：導入 AWS SDK 核心連線套件
import json
import base64  # 🌟 視覺升級：導入 base64 解碼模組處理圖片封包
import datetime
import time

# ==========================================
# 0. 網頁基礎環境設定
# ==========================================
st.set_page_config(page_title="國道智慧交通 AI 決策平台", layout="wide")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  
plt.rcParams['axes.unicode_minus'] = False

# ⚠️ 雲端設定區：請在此填入你的 IAM 使用者金鑰
AWS_ACCESS_KEY = "*************"
AWS_SECRET_KEY = "***********************"
# 🌟 跨區解耦架構：將 AI 算力與大數據倉庫分離
AWS_REGION_BEDROCK = "us-west-2"  # AI 大腦專屬機房 (鎖定活躍的 Claude 3.5 Sonnet)
AWS_REGION_DATA = "ap-east-2" # 🌟 你的 S3 所在機房 (請填入你 S3 的真實代號，若在亞洲通常為 ap-northeast-1 或 ap-east-1)

# ==========================================
# 🌟 AWS 雲端雙引擎 (Athena 檢索 + Bedrock AI)
# ==========================================
def run_athena_query(query_string):
    """將 SQL 語法發送至 AWS Athena 並從 S3 取回結果轉為 DataFrame"""
    # 🌟 資料庫引擎：強制連線到資料所在的亞洲機房
    athena_client = boto3.client('athena', region_name=AWS_REGION_DATA, 
                                 aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    s3_client = boto3.client('s3', region_name=AWS_REGION_DATA, 
                             aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    
    # 啟動查詢 (指定你的 S3 作為 Athena 運算結果的暫存區)
    response = athena_client.start_query_execution(
        QueryString=query_string,
        QueryExecutionContext={'Database': 'default'}, 
        ResultConfiguration={'OutputLocation': 's3://112021003-lanchiakai-bigdataclass-03/athena_query_output/'} 
    )
    query_id = response['QueryExecutionId']
    
    # 等待 Athena 運算完成 (Polling)
    state = 'RUNNING'
    while state in ['RUNNING', 'QUEUED']:
        response = athena_client.get_query_execution(QueryExecutionId=query_id)
        state = response['QueryExecution']['Status']['State']
        if state == 'FAILED':
            reason = response['QueryExecution']['Status']['StateChangeReason']
            st.error(f"❌ Athena SQL 執行失敗: {reason}")
            return None
        time.sleep(1)
        
    # 直接透過 Query ID 精準抓取 S3 最終物理結果檔
    try:
        actual_s3_path = response['QueryExecution']['ResultConfiguration']['OutputLocation']
        path_parts = actual_s3_path.replace("s3://", "").split("/")
        bucket_name = path_parts[0]
        key_name = "/".join(path_parts[1:]) 
        
        obj = s3_client.get_object(Bucket=bucket_name, Key=key_name)
        df_result = pd.read_csv(io.BytesIO(obj['Body'].read()))
        return df_result
    except Exception as e:
        st.error(f"❌ 無法從 S3 讀取查詢結果: {str(e)}")
        return None

def call_aws_bedrock(prompt_text):
    try:
        # 🌟 AI 引擎：強制連線到美西機房呼叫大模型
        bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION_BEDROCK
        )
        model_id = "us.anthropic.claude-sonnet-4-6" 
        body_payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4500,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt_text}]
        }
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body_payload)
        )
        response_body = json.loads(response.get("body").read())
        return response_body['content'][0]['text']
    except Exception as e:
        return f"❌ AWS 雲端連線失敗，原因：{str(e)}"

@st.cache_data(show_spinner=False)
def load_gantry_data(gantry_file):
    """前六個功能絕不驚動 Athena，直接發送 API 直讀 S3 快取好的極輕量 CSV 檔"""
    try:
        # 🌟 快取引擎：同樣連線到資料所在的亞洲機房
        s3_client = boto3.client('s3', region_name=AWS_REGION_DATA, 
                                 aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
        obj = s3_client.get_object(Bucket="112021003-lanchiakai-bigdataclass-03", Key=f"Traffic_Stats_Long_all/{gantry_file}.csv")
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        
        if df is not None and not df.empty:
            vt_mapping = {31: '31_小客車', 32: '32_小貨車', 41: '41_大客車', 42: '42_大貨車', 5: '5_聯結車'}
            df['VehicleName'] = df['VehicleType'].map(vt_mapping)
            df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
            return df
    except Exception as e:
        st.error(f"⚠️ 雲端 S3 直讀失敗，原因：{str(e)}")
    return pd.DataFrame()

# ==========================================
# 🌟 模組化封裝：AI 智慧決策助理頁尾 (功能五 & 六)
# ==========================================
def render_ai_assistant_footer(tab_key, local_context_summary):
    """將原本的頁尾打包成函式，只在需要的分頁內呼叫，並透過 local_context_summary 防止 AI 看到其他閘道"""
    st.markdown("---")
    st.header("🤖 AWS Bedrock AI 智慧決策助理")

    weekday_chinese = {0: "週一", 1: "週二", 2: "週三", 3: "週四", 4: "週五", 5: "週六", 6: "週日"}
    start_weekday = weekday_chinese[selected_date_range[0].weekday()]
    end_weekday = weekday_chinese[selected_date_range[1].weekday()]

    if tab_key == "macro":
        display_gantries = selected_gantries if selected_gantries else "未勾選路段"
    else:
        display_gantries = target_gantry if 'target_gantry' in globals() else "未選擇路段"

    current_filters = {
        "圖表型態": chart_type,
        "觀測門架/路段": display_gantries,
        "觀測車型": selected_vehicles if selected_vehicles else "未勾選車種",
        "指標型態": data_type,
        "節日篩選": selected_holiday,
        "時間區間微調": f"{selected_date_range[0].strftime('%m/%d')}({start_weekday}) 至 {selected_date_range[1].strftime('%m/%d')}({end_weekday})"
    }

    axis_info = ""
    data_scope = ""
    if "12週全量每日各車種車流量趨勢" in chart_type:
        axis_info = "X 軸為【長線連續日期 (Date)】、Y 軸為【日車流總量】。圖上利用彩色陰影標註了清明、勞動、母親節三個節日區塊。"
        data_scope = f"這是巨觀的時間序列長線趨勢圖。目前畫面上『只有』顯示這些勾選車種：{current_filters['觀測車型']}。其餘沒勾選的車種絕對不在圖上！"
    elif "24小時各車種車流趨勢圖" in chart_type:
        axis_info = f"X 軸為【24小時制 (0-23時)】、Y 軸為【平均每小時車流量】。當前數據僅涵蓋特定日期分流：『{selected_holiday}』。"
        data_scope = f"這是微觀的單日行為圖。展示各車種在一天當中的雙峰（上下班尖峰）消長規律。"
    elif "各時段車種組成比例圖" in chart_type:
        axis_info = f"X 軸為【24小時制 (0-23時)】、Y 軸為【100% 堆疊長條比例 (Stacked Bar)】。當前日期型態：『{selected_holiday}』。"
        data_scope = f"此圖展示的是『結構佔比』，不是絕對車輛數！用來觀察在哪些特定時段大貨車或聯結車的市佔率會壓倒小客車。"
    elif "交通時鐘雷達擴散圖" in chart_type:
        axis_info = f"此圖為【極座標雷達擴散圖 (Radar Chart)】。圓周 360 度代表【24小時時鐘】，半徑向外擴散代表流量高低。當前型態：『{selected_holiday}』。"
        data_scope = f"越往外擴散代表該時段流量越大。請從幾點鐘方向有發生『流量大擴散』來進行診斷。"
    elif "一週各時段車流熱力圖" in chart_type:
        axis_info = "X 軸為【24小時 (0-23時)】、Y 軸為【星期一到星期日 (Monday-Sunday)】。格子顏色越深（黃到紅）代表車流密度越高。"
        data_scope = f"這是週與時段的二維矩陣熱力圖。目前畫面上呈現的是清明連假四天（週五、週六、週日、週一）的純淨特徵，請專注分析這四天對應的格子顏色深淺。"
    elif "四大日期型態交通量特徵對比圖" in chart_type:
        axis_info = "X 軸為【24小時 (0-23時)】、Y 軸為【平均車流量】。圖上的不同線條代表不同的節日（常規日子 vs 清明 vs 勞動 vs 母親節）。"
        data_scope = f"這是跨節日對比圖。請嚴格比對清明連假與常規日子在同一小時內的流量差距，抓出交通行為的本質差異。"
    elif "一週各日各車種分組對照圖 (Grouped Bar)" in chart_type:
        if data_type == "Min-Max 標準化流量":
            axis_info = "X 軸為【星期一到星期日 (Mon-Sun)】、Y 軸為【Min-Max 標準化流量值 [0, 1]】。每個星期底下並排拆出了 5 個獨立長條，分別代表 5 種車種。"
            data_scope = f"這是高階的『分組無量綱標準化對照圖』！目前畫面上各車種的流量已經被縮放到 0 到 1 之間，徹底解耦了小客車的數量霸凌！請注意：各車種此時是在同一個基準面上公平對比。"
        else:
            axis_info = f"X 軸為【星期一到星期日 (Mon-Sun)】、Y 軸為【平均每小時原始車流量 (輛)】。每個星期底下並排拆出了 5 個獨立長條，分別代表 5 種車種。"
            data_scope = f"這是『分組原始數據長條圖』。因為沒有經過標準化，小客車的藍色長條依然會高得離譜（壓倒性巨量），而大型車種的長條會顯得非常矮小。請誠實指出這個視覺特徵！"
    elif "24小時車流穩定度與離群值分析" in chart_type:
        axis_info = "X 軸為【24小時 (0-23時)】、Y 軸為【車流量數據點的分佈】。圖面呈現的是箱型圖 (Box Plot) 包含中位數與離群點。"
        data_scope = f"這張圖是用來檢視交通混亂度。箱子愈高或離群小點點愈多，代表該時段的車流愈不穩定、極易發生突發性回堵。"

    metric_warning = f"【⚠️ 注意 Y 軸計量法】：當前數據指標型態為『{data_type}』。如果是標準化流量值，數值嚴格限制在 0 到 1 之間，絕對不可以吐出『幾萬輛車』這種與畫面不符的字眼！"

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📝 功能五：AI 自動看圖診斷")
        if st.button("🚀 發動 AWS AI 自動看圖 analysis", key=f"btn_analyze_{tab_key}"):
            with st.spinner("🤖 AI 專家正在接入國道大數據流，撰寫診斷報告中..."):
                system_prompt = f"""你現在是高公局裡一位講話犀利、但對大數據圖表與軟體工程嚴謹度極度挑剔的天才智慧交通專家。
你眼前正在審查一張由 Streamlit 前端渲染出來的國道 ETC 大數據分析圖。

【🔥 這是目前網頁畫面真實渲染的數據與篩選條件（鐵證如山，不准忽視）】：
- 正在生成的圖表種類：{current_filters['圖表型態']}
- 官方圖表軸線與幾何定義：{axis_info}
- 當前畫面觀測對象：觀測門架路段 = {current_filters['觀測門架/路段']} | 勾選車種 = {current_filters['觀測車型']}
- 數據時間軸區間：{current_filters['時間區間微調']} (特定連假分流：{current_filters['節日篩選']})
- 專家觀測與物理量限制：{data_scope}
- {metric_warning}
- 【🚨 時間維度最高限制令】：請看清楚『時間區間微調』標籤。目前分析的時間『嚴格限定』在 {current_filters['時間區間微調']}。你只能討論這段區間內的星期，絕對不准自行盲推、發明或腦補任何錯誤的星期對應關係！

【📊 核心直讀引擎回傳之底層統計摘要】：
{local_context_summary}

請嚴格遵守資工人的誠實原則：
1. 畫面上有幾條線、什麼軸、什麼車，你就只能針對上述【真實渲染數據與軸定義】進行診斷，絕對不准無中生有！
2. 如果使用者只選了「小客車」，你看到的圖就只有小客車，絕對不准腦補大貨車或聯結車的趨勢！
3. 【最嚴格限制】：你的診斷報告『絕對只能』涵蓋【當前畫面觀測對象】中列出的路段。如果在統計摘要中看到其他路段的數據，請當作沒看見，絕對不准在報告中提及未在畫面上渲染的閘道資訊！

請直接輸出以下三點精簡內容（繁體中文，拒絕客套，一針見血）：
1. 【一眼看穿的圖表真相】：
   (請誠實、精準地描述上述條件下的圖表真實特徵。)
2. 【AI 腦補的節日心理學】：
   (結合上述篩選條件與圖表種類，用幽默風趣的角度解釋數據背後的人性行車行為。)
3. 【想提早下班的誠懇建議】：
   (針對圖表顯示的規律，給出交通調度或看板推播（CMS）口號建議。)"""
                ai_report = call_aws_bedrock(system_prompt)
                st.info("📊 **AWS Bedrock 專家診斷報告**")
                st.markdown(ai_report)

    with col_b:
        st.subheader("💬 功能六：對著大數據直接發問")
        user_question = st.text_input("輸入你想詢問的交通問題：", placeholder="例如：為什麼清明連假的車流看起來比平常低？", key=f"q_input_{tab_key}")
        if st.button("🔍 提問給 AI 專家", key=f"btn_ask_{tab_key}"):
            if user_question:
                with st.spinner(f"💡 AI 專家正在針對問題『{user_question}』進行數據思維推演..."):
                    qa_prompt = f"""你現在是高公局裡說話但專業爆表的天才智慧交通專家。你眼前正看著一張 Streamlit 渲染出的國道大數據分析圖。

【當前畫面真實上下文】：
- 圖表種類：{current_filters['圖表型態']}
- 軸線與數據定義：{axis_info}
- 目前篩選路段：{current_filters['觀測門架/路段']}
- 目前勾選車種：{current_filters['觀測車型']}
- 數據指標型態：{current_filters['指標型態']}
- 時間區間範圍：{current_filters['時間區間微調']}
- 底層統計特徵：{local_context_summary}

請結合上述實際數據與軸線資訊，用幽默、接地氣且極度精簡的繁體中文，直接切入核心回答使用者的提問。
限制：如果使用者問的問題跟當前勾選的車種、圖表或路段無關，請用資工人的誠實原則犀利地提醒他，並根據現有數據給予最專業的解答。字數要少，廢話要刪，絕對不要包含任何客套填充字！

使用者提問：{user_question}"""
                    ai_answer = call_aws_bedrock(qa_prompt)
                    st.success("🤖 **AI 專家即時回覆**")
                    st.markdown(ai_answer)


# ==========================================
# UI 核心渲染區
# ==========================================
st.title("🚀 國道電子收費 (ETC) 智慧交通大數據 AI 決策平台")
st.markdown("---")

st.sidebar.header("🛠️ 數據交叉分析控制面板")

gantry_options = {
    "南下下閘道 (03F2100S)北部南下進來到霧峰": "GantryD_03F2100S",  
    "南下上閘道 (03F2129S)離開霧峰南下到南部": "GantryO_03F2129S",  
    "北上上閘道 (03F2100N)離開霧峰北上到北部": "GantryO_03F2100N",  
    "北上下閘道 (03F2125N)南部北上進來霧峰": "GantryD_03F2125N"   
}

st.sidebar.markdown("**1. 勾選觀測閘道門架 (可自由複選對照)**")
selected_gantries = []
for display_name, file_name in gantry_options.items():
    if st.sidebar.checkbox(display_name, value=True, key=f"gantry_{file_name}"):
        selected_gantries.append(display_name)

st.sidebar.markdown("---")

st.sidebar.markdown("**2. 選擇觀測車種 (降低圖表維度雜訊)**")
vehicle_list = ['31_小客車', '32_小貨車', '41_大客車', '42_大貨車', '5_聯結車']
selected_vehicles = []
for v_name in vehicle_list:
    if st.sidebar.checkbox(v_name, value=True, key=f"vehicle_{v_name}"):
        selected_vehicles.append(v_name)

st.sidebar.markdown("---")

chart_type = st.sidebar.selectbox("3. 選擇欲生成的分析圖表種類", [
    "12週全量每日各車種車流量趨勢 (Line Chart)",
    "24小時各車種車流趨勢圖 (Line Chart)",
    "各時段車種組成比例圖 (Stacked Bar)",
    "交通時鐘雷達擴散圖 (Radar Chart)",
    "一週各時段車流熱力圖 (Heatmap)",
    "四大日期型態交通量特徵對比圖 (Line Chart)",
    "一週各日各車種分組對照圖 (Grouped Bar)",
    "24小時車流穩定度與離群值分析 (Box Plot)"
])

is_percentage_chart = "組成比例圖" in chart_type or "100% 比例消長" in chart_type

if is_percentage_chart:
    data_type = "原始數據"
    st.sidebar.markdown("⚙️ **4. 分析數據指標型態**")
    st.sidebar.info("🔒 **指標已鎖定**：當前圖表為比例圖，已自動固定為原始數據。")
else:
    data_type = st.sidebar.radio("4. 選擇分析數據指標型態", ["原始數據", "Min-Max 標準化流量"])

if chart_type in ["24小時各車種車流趨勢圖 (Line Chart)", "各時段車種組成比例圖 (Stacked Bar)", "交通時鐘雷達擴散圖 (Radar Chart)"]:
    holiday_display_options = ["全量數據", "常規日子", "清明連假(4/3-4/6)", "勞動節連假(5/1-5/3)", "母親節週末(5/9-5/10)"]
    user_choice = st.sidebar.selectbox("5. 選擇特定日期型態分流", holiday_display_options)
    
    holiday_mapping = {
        "全量數據": "全量數據",
        "常規日子": "常規日子",
        "清明連假(4/3-4/6)": "清明連假",
        "勞動節連假(5/1-5/3)": "勞動節連假",
        "母親節週末(5/9-5/10)": "母親節週末"
    }
    selected_holiday = holiday_mapping[user_choice]
else:
    selected_holiday = "全量數據"
    st.sidebar.info("💡 當前圖表已自動解鎖【全量 84 天總體時空資料】")

st.sidebar.markdown("---")
st.sidebar.markdown("**📅 6. 觀測時間軸區間微調**")
min_date = pd.to_datetime("2026-03-01")
max_date = pd.to_datetime("2026-05-23")

selected_date_range = st.sidebar.slider(
    "調整欲觀測的時間範圍：",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="MM/DD"
)

if not selected_gantries:
    st.warning("⚠️ 請在左側側邊欄【至少勾選一個閘道門架】！")
elif not selected_vehicles:
    st.error("⚠️ 請在左側側邊欄【至少勾選一種車種】以釋放圖表雜訊！")
else:
    tab_macro, tab_micro, tab_cloud = st.tabs([
        "🌐 雙軸四宮格全景對照", 
        "🔍 單一閘道多維度診斷對比", 
        "☁️ AWS Athena 雲端檢索中樞"
    ])

    y_column = "Count" if data_type == "原始數據" else "Volume_Normalized"
    y_label = "平均每小時車流量 (輛)" if data_type == "原始數據" else "標準化流量值 [0, 1]"
    ai_context_list = []

    # ==========================================================
    # 📑 分頁一：雙軸四宮格全景對照 (Macro View)
    # ==========================================================
    with tab_macro:
        num_gantries = len(selected_gantries)
        if num_gantries == 1:
            fig, axes = plt.subplots(1, 1, figsize=(11, 4.8))
            axes_flat = [axes]
        else:
            rows = 1 if num_gantries == 2 else 2
            cols = 2
            fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
            axes_flat = axes.flatten()

        for idx, gantry_name in enumerate(selected_gantries):
            current_ax = axes_flat[idx]
            file_name = gantry_options[gantry_name]
            
            df_raw = load_gantry_data(file_name)
            
            if selected_holiday != "全量數據":
                df_filtered = df_raw[df_raw['HolidayType'] == selected_holiday].reset_index(drop=True)
            else:
                df_filtered = df_raw.copy()
                
            start_filter, end_filter = selected_date_range
            df_filtered = df_filtered[
                (df_filtered['Date'] >= pd.to_datetime(start_filter)) & 
                (df_filtered['Date'] <= pd.to_datetime(end_filter))
            ].reset_index(drop=True)
                
            df_filtered = df_filtered[df_filtered['VehicleName'].isin(selected_vehicles)].reset_index(drop=True)

            if df_filtered.empty:
                current_ax.text(0.5, 0.5, "當前條件下暫無特徵資料", ha='center', va='center')
                continue

            avg_traffic = round(df_filtered['Count'].mean(), 1) if 'Count' in df_filtered.columns else 0
            max_traffic_row = df_filtered.groupby('Hour')['Count'].mean().idxmax() if 'Count' in df_filtered.columns else 0
            ai_context_list.append(f"- 路段[{gantry_name}]在目前篩選時間內，平均每小時車流為 {avg_traffic} 輛，全天最高車流波峰發生在 {max_traffic_row} 點。")

            if chart_type == "12週全量每日各車種車流量趨勢 (Line Chart)":
                show_legend = True if idx == 0 else False
                if y_column == "Count":
                    daily_trend = df_filtered.groupby(['Date', 'VehicleName'])['Count'].sum().reset_index()
                    sns.lineplot(data=daily_trend, x='Date', y='Count', hue='VehicleName', ax=current_ax, legend=show_legend)
                    current_ax.set_ylabel("日總車流量 (輛)", fontsize=9)
                else:
                    daily_norm_trend = df_filtered.groupby(['Date', 'VehicleName'])['Volume_Normalized'].mean().reset_index()
                    sns.lineplot(data=daily_norm_trend, x='Date', y='Volume_Normalized', hue='VehicleName', ax=current_ax, legend=show_legend)
                    current_ax.set_ylabel("標準化日流量平均值", fontsize=9)
                
                special_periods = [
                    {"start": "2026-04-03", "end": "2026-04-06", "label": "清明連假", "color": "red"},
                    {"start": "2026-05-01", "end": "2026-05-03", "label": "勞動節", "color": "orange"},
                    {"start": "2026-05-09", "end": "2026-05-10", "label": "母親節週末", "color": "magenta"}
                ]
                for period in special_periods:
                    p_start = pd.to_datetime(period["start"])
                    p_end = pd.to_datetime(period["end"])
                    if df_filtered['Date'].min() <= p_start <= df_filtered['Date'].max():
                        current_ax.axvspan(p_start, p_end, color=period["color"], alpha=0.1, linestyle=':', linewidth=0.8)
                        current_ax.text(p_start + (p_end - p_start)/2, 0.88, period["label"], 
                                        color=period["color"], fontsize=7, fontweight='bold',
                                        ha='center', va='center', transform=current_ax.get_xaxis_transform(),
                                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

                short_title = gantry_name.split(" ")[0] + " " + gantry_name.split(" ")[1]
                current_ax.set_title(short_title, fontsize=10, fontweight='bold')
                current_ax.tick_params(axis='x', labelrotation=15, labelsize=7)
                current_ax.tick_params(axis='y', labelsize=8)
                current_ax.grid(True, linestyle=':', alpha=0.5)
                if show_legend: current_ax.legend(title="車種", loc='upper right', fontsize=7, title_fontsize=8)

            elif chart_type == "24小時各車種車流趨勢圖 (Line Chart)":
                trend_data = df_filtered.groupby(['Hour', 'VehicleName'])[y_column].mean().reset_index()
                show_legend = True if idx == 0 else False
                sns.lineplot(data=trend_data, x='Hour', y=y_column, hue='VehicleName', marker='o', ax=current_ax, legend=show_legend)
                short_title = gantry_name.split(" ")[0] + " " + gantry_name.split(" ")[1]
                current_ax.set_title(short_title, fontsize=10, fontweight='bold')
                current_ax.set_xticks(range(24))
                current_ax.grid(True, linestyle=':', alpha=0.5)
                if show_legend: current_ax.legend(title="車種", loc='upper right', fontsize=8)

            elif chart_type == "各時段車種組成比例圖 (Stacked Bar)":
                hourly_total = df_filtered.groupby(['Hour', 'VehicleName'])['Count'].sum().unstack().fillna(0)
                available_cols = [c for c in selected_vehicles if c in hourly_total.columns]
                if available_cols:
                    hourly_total = hourly_total[available_cols]
                    hourly_pct = hourly_total.div(hourly_total.sum(axis=1), axis=0) * 100
                    show_legend = True if idx == 0 else False
                    hourly_pct.plot(kind='bar', stacked=True, colormap='Set2', ax=current_ax, width=0.75, legend=show_legend)
                short_title = gantry_name.split(" ")[0] + " " + gantry_name.split(" ")[1]
                current_ax.set_title(short_title, fontsize=10, fontweight='bold')
                if show_legend: current_ax.legend(title="車種", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

            elif chart_type == "交通時鐘雷達擴散圖 (Radar Chart)":
                current_ax.remove()
                current_ax = fig.add_subplot(1 if num_gantries==1 else (1 if num_gantries==2 else 2), num_gantries if num_gantries<=2 else 2, idx+1, polar=True)
                current_ax.set_theta_offset(np.pi / 2)
                current_ax.set_theta_direction(-1)
                
                radar_data = df_filtered.groupby(['Hour', 'VehicleName'])[y_column].mean().unstack().fillna(0)
                available_cols = [c for c in selected_vehicles if c in radar_data.columns]
                
                labels = np.array([str(i) for i in range(24)])
                angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
                angles += angles[:1]
                colors = cm.get_cmap('Set1')(np.linspace(0, 1, len(vehicle_list)))
                
                for v_idx, vt in enumerate(available_cols):
                    values = radar_data[vt].tolist()
                    values += values[:1]
                    c_color = colors[vehicle_list.index(vt)]
                    current_ax.plot(angles, values, color=c_color, linewidth=1.2, label=vt if idx==0 else "")
                    current_ax.fill(angles, values, color=c_color, alpha=0.03)
                current_ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
                short_title = gantry_name.split(" ")[0] + " " + gantry_name.split(" ")[1]
                current_ax.set_title(short_title, fontsize=10, fontweight='bold', y=1.1)
                if idx == 0: current_ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=8)

            elif chart_type == "一週各時段車流熱力圖 (Heatmap)":
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                df_filtered['Weekday'] = pd.Categorical(df_filtered['Weekday'], categories=weekday_order, ordered=True)
                heatmap_data = df_filtered.groupby(['Weekday', 'Hour'])[y_column].mean().unstack()
                sns.heatmap(heatmap_data, cmap='YlOrRd', linewidths=.2, ax=current_ax, cbar=(idx==0))
                short_title = gantry_name.split(" ")[0] + " " + gantry_name.split(" ")[1]
                current_ax.set_title(short_title, fontsize=10, fontweight='bold')

            elif chart_type == "四大日期型態交通量特徵對比圖 (Line Chart)":
                holiday_data = df_filtered.groupby(['HolidayType', 'Hour'])[y_column].mean().reset_index()
                show_legend = True if idx == 0 else False
                sns.lineplot(data=holiday_data, x='Hour', y=y_column, hue='HolidayType', marker='s', ax=current_ax, legend=show_legend)
                short_title = gantry_name.split(" ")[0] + " " + gantry_name.split(" ")[1]
                current_ax.set_title(short_title, fontsize=10, fontweight='bold')
                current_ax.set_xticks(range(24))
                current_ax.grid(True, linestyle=':', alpha=0.5)
                if show_legend: current_ax.legend(fontsize=8)

            elif chart_type == "一週各日各車種分組對照圖 (Grouped Bar)":
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                df_filtered['Weekday'] = pd.Categorical(df_filtered['Weekday'], categories=weekday_order, ordered=True)
                grouped_data = df_filtered.groupby(['Weekday', 'VehicleName'], observed=False)[y_column].mean().reset_index()
                show_legend = True if idx == 0 else False
                sns.barplot(data=grouped_data, x='Weekday', y=y_column, hue='VehicleName', palette='Set2', ax=current_ax)
                
                short_title = gantry_name.split(" ")[0] + " " + gantry_name.split(" ")[1]
                current_ax.set_title(short_title, fontsize=10, fontweight='bold')
                current_ax.set_ylabel(y_label, fontsize=9)
                current_ax.tick_params(axis='x', labelsize=8)
                current_ax.grid(True, linestyle=':', alpha=0.5, axis='y')
                if not show_legend:
                    current_ax.get_legend().remove()
                else:
                    current_ax.legend(title="車種", loc='upper right', fontsize=7, title_fontsize=8)

            elif chart_type == "24小時車流穩定度與離群值分析 (Box Plot)":
                sns.boxplot(data=df_filtered, x='Hour', y='Count', palette='Set3', ax=current_ax)
                short_title = gantry_name.split(" ")[0] + " " + gantry_name.split(" ")[1]
                current_ax.set_title(short_title, fontsize=10, fontweight='bold')
                current_ax.grid(True, linestyle=':', alpha=0.5, axis='y')

        for j in range(num_gantries, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        plt.suptitle("國道多維時空交叉對照矩陣 — 指標: " + data_type, fontsize=13, fontweight='bold', y=0.985)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        st.pyplot(fig)
        
        buf_macro = io.BytesIO()
        fig.savefig(buf_macro, format="png", dpi=300)
        buf_macro.seek(0)
        st.download_button(
            label="📥 下載全景對照圖表 (高解析度 PNG)",
            data=buf_macro,
            file_name="macro_gantry_comparison.png",
            mime="image/png",
            key="btn_download_macro"
        )

        current_report_data_summary = f"使用者篩選條件：\n- 分析圖表：{chart_type}\n- 日期型態分流：{selected_holiday}\n- 分析數據指標：{data_type}\n\n實體統計特徵摘要：\n" + "\n".join(ai_context_list)
        
        # 🌟 呼叫智能頁尾（傳入四個閘道的綜合摘要）
        render_ai_assistant_footer(tab_key="macro", local_context_summary=current_report_data_summary)


    # ==========================================================
    # 📑 分頁二：🔍 單一閘道多維度診斷對比 (Micro Drill-down)
    # ==========================================================
    with tab_micro:
        st.subheader("🔍 單一門架多維度交叉診斷面板")
        
        target_gantry = st.selectbox("🎯 請選擇欲剖析的目標國道閘道：", selected_gantries, key="micro_gantry_select")
        target_file = gantry_options[target_gantry]
        df_target_raw = load_gantry_data(target_file)
        
        start_filter, end_filter = selected_date_range
        df_target_time_sliced = df_target_raw[
            (df_target_raw['Date'] >= pd.to_datetime(start_filter)) & 
            (df_target_raw['Date'] <= pd.to_datetime(end_filter))
        ].reset_index(drop=True)

        st.markdown("---")
        
        micro_mode = st.radio(
            "🛠️ 請選擇微觀深度 analysis 模式：",
            [
                "1. 5大車種獨立矩陣對照 (解耦車種雜訊)", 
                "2. 單一路段四大日期型態疊加對照 (🌟支援多車種複選勾選)", 
                "3. 單一路段多車種混和 24H 流量重疊對照 (🌟高公局核心決策大圖)"
            ],
            horizontal=True
        )

        st.markdown(f"📊 當前觀測路段：`{target_gantry}` | 數據指標：`{data_type}`")

        if "1. 5大車種獨立矩陣" in micro_mode:
            if selected_holiday != "全量數據":
                df_m1 = df_target_time_sliced[df_target_time_sliced['HolidayType'] == selected_holiday].reset_index(drop=True)
            else:
                df_m1 = df_target_time_sliced.copy()
                
            active_vehicles = [v for v in selected_vehicles if v in df_m1['VehicleName'].unique()]
            
            if not active_vehicles:
                st.info("💡 請在左側側邊欄至少勾選一種車種！")
            else:
                fig_m1, axes_m1 = plt.subplots(2, 3, figsize=(16, 7.5))
                axes_m1_flat = axes_m1.flatten()
                palette_colors = sns.color_palette("husl", 5)
                
                for v_idx, v_name in enumerate(vehicle_list):
                    ax = axes_m1_flat[v_idx]
                    if v_name not in active_vehicles:
                        ax.text(0.5, 0.5, "未在側邊欄勾選 " + v_name, ha='center', va='center', color='gray', fontsize=10)
                        ax.set_xticks([]); ax.set_yticks([])
                        continue
                        
                    df_v = df_m1[df_m1['VehicleName'] == v_name]
                    
                    if "12週全量每日" in chart_type:
                        plot_data = df_v.groupby('Date')[y_column].sum().reset_index() if y_column=="Count" else df_v.groupby('Date')[y_column].mean().reset_index()
                        sns.lineplot(data=plot_data, x='Date', y=y_column, color=palette_colors[v_idx], ax=ax, linewidth=1.5)
                        ax.tick_params(axis='x', labelrotation=30, labelsize=7)
                    else:
                        plot_data = df_v.groupby('Hour')[y_column].mean().reset_index()
                        sns.lineplot(data=plot_data, x='Hour', y=y_column, marker='o', color=palette_colors[v_idx], ax=ax, linewidth=1.8)
                        ax.set_xticks(range(0, 24, 4))
                    
                    ax.set_title(v_name + " - " + y_column, fontsize=10, fontweight='bold')
                    ax.grid(True, linestyle=':', alpha=0.6)
                
                fig_m1.delaxes(axes_m1_flat[5])
                plt.suptitle("各車種獨立流量響應特徵矩陣 (路段: " + target_gantry.split(' ')[0] + ")", fontsize=12, fontweight='bold', y=0.98)
                plt.tight_layout()
                st.pyplot(fig_m1)
                
                buf_m1 = io.BytesIO()
                fig_m1.savefig(buf_m1, format="png", dpi=300)
                buf_m1.seek(0)
                st.download_button(label="📥 下載車種獨立矩陣圖表 (PNG)", data=buf_m1, file_name="micro_vehicle_matrix.png", mime="image/png", key="btn_download_m1")

        elif "2. 單一路段四大日期型態" in micro_mode:
            st.markdown("💡 **操作指南**：請在下方勾選你想觀測的車種。系統會為每種車種獨立畫出一張「四大日期型態疊加圖」，方便比對。")
            
            st.markdown("**請選擇欲重疊觀測的車種 (可複選)：**")
            m2_cols = st.columns(5)
            m2_selected_vehicles = []
            for v_idx, v_name in enumerate(vehicle_list):
                is_default_checked = v_name in selected_vehicles
                if m2_cols[v_idx].checkbox(v_name, value=is_default_checked, key=f"m2_v_chk_{v_name}"):
                    m2_selected_vehicles.append(v_name)
            
            if not m2_selected_vehicles:
                st.warning("⚠️ 請至少勾選一種車種以渲染圖表！")
            else:
                num_m2 = len(m2_selected_vehicles)
                fig_m2, axes_m2 = plt.subplots(num_m2, 1, figsize=(14, 3.8 * num_m2), sharex=True)
                axes_m2_flat = [axes_m2] if num_m2 == 1 else axes_m2.flatten()
                
                for idx_m2, v_name in enumerate(m2_selected_vehicles):
                    ax = axes_m2_flat[idx_m2]
                    df_m2_v = df_target_time_sliced[df_target_time_sliced['VehicleName'] == v_name].reset_index(drop=True)
                    
                    if "12週全量每日" in chart_type:
                        sns.lineplot(data=df_m2_v, x='Date', y=y_column, hue='HolidayType', ax=ax, linewidth=1.5)
                        ax.set_title(v_name + " - 12週日期型態長線消長趨勢", fontsize=10, fontweight='bold')
                    else:
                        m2_trend = df_m2_v.groupby(['HolidayType', 'Hour'])[y_column].mean().reset_index()
                        sns.lineplot(data=m2_trend, x='Hour', y=y_column, hue='HolidayType', marker='s', ax=ax, linewidth=1.8)
                        ax.set_xticks(range(24))
                        ax.set_title(v_name + " - 四大日期型態 24H 雙峰響應對照", fontsize=10, fontweight='bold')
                    
                    ax.set_ylabel(y_column, fontsize=8)
                    ax.grid(True, linestyle=':', alpha=0.5)
                    ax.legend(title="日期型態", fontsize=7, title_fontsize=8, loc='upper right')
                
                plt.suptitle(target_gantry.split(' ')[0] + " - 跨節日全量行為矩陣", fontsize=12, fontweight='bold', y=0.99)
                plt.tight_layout()
                st.pyplot(fig_m2)
                
                buf_m2 = io.BytesIO()
                fig_m2.savefig(buf_m2, format="png", dpi=300)
                buf_m2.seek(0)
                st.download_button(label="📥 下載日期型態疊加圖表 (PNG)", data=buf_m2, file_name="micro_holiday_overlay.png", mime="image/png", key="btn_download_m2")

        elif "3. 單一路段多車種混和" in micro_mode:
            st.markdown("💡 **說明**：此模式已升級為**五大車種 24H 獨立子圖面板**。各車種的時間軸垂直對齊，Y 軸自動適應，完美解決小客車流量過大導致大型車曲線被壓扁的問題，極具交通決策價值。")
            
            if selected_holiday != "全量數據":
                df_m3 = df_target_time_sliced[df_target_time_sliced['HolidayType'] == selected_holiday].reset_index(drop=True)
                title_holiday_suffix = " (特定型態: " + user_choice + ")"
            else:
                df_m3 = df_target_time_sliced.copy()
                title_holiday_suffix = " (特定型態: 全量數據基準線)"
                
            active_vehicles_m3 = sorted([v for v in selected_vehicles if v in df_m3['VehicleName'].unique()])
            
            if not active_vehicles_m3:
                st.info("💡 請在左側側邊欄勾選欲觀測的車種。")
            else:
                trend_data_m3 = df_m3.groupby(['Hour', 'VehicleName'])[y_column].mean().reset_index()
                num_vehicles = len(active_vehicles_m3)
                
                fig_m3, axes_m3 = plt.subplots(num_vehicles, 1, figsize=(14, 3.2 * num_vehicles), sharex=True)
                axes_m3_flat = [axes_m3] if num_vehicles == 1 else axes_m3.flatten()
                
                color_map = {
                    "31_小客車": "#1f77b4", "32_小貨車": "#ff7f0e",
                    "41_大客車": "#2ca02c", "42_大貨車": "#d62728", "5_聯結車": "#9467bd"
                }
                
                for idx_m3, v_name in enumerate(active_vehicles_m3):
                    ax = axes_m3_flat[idx_m3]
                    v_data = trend_data_m3[trend_data_m3['VehicleName'] == v_name]
                    v_color = color_map.get(v_name, "#7f7f7f")
                    
                    sns.lineplot(data=v_data, x='Hour', y=y_column, marker='o', color=v_color, linewidth=1.8, ax=ax)
                    
                    ax.set_xticks(range(24))
                    ax.set_xlim(-0.5, 23.5)
                    ax.set_ylabel(y_label, fontsize=8)
                    ax.grid(True, linestyle=':', alpha=0.5)
                    
                    short_gantry = target_gantry.split(' ')[0]
                    ax.set_title(short_gantry + " - " + v_name + " 24H 流量趨勢" + title_holiday_suffix, fontsize=10, fontweight='bold', loc='left')
                    
                    if idx_m3 == num_vehicles - 1:
                        ax.set_xlabel("一天之中的時間 (小時制 0-23)", fontsize=9, fontweight='bold')
                
                plt.suptitle("單一門架微觀剖析矩陣 (各車種特徵完全解耦對照)", fontsize=12, fontweight='bold', y=0.99)
                plt.tight_layout()
                st.pyplot(fig_m3)
                
                buf_m3 = io.BytesIO()
                fig_m3.savefig(buf_m3, format="png", dpi=300)
                buf_m3.seek(0)
                st.download_button(label="📥 下載微觀剖析五車種對照圖表 (PNG)", data=buf_m3, file_name="micro_gantry_drilldown.png", mime="image/png", key="btn_download_m3")

        plt.close('all')

        # 🌟 即時計算單一路段專屬摘要，物理隔離其他路段污染
        avg_micro = round(df_target_time_sliced['Count'].mean(), 1) if 'Count' in df_target_time_sliced.columns else 0
        max_micro_hr = df_target_time_sliced.groupby('Hour')['Count'].mean().idxmax() if 'Count' in df_target_time_sliced.columns else 0
        micro_summary = f"使用者篩選條件：\n- 分析圖表：當前為單一閘道微觀分析模式\n- 觀測路段：{target_gantry}\n- 日期型態分流：{selected_holiday}\n\n實體統計特徵摘要：\n- 路段[{target_gantry}]在目前篩選時間內，平均每小時車流為 {avg_micro} 輛，全天最高車流波峰發生在 {max_micro_hr} 點。"

        # 🌟 呼叫智能頁尾（傳入單一閘道專屬摘要）
        render_ai_assistant_footer(tab_key="micro", local_context_summary=micro_summary)


   # ==========================================================
    # 📑 分頁三：☁️ 功能七 (Agentic AI 雲端直連動態生成)
    # ==========================================================
    with tab_cloud:
        st.subheader("☁️ 功能七：Agentic AI 雲端直連動態生成")
        st.markdown("💬 **你可以這樣問**：清明連假車流真的會比較多嗎？ / 大貨車真的都在半夜出沒嗎？")
        
        agent_question = st.text_input("輸入你要探索的大數據問題：", key="agent_input")
        
        if st.button("🚀 啟動 Athena 動態資料探勘"):
            if agent_question:
                with st.spinner("🧠 AI 正在解析語意並編譯分散式 SQL..."):
                    # 🌟 升級 1：賦予 AI 數據分析師的靈魂，強制規範其思考邏輯與【視覺化決策】
                    sql_prompt = f"""
                    你是一個資深的資料庫工程師與高階數據分析師，專精於 AWS Athena (Presto SQL)。
                    我的 Athena 資料表名稱為 `etc_cleaned_data`。

                    【📊 資料表 Schema 與型態定義 (🚨非常重要，請嚴格遵守資料型態)】：
                    - `sourcefile` (VARCHAR)：門架檔案名稱
                    - `timebasis` (VARCHAR)：時間基準
                    - `year` (INT), `month` (INT), `day` (INT), `hour` (INT)：時間維度
                    - `weekday` (VARCHAR)：星期幾 (數值必須是英文字串，例如：'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
                    - `holidaytype` (VARCHAR)：假日型態 (只有這四種絕對值：'常規日子', '清明連假', '勞動節連假', '母親節週末')
                    - `vehicletype` (INT)：車種代碼 (🚨這是整數，對應關係：31=小客車, 32=小貨車, 41=大客車, 42=大貨車, 5=聯結車)
                    - `count` (INT)：車流量
                    - `volume_normalized` (DOUBLE)：標準化車流量

                    🚨【重要門架知識庫】：若使用者詢問特定方向，請務必使用 `sourcefile` 進行 WHERE 過濾 (必須用 VARCHAR 單引號，可用 LIKE '%代號%')：
                    - 離開霧峰北上 (往北部)：對應代號 `03F2100N`
                    - 南部北上進霧峰 (回霧峰)：對應代號 `03F2125N`
                    - 離開霧峰南下 (往南部)：對應代號 `03F2129S`
                    - 北部南下進霧峰 (回霧峰)：對應代號 `03F2100S`

                    🚨【📅 節日時間字典 (若遇到模糊日期請直接查表替換)】：
                    - 清明連假：4/3 (週五) 至 4/6 (週一)。(清明節第一天是 4/3，最後一天是 4/6)
                    - 勞動節連假：5/1 (週五) 至 5/3 (週日)。(勞動節第一天是 5/1，最後一天是 5/3)
                    - 母親節週末：5/9 (週六) 至 5/10 (週日)。(母親節第一天是 5/9，最後一天是 5/10)

                    【🎯 Agent 核心思考指令】：
                    1. 【型態防呆鎖死】：所有 VARCHAR 欄位的條件判斷，絕對要用單引號包覆字串 (例如：`weekday = 'Sunday'`，絕對不可以寫 `weekday = 7`，否則系統會崩潰)！
                    2. 若詢問「時間趨勢/何時最塞」，請 SELECT `hour` 與聚合後的 `AVG(count) AS count`，並 `GROUP BY hour ORDER BY hour`。
                    3. 若詢問「車種比例/誰最多」，請 SELECT `vehicletype` 與 `AVG(count) AS count`，並 GROUP BY `vehicletype`。
                    4. 【語意量化】：遇到「最後一天」、「第一天」等模糊時間，請直接對照【節日時間字典】，將其轉換為精確的 `month` 與 `day` 條件。
                       - 範例：使用者問「清明連假最後一天」，你查表得知是 4/6，則 SQL 條件請直接寫 `WHERE holidaytype = '清明連假' AND month = 4 AND day = 6`。絕對不要寫死星期幾！
                    5. 若需比較特定情境，請務必把對照組也撈出來，並 SELECT `holidaytype` 以供比較。
                    6. 【車種代碼轉換】：若使用者詢問特定車種 (例如大貨車)，請務必將其轉換為整數代碼查詢 (例如寫死：`vehicletype = 42`)，絕對不可以使用 LIKE 或字串進行比對！
                    7. 【超出資料範圍防呆】：這份 ETC 資料庫的收集範圍僅限 3/1 至 5/23。如果使用者詢問不在範圍內的任何日期（例如：5/24、新年、中秋節），請絕對不要自己發明日期或標籤！請一律回傳這個必定為空的防呆 SQL：SELECT 0 AS hour, 0 AS count FROM etc_cleaned_data WHERE 1=0

                    【🎨 視覺化決策指令】：
                    請根據你寫出的 SQL 邏輯，決定最適合的圖表：
                    - 若 X 軸是時間 (`hour`) -> 選擇 'line' (折線圖)
                    - 若 X 軸是車種 (`vehicletype`) 或純類別比較 -> 選擇 'bar' (長條圖)

                    🚨 【嚴格輸出格式】：你只能回傳以下兩行純文字，絕對不准包含 Markdown 符號 (` ``` `) 或任何解釋！
                    ChartType: [line 或 bar]
                    SQL: [你的 SQL 語句]

                    使用者問題：{agent_question}
                    """
                    
                    # 🌟 解析 AI 的雙重指令
                    raw_response = call_aws_bedrock(sql_prompt).strip()
                    
                    ai_chart_type = "line"  # 預設值防呆
                    generated_sql = ""
                    
                    for line in raw_response.split('\n'):
                        if line.upper().startswith("CHARTTYPE:"):
                            ai_chart_type = line.split(":")[1].strip().lower()
                        elif line.upper().startswith("SQL:"):
                            generated_sql = line.split(":", 1)[1].strip()
                    
                    # 容錯處理：如果 AI 還是吐出 Markdown 標籤，就清掉
                    if generated_sql.startswith("```sql"):
                        generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
                    elif generated_sql.endswith("```"):
                        generated_sql = generated_sql.replace("```", "").strip()
                    
                    st.markdown(f"**🤖 AI 動態編譯之洞察 SQL (自動決策圖表: `{ai_chart_type}`)：**")
                    st.code(generated_sql, language="sql")
                    
                with st.spinner("⚡ 正在呼叫 AWS Athena 進行分散式運算..."):
                    start_time = time.time()
                    df_result = run_athena_query(generated_sql)
                    end_time = time.time()
                    
                    if df_result is not None and not df_result.empty:
                        st.success(f"✅ Athena 檢索完成！耗時：{end_time - start_time:.2f} 秒，共撈取 {len(df_result)} 筆特徵數據。")
                        
                        # 🌟 步驟 1：強制欄位全部降轉小寫
                        df_result.columns = df_result.columns.str.lower()
                        
                        # 🌟 步驟 2：防呆機制！不管 AI 的 SQL 欄位聚合後叫 count 還是 avg_count，通通更名為 count
                        if 'avg_count' in df_result.columns:
                            df_result = df_result.rename(columns={'avg_count': 'count'})
                        elif 'sum_count' in df_result.columns:
                            df_result = df_result.rename(columns={'sum_count': 'count'})
                        
                        st.write("📊 **檢索結果數據預覽：**")
                        st.dataframe(df_result.head())
                        
                        # 🌟 步驟 3 & 4：全新寬表格變形引擎與 AI 洞察
                        # 動態判定 X 軸是時間還是車種
                        x_axis_col = 'hour' if 'hour' in df_result.columns else ('vehicletype' if 'vehicletype' in df_result.columns else None)
                        
                        if x_axis_col and 'count' in df_result.columns:
                            st.write(f"📈 **Agent 動態生成之多維度對比圖 ({ai_chart_type})：**")
                            try:
                                # A. 如果有節日型態特徵，把節日拉成不同的獨立圖例
                                if 'holidaytype' in df_result.columns:
                                    chart_data = df_result.pivot_table(index=x_axis_col, columns='holidaytype', values='count', aggfunc='mean')
                                # B. 如果是以時間為 X 軸，但有車種特徵，把車種拉成不同的獨立圖例
                                elif x_axis_col == 'hour' and 'vehicletype' in df_result.columns:
                                    chart_data = df_result.pivot_table(index='hour', columns='vehicletype', values='count', aggfunc='mean')
                                # C. 單一長線/長條趨勢
                                else:
                                    chart_data = df_result.groupby(x_axis_col)['count'].mean()
                                
                                # 🌟 修復時間軸 X 軸詭異的留白：強制把數字轉為文字標籤！
                                # 🌟 修復時間軸 X 軸詭異的留白與「字典排序陷阱」：自動補零 (Zero-Padding)！
                                if x_axis_col == 'hour':
                                    # zfill(2) 會把 '1' 變成 '01'，'2' 變成 '02'，保證字串排序 = 數字排序
                                    chart_data.index = chart_data.index.astype(str).str.zfill(2) + "時"
                                
                                # 🚀 聽從 AI 大腦的指示，發動對應的高階直繪引擎
                                if ai_chart_type == "bar":
                                    st.bar_chart(chart_data)
                                else:
                                    st.line_chart(chart_data)
                                
                                # 🌟 步驟 4：AI 決策洞察 (放在 try 裡面，確保 chart_data 存在才執行！)
                                explain_prompt = f"""
                                根據這份包含完整趨勢對比的數據 {chart_data.round(1).to_dict()}，
                                請用繁體中文、幽默風趣的語氣，直接回答使用者的原始問題：『{agent_question}』。
                                🚨 注意：請務必綜觀全域數據的變化（例如白天誰比較高、半夜誰比較高，有沒有發生反轉），給出最精確的結論！
                                """
                                st.info("💡 **AI 決策洞察：**")
                                st.markdown(call_aws_bedrock(explain_prompt))

                            except Exception as chart_err:
                                st.warning(f"⚠️ 數據結構變形失敗（{chart_err}），已退回純文字摘要分析。")
                                explain_prompt = f"根據數據 {df_result.to_dict()}，請用繁體中文回答：『{agent_question}』。"
                                st.info("💡 **AI 決策洞察：**")
                                st.markdown(call_aws_bedrock(explain_prompt))
                        else:
                            # 🌟 防呆備用路線：如果 AI 寫的 SQL 既沒有 hour 也沒有 vehicletype (例如只查總數)
                            explain_prompt = f"根據這份數據 {df_result.to_dict()}，請用繁體中文回答使用者的問題：『{agent_question}』。"
                            st.info("💡 **AI 決策洞察：**")
                            st.markdown(call_aws_bedrock(explain_prompt))
                    else:
                        st.error("❌ 查詢無結果。可能是條件內無資料或 SQL 語意無法解析。")