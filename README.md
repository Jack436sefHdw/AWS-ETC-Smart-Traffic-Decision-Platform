# 🚀 國道 ETC 智慧交通大數據 AI 決策平台
> 處理 2.74 億筆真實交通數據 | AWS 雲端管線建置 | Agentic AI (Text-to-SQL) 實作

⚠️ **架構狀態聲明 (Infrastructure Status):** 
本專案之 AWS 雲端實體資源（EC2, Athena, Glue 等）為撙節長期營運成本，目前已將服務安全下線 (Tear-down)。本頁面提供系統完整運作之實機 Demo 影片、雲端架構圖與大數據分析報告。

## 📖 專案簡介 (About The Project)
傳統的交通流量分析高度仰賴專業工程師撰寫 SQL，且靜態圖表無法提供非技術人員即時的決策輔助。
本專案利用 AWS 雲端平台處理交通部高公局 12 週、逾 2.74 億筆（58.9 GB）的國道 ETC 交易大數據。透過建立無伺服器 (Serverless) 資料管線與前端快取機制，打造具備高互動性的交通決策儀表板；並創新導入 Agentic AI 助理，讓使用者能以自然語言直接向大數據資料庫進行提問與視覺化圖表生成。

## 🎥 實機運作 Demo (Live Demonstration)
* **[點此觀看系統完整實機 Demo 影片 (YouTube)](#)** <!-- 替換為你的 YouTube 連結 -->

## ✨ 核心技術與亮點 (Key Features & Architecture)
* **雲端自動化資料管線：** 透過 Amazon EventBridge 觸發 Lambda，每日定時抓取 raw data 匯入 S3 資料湖。
* **分散式 ETL 計算：** 利用 AWS Glue (Apache Spark) 執行資料清洗，運用笛卡爾積預建時空骨架，將上億筆雜亂紀錄標準化精煉為 4 萬筆輕量特徵表，大幅降低查詢延遲。
* **無伺服器高效檢索：** 導入 Amazon Athena 進行分散式 SQL 查詢，將億級資料校驗時間由傳統本機的 31 秒大幅降至 4 秒內。
* **Agentic AI 落地實作：** 整合 AWS Bedrock (Claude) 構建 Text-to-SQL 對話助理。系統能即時將使用者提問（如「清明收假何時出發車最少」）自動編譯為精準 SQL 並攔截惡意查詢，實現「自然語言即 UI」的決策輔助。
* **極致成本控制 (FinOps)：** 透過輕量化特徵提取與前後端快取分離，在 EC2 (t3.micro) 環境下以每月不到台幣 350 元的極低成本流暢支撐大數據 Web 服務。

## 🛠️ 技術棧 (Tech Stack)
* **雲端基礎設施：** AWS (S3, Lambda, EventBridge, EC2)
* **大數據與分析：** AWS Glue, Amazon Athena, PySpark, SQL
* **人工智慧：** AWS Bedrock (Claude Sonnet), Prompt Engineering
* **前端與部署：** Streamlit, Python

## 📄 專案文件 (Documents)
* **[閱讀完整大數據分析與架構報告 (PDF)](AWS期末.pdf)** <!-- 替換為你的 PDF 連結 -->