# Project Context

## 專案背景 (Project Context)

### 目的 (Purpose)
本專案為「HW3: Email Spam Classification」課程作業（物聯網應用與資料分析），目標是建立一個可複現的電子郵件垃圾郵件分類流程。重點包括：資料前處理、特徵工程、模型訓練與評估、以及最小可部署的模型匯出（例如使用 joblib）。此文件記錄專案約定、技術棧、執行與測試方式，供助教、同學或自動化代理使用。

### 範圍與成功判準
- 能從原始資料（CSV 或類似格式）復現資料處理與訓練流程。
- 提供可重現的實驗（固定 random seed、版本化依賴）。
- 提供至少一個 baseline 模型（例如 Logistic Regression 或 Random Forest）與評估報表（accuracy, precision, recall, F1, AUC）。

### 假設 (Assumptions)
- 主要開發語言為 Python（若非 Python，請在此文件更新註明）。
- 常見開發環境包含 Jupyter Notebook / VS Code。若您的環境不同，請告知以便我調整建議。

## 技術棧 (Tech Stack)
- 語言: Python 3.8+（建議）
- 資料處理: pandas, numpy
- 特徵 / ML: scikit-learn
- 模型序列化: joblib 或 pickle
- 視覺化: matplotlib, seaborn
- 互動環境: Jupyter Notebook / JupyterLab
- 測試: pytest
- 風格與靜態檢查: black (格式化), flake8 (lint)

（註：如果專案改用其他框架如 TensorFlow / PyTorch，請更新上方清單。）

## 專案約定 (Project Conventions)

### 檔案/目錄結構建議
- data/
	- raw/         # 原始不可變動之資料
	- processed/   # 經前處理後用於訓練/評估的資料
- notebooks/     # 互動式實驗（註明 notebook 的目的）
- src/           # 可重用程式碼 (data/, features/, models/, utils/)
- models/        # 訓練完成並序列化的模型檔
- tests/         # pytest 單元/整合測試
- reports/       # 評估報表、圖表

### 程式碼風格 (Code Style)
- 遵循 PEP8 命名與風格慣例。
- 使用 black 進行自動格式化（預設 line-length 88，可視專案調整）。
- 變數/函式命名：snake_case；類別：PascalCase。

### 開發/架構模式 (Architecture Patterns)
- Notebook-first for exploration, then extract reusable code into `src/`.
- 將資料處理、特徵工程、模型定義、訓練與評估分層（清晰職責分離）。
- 配置與參數應集中管理（例如使用 YAML/JSON 或一個小型 `config.py`）。

### 命名與提交約定 (Commit & Branching)
- 主要分支：`main`（穩定、可發佈）
- 開發分支：`develop`（可選）
- 功能分支：`feature/<short-description>`
- Bugfix/Hotfix：`fix/<short-description>` 或 `hotfix/<short-description>`
- Pull Request: 每個 PR 需包含變更說明、相關測試與簡短執行說明。
- Commit message 採用簡潔描述並盡量使用 Conventional Commits（例如 `feat: add preprocessing pipeline`, `fix: correct label mapping`）。

## 測試策略 (Testing Strategy)
- 單元測試：針對資料前處理、特徵工程函式、評估工具撰寫 pytest 測試。
- 整合測試：建立一個快速的端對端 smoke test（使用小型 fixture 資料），驗證從資料讀取 -> 特徵 -> 訓練 -> 預測流程可執行。
- 隨機性控制：測試中明確設定 random_state/seed 以保證可重現性。

## 執行與重現 (How to run / Reproducibility)
推薦使用虛擬環境 (venv 或 conda)。簡要 PowerShell 範例：

```powershell
# 建立並啟用 venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1
# 安裝依賴（假設有 requirements.txt）
pip install -r requirements.txt
```

常用工作流程：
- 在 `notebooks/` 查看實驗；將可重用程式碼放到 `src/`。
- 使用 `python -m src.train --config config.yaml` 之類的 CLI 觸發訓練（視專案實作而定）。

## 執行品質閘 (Quality gates)
- 建置: N/A（純 Python 腳本），若使用 CI，可加入 `pip install -r requirements-dev.txt` 並執行測試。
- Lint/Typecheck: 建議加入 `black --check` 與 `flake8`。
- Tests: 使用 `pytest -q` 運行測試集。

## 領域背景 (Domain Context)
- 任務：二分類（spam / ham）。
- 常見特徵：詞袋（Bag-of-Words）、TF-IDF、標題與內容長度、是否含連結、發件人或主旨模式等。
- 評估重點：偽陽性（誤判正常郵件為垃圾郵件）常比偽陰性更為敏感，評估時應同時注意 precision 與 recall，並依需求調整 decision threshold。

## 重要限制 (Important Constraints)
- 資料隱私：請勿上傳含有個資的真實郵件到公開服務。
- 計算資源：若使用大型模型，訓練時間與記憶體需求應列入考量；目前建議使用輕量模型以利本地執行。
- 資料授權：確認使用之資料集（如 Enron, SpamAssassin）授權與用途限制。

## 外部相依 (External Dependencies)
- 主要套件：pandas, numpy, scikit-learn, joblib, matplotlib, seaborn
- 開發工具：black, flake8, pytest
- 可選雲端/服務：無（本作業建議本地可複現）

## 維護者與聯絡方式 (Maintainers / Contacts)
- 作者 / 學生：請在此填入你的名字與 email。
- 助教 / 指導老師：請填入課程助教或老師聯絡資訊（若有）。

## 後續建議 (Next steps / Proactive extras)
1. 新增 `requirements.txt`（列出最小依賴），並把 `notebooks/` 與 `src/` 初始樣板加入 repo。
2. 增加一個簡單的 `README.md`，描述如何快速執行 baseline 實驗。
3. 如果需要，我可以幫你把這些檔案 scaffold 出來（`requirements.txt`, `README.md`, 測試樣板）。

---

註：上述內容基於一般學生 ML 作業慣例與倘若 repository 尚未包含明確程式碼的情況所做的合理假設；若你的專案使用不同語言或工具（例如 R、MATLAB，或使用深度學習框架），請告訴我，我會依專案實際狀態調整內容。
