

# 💹 TradeXFinal — Multi-Agent AI Trading Decision System (LangGraph + Hybrid Forecasting)

**TradeXFinal** is an advanced **AI-powered stock analysis platform** that integrates
📈 **Hybrid price forecasting**,
📰 **Market news summarization**,
💬 **Reddit sentiment analysis**, and
🧠 **multi-agent debate reasoning via LangGraph**
to generate **balanced Buy / Sell / Hold recommendations**.

It brings together forecasting models, LLMs, and autonomous financial agents to simulate realistic investment research — all accessible through an interactive **Streamlit interface**.

---

## 🚀 Key Features

| Module                 | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| Hybrid Forecast Engine | Prophet + XGBoost ensemble for accurate price projections |
| News Agent             | Extracts and summarizes latest headlines for the stock    |
| Reddit Agent           | Analyzes retail investor sentiment                        |
| Bull vs Bear Agents    | Debate market positions using LangGraph                   |
| Risk Analysis Agents   | Weighs upside opportunity vs downside risk                |
| Strategy Agent         | Produces final trading recommendation                     |
| CSV Logging            | Tracks stock-specific outputs for consistency             |
| Modular UI             | Each agent has a separate Streamlit UI section            |

---

## 🤖 Multi-Agent Architecture (LangGraph Workflow)

```
User → Forecast Engine
        ↓
 Market News Agent ──────────────► Reddit Sentiment Agent
        ↓                                      ↓
 Bull Agent ↔ Bear Agent ↔ Facilitator Agent (Research)
        ↓
 Positive Risk Agent ↔ Negative Risk Agent ↔ Risk Facilitator
        ↓
 Strategy Agent → Final Recommendation
```

This structure mimics professional financial decision-making by combining **market data + sentiment + risk + strategic views**.

---

## 🔍 What the System Generates

After providing a stock ticker and share amount:

| Output            | Details                                                  |
| ----------------- | -------------------------------------------------------- |
| Current Matrix    | Key financial indicators (RSI, SMA, MACD, PE, ATR, etc.) |
| Forecast Metrics  | MAE, RMSE, MAPE, SMAPE                                   |
| Future Averages   | Next day, week, month, quarter, 6-month & 1-year price   |
| News Summary      | Condensed financial headlines                            |
| Reddit Summary    | Opinion and psychology of market participants            |
| Researcher Debate | Bull vs Bear arguments with facilitator summary          |
| Risk Debate       | Positive vs negative risks evaluated                     |
| Final Strategy    | Actionable insight with reasoning                        |

---

## 📂 Repository Structure

```
TradeXFinal/
│
├── tradexfinal.py                # Main Streamlit application
├── ticker_dataset.py
├── hybrid_model.py
├── news_agent_node.py
├── news_ui_agent_node.py
├── reddit_agent_node.py
├── reddit_ui.py
├── redditfinal.py
├── bearbull.py
├── bearbullui.py
├── riskanalysis.py
├── riskui.py
├── strategies.py
├── strategiesui.py
├── graph_lang.py / graph_lang2.py / langraphnew.py     # LangGraph modules
├── schema.py
├── market_news.json
├── stock_csvs/ (if present)
├── requirements.txt
└── README.md
```

---

## ▶ How to Run

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Launch the application

```
streamlit run tradexfinal.py
```

### 3️⃣ Provide:

✔ Stock ticker (AAPL, TSLA, MSFT…)
✔ Buy / Sell / Hold
✔ Number of shares

The platform will automatically generate the full multi-agent analysis.

---

## ⚠ Disclaimer

TradeXFinal is for **education and research** in financial AI and multi-agent systems.
It **does not provide certified investment advice** and must not be used for real-money trading without independent financial consultation.

---

## 🧠 Roadmap / Future Enhancements

* Broker API integration (paper trading)
* Conversation memory between agents
* Cryptocurrency & forex support
* Portfolio optimization agent
* Reinforcement learning for execution timing
* GPU acceleration for large-scale forecasting

---

👥 Contributors

Muhammed Rashid

Naveed PN

Afrah Anas

Ahsana
