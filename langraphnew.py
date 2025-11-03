from langgraph.graph import StateGraph
from schema import MarketState
from news_agent_node import news_agent
from news_ui_agent_node import display_news_ui

graph = StateGraph(state_schema=MarketState)

# Add Python functions directly
graph.add_node("news_agent", news_agent)
graph.add_node("news_ui_agent", display_news_ui)

# Execution order
graph.set_entry_point("news_agent")
graph.add_edge("news_agent", "news_ui_agent")

graph.set_finish_point("news_ui_agent")
# Compile
news_app = graph.compile()
