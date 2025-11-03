from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any

from news_agent_node import news_agent   # ✅ your original function


class NewsAgentArgs(BaseModel):
    company_name: str
    news_analyst: Any | None = None
    news_container: Any | None = None


class NewsAgentTool(BaseTool):
    name = "news_agent"
    description = "Fetches and summarizes news articles."

    args_schema = NewsAgentArgs

    def _run(self, company_name, news_analyst=None, news_container=None):
        context = {
            "company_name": company_name,
            "news_analyst": news_analyst,
            "news_container": news_container
        }
        return news_agent(context)

    async def _arun(self, **kwargs):
        return self._run(**kwargs)


news_agent_tool = NewsAgentTool()
