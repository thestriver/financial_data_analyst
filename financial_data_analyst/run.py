#!/usr/bin/env python
from dotenv import load_dotenv
import os
import yfinance as yf
from naptha_sdk.schemas import AgentRunInput
from naptha_sdk.utils import get_logger
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from financial_data_analyst.schemas import DataAnalystInput, InputSchema

load_dotenv()
logger = get_logger(__name__)

print("OpenAI Key loaded:", bool(os.getenv("OPENAI_API_KEY")))

class FinancialDataAnalyst:
    def __init__(self, module_run):
        self.module_run = module_run
        self.llm_config = module_run.deployment.config.llm_config 
        self.setup_llm()

    def setup_llm(self):
        """Initialize LLM configuration"""
        self.llm = ChatOpenAI(
            model_name=self.llm_config.model,
            temperature=self.llm_config.temperature
        )

    def get_financial_data(self, symbol: str, period: str) -> Dict:
        ticker = yf.Ticker(symbol)
        return {
            'info': ticker.info,
            'income_stmt': ticker.income_stmt,
            'balance_sheet': ticker.balance_sheet,
            'calendar': ticker.calendar,
            'history': ticker.history(period=period)
        }

    def analyze_metrics(self, data: Dict, metrics: List[str]) -> Dict:
        analysis = {}
        for metric in metrics:
            if metric in data['info']:
                analysis[metric] = data['info'][metric]
        return analysis

    def analyze(self, input_data: DataAnalystInput) -> Dict[str, Any]:
        try:
            results = {}
            for symbol in input_data.ticker_symbols:
                data = self.get_financial_data(symbol, input_data.time_period)
                metric_analysis = self.analyze_metrics(data, input_data.specific_metrics)
                
                prompt = f"""
                Analyze the following financial data for {symbol}:
                Financial Metrics: {metric_analysis}
                Income Statement: {data['income_stmt'].to_dict()}
                Balance Sheet: {data['balance_sheet'].to_dict()}
                
                Provide a {input_data.analysis_type} analysis focusing on:
                1. Key financial metrics and their trends
                2. Financial statement analysis
                3. Notable changes or anomalies
                4. Financial health indicators
                """
                
                response = self.llm.invoke(prompt)
                results[symbol] = {
                    "metrics": metric_analysis,
                    "analysis": response.content
                }
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

def run(module_run, *args, **kwargs):
    """Main entry point for the financial data analyst"""
    analyst = FinancialDataAnalyst(module_run)
    
    if isinstance(module_run.inputs, dict):
        input_params = InputSchema(**module_run.inputs)
    else:
        input_params = module_run.inputs
        
    return analyst.analyze(input_params.tool_input_data)

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment(
        "agent", 
        "financial_data_analyst/configs/deployment.json",
        node_url=os.getenv("NODE_URL")
    ))

    input_params = InputSchema(
        tool_name="analyze",
        tool_input_data=DataAnalystInput(
            ticker_symbols=["AAPL", "MSFT"],
            time_period="1y",
            analysis_type="comprehensive",
            specific_metrics=["PE", "Revenue Growth", "Profit Margins"]
        )
    )

    module_run = AgentRunInput(
        inputs=input_params,
        deployment=deployment,
        consumer_id=naptha.user.id,
    )

    response = run(module_run)
    print("\nFinancial Analysis Results:")
    print("=========================")
    print(response)