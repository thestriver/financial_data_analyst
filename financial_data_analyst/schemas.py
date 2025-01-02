#!/usr/bin/env python
from pydantic import BaseModel
from typing import List, Optional

class DataAnalystInput(BaseModel):
    ticker_symbols: List[str]
    time_period: str
    analysis_type: str = "brief"
    specific_metrics: List[str] = ["PE", "Revenue Growth", "Profit Margins"]

class InputSchema(BaseModel):
    tool_name: str = "analyze"
    tool_input_data: DataAnalystInput