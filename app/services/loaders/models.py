from dataclasses import dataclass
from typing import Dict, Any, List
import pandas as pd


@dataclass
class ExcelLoadResult:
    text_content: Dict[str, str]
    dataframes: Dict[str, pd.DataFrame]
    sheet_names: list[str]
    metadata: Dict[str, Any]
    image_paths: Dict[str, List[str]] = None
