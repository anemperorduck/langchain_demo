# Pydantic 结构化输出模型定义
from typing import Optional
from pydantic import Field
from pydantic import BaseModel


class TravelDiaryEntry(BaseModel):
    title: str = Field(description = "富有诗意的日记标题，不超过20字")
    location: str = Field(description = "推测的拍摄地点，如'日本京都岚山'")
    date: Optional[str] = Field(description = "推测的日期或季节，如'2026年深秋'")
    weather: str = Field(description="天气状况，如'晴朗有微风'")
    mood: str = Field(description = "照片传递的情绪，如'宁静而喜悦'")
    diary_text: str = Field(description = "一段100字左右的第一人称日记，融入图片细节和感受")