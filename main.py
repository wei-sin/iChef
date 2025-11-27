# main.py

import os
import sys
from typing import List
from pathlib import Path

# 引入函式庫
try:
    from google import genai
    from dotenv import load_dotenv
except ImportError:
    print("❌ 錯誤：請確保已安裝 google-genai 和 python-dotenv。")
    print("請運行: pip install google-genai python-dotenv")
    sys.exit(1)

# =================================================================
# 1. 環境設定與 Client 初始化
# =================================================================

# 從 .env 檔案中載入環境變數
DOTENV_PATH = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=DOTENV_PATH)

# 取得 API Key
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ 錯誤：GEMINI_API_KEY 未設定。請檢查您的 .env 檔案並確保格式正確。")

try:
    # 初始化 Gemini 客戶端
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"❌ 初始化 Gemini 客戶端失敗，程式將無法運行。錯誤：{e}")
    sys.exit(1)


# =================================================================
# 2. AI 核心邏輯函式
# =================================================================

def create_recipe_prompt(ingredients: List[str], preference: str) -> str:
    """
    根據食材清單和個人偏好，生成給 LLM 的提示詞。
    """
    ingredients_str = ", ".join(ingredients)

    prompt = f"""
    你是一位專業的食譜設計師和營養師。
    
    請根據我提供的現有食材清單：【{ingredients_str}】，以及我的飲食偏好：【{preference}】，為我設計一個完整的食譜。

    請輸出一個清晰、結構化的食譜，包含以下欄位：
    1. 食譜名稱 (創意且吸引人)
    2. 客製化調整說明 (說明你如何根據我的偏好調整了食譜)
    3. 所需食材清單 (請列出所有需要的食材，包含調味料，並標明用量)
    4. 營養速覽 (估計的卡路里、蛋白質、脂肪、碳水化合物含量)
    5. 詳細烹飪步驟 (分點列出，清晰易懂)
    
    請確保食譜內容健康且易於執行。
    """
    return prompt

def generate_recipe_from_ai(ingredients_text: str, preference_text: str) -> str:
    """
    處理使用者輸入，呼叫 Gemini API 生成食譜。
    """
    # 處理食材輸入，移除空白並過濾空值
    ingredients = [i.strip() for i in ingredients_text.split(',') if i.strip()]
    
    if not ingredients:
        return "🚨 請輸入至少一項食材！"
    
    final_prompt = create_recipe_prompt(ingredients, preference_text)
    model_name = "gemini-2.5-flash"

    try:
        # 執行 API 呼叫
        response = client.models.generate_content(
            model=model_name,
            contents=final_prompt,
            config={"temperature": 0.7} # 調整創意程度
        )
        return response.text
    
    except Exception as e:
        return f"\n❌ 呼叫 AI 失敗。請檢查您的 API Key 或網路連線。錯誤：{e}"

# =================================================================
# 3. 命令列互動介面 (CLI)
# =================================================================

def main():
    """
    主函式，處理使用者輸入和輸出。
    """
    print("="*45)
    print("✨ AI 食譜客製化工具 (CLI 版本) ✨")
    print("="*45)
    
    # 接收食材輸入
    ingredients_input = input("請輸入您現有的食材清單 (用逗號分隔，例如: 雞蛋, 番茄, 麵粉): ").strip()
    
    # 接收偏好輸入
    preference_input = input("請輸入您的飲食偏好或客製化要求 (例如：低碳水、少油): ").strip()
    
    if not ingredients_input:
        print("\n❌ 輸入無效：食材清單不能為空。")
        return

    print("\n🔄 正在生成並客製化食譜中，請稍候...")
    
    # 呼叫核心邏輯
    recipe_output = generate_recipe_from_ai(ingredients_input, preference_input)
    
    # 輸出結果
    print("\n" + "="*45)
    print("✅ 客製化食譜結果：")
    print("="*45)
    print(recipe_output)
    print("="*45)

if __name__ == "__main__":
    main()