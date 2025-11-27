# app.py
  #Local URL: http://localhost:8501
  #Network URL: http://172.20.10.2:8501
  #使用 python -m streamlit 執行您的 app.py 檔案
  #(ai-recipe) G:\...\iChef> python -m streamlit run app.py

import os
import sys
from typing import List
from pathlib import Path

# Streamlit UI 函式庫
import streamlit as st
from PIL import Image

# Google AI SDK
try:
    from google import genai
    from dotenv import load_dotenv
except ImportError:
    st.error("❌ 缺少必要的函式庫。請運行: pip install google-genai python-dotenv streamlit pillow")
    st.stop()

# =================================================================
# 1. 環境設定與 Client 初始化
# =================================================================

# 顯式指定 .env 檔案路徑並載入，確保在任何運行環境下都能正確找到金鑰
DOTENV_PATH = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=DOTENV_PATH) 
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("❌ 錯誤：GEMINI_API_KEY 未設定。請檢查您的 .env 檔案 (與 app.py 同目錄)。")
    st.stop()

try:
    # 初始化 Gemini 客戶端
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    st.error(f"❌ 初始化 Gemini 客戶端失敗，請檢查 API Key 是否有效。錯誤：{e}")
    st.stop()


# =================================================================
# 2. AI 核心邏輯函式
# =================================================================

def create_recipe_prompt(ingredients: List[str], preference: str) -> str:
    """根據食材清單和個人偏好，生成給 LLM 的提示詞。"""
    ingredients_str = ", ".join(ingredients)
    
    # 使用 Markdown 格式強化 LLM 的輸出結構
    prompt = f"""
    您是一位專業的食譜設計師和營養師。
    
    請根據我提供的現有食材清單：**【{ingredients_str}】**，以及我的飲食偏好：**【{preference}】**，為我設計一個完整的食譜。

    請使用 Markdown 格式，輸出一個清晰、結構化的食譜，包含以下欄位：
    # 食譜名稱 (創意且吸引人)
    
    ## 客製化調整說明
    (說明你如何根據我的偏好和現有食材調整了食譜內容)
    
    ## 所需食材清單
    (請列出所有需要的食材，包含調味料，並標明用量)
    
    ## 營養速覽
    (估計的卡路里、蛋白質、脂肪、碳水化合物含量)
    
    ## 詳細烹飪步驟
    (分點列出，清晰易懂)
    
    請確保食譜內容健康且易於執行。
    """
    return prompt

@st.cache_data(show_spinner=False)
def generate_recipe_from_ai(ingredients_text: str, preference_text: str) -> str:
    """呼叫 LLM 進行食譜生成與客製化。"""
    ingredients = [i.strip() for i in ingredients_text.split(',') if i.strip()]
    
    if not ingredients:
        return "🚨 你不乖!沒輸入!吃空氣去吧！"
    
    final_prompt = create_recipe_prompt(ingredients, preference_text)
    model_name = "gemini-2.5-flash"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=final_prompt,
            config={"temperature": 0.7} 
        )
        return response.text
    
    except Exception as e:
        return f"❌ 呼叫 AI 失敗。錯誤：{e}"

# 移除 @st.cache_data 避免開發階段緩存錯誤
def generate_ingredients_from_image(image: Image.Image) -> str:
    """呼叫 Gemini Pro Vision API 辨識圖片中的食材。"""
    
    prompt = "請詳細辨識圖片中的所有食材，只列出食材名稱，以逗號分隔。請勿提供烹飪建議，只輸出食材清單。"
    model_name = "gemini-2.5-flash" # 使用支援多模態的 flash 模型

    # 確保圖片和文字提示都正確傳入
    contents = [prompt, image]

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents, 
            config={"temperature": 0.3}
        )
        
        ingredients_text = response.text.strip()
        
        if not ingredients_text or "無法辨識" in ingredients_text:
            return ""
        
        return ingredients_text
    
    except Exception as e:
        # 將錯誤詳細印出到 Streamlit 介面
        return f"❌ 圖片辨識失敗：請檢查 API 權限或圖片格式。詳細錯誤：{e}"

# =================================================================
# 3. Streamlit 前端介面設計
# =================================================================

def main_app():
    st.set_page_config(page_title="iChef 食譜客製化工具", layout="wide")
    
    st.title("👨‍🍳 iChef 食譜客製化與食材管家")
    st.markdown("歡迎使用 iChef。上傳食材圖片或輸入清單，讓 iChef 為您打造專屬食譜。")
    st.markdown("---")
    
    # 初始化 Session State (如果沒有被定義，則賦予空字串)
    if "ingredients_text" not in st.session_state:
        st.session_state.ingredients_text = ""
    if "recipe_output" not in st.session_state:
        st.session_state.recipe_output = ""
    if "last_upload_name" not in st.session_state:
        st.session_state.last_upload_name = None


    # ----- 圖片上傳區 -----
    col_img, col_input = st.columns([1, 2])
    
    with col_img:
        st.subheader("📸 圖片上傳")
        uploaded_file = st.file_uploader("上傳冰箱/食材圖片", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
        
        # 處理圖片上傳
        if uploaded_file is not None:
            # 確保圖片只處理一次，或檔案名改變時才處理
            if st.session_state.last_upload_name != uploaded_file.name:
                
                st.session_state.last_upload_name = uploaded_file.name # 立即更新檔案名
                
                image = Image.open(uploaded_file)
                st.image(image, caption="您上傳的食材圖片", use_container_width=True)
                
                with st.spinner("iChef 正在辨識圖片中的食材..."):
                    identified_ingredients = generate_ingredients_from_image(image)
                    
                    if identified_ingredients and not identified_ingredients.startswith("❌"):
                        st.session_state.ingredients_text = identified_ingredients
                        st.success(f"圖片辨識完成！已自動帶入食材清單。")
                    elif identified_ingredients.startswith("❌"):
                         st.error(identified_ingredients)
                    else:
                        st.warning("圖片中未辨識出明確食材，請嘗試手動輸入。")
                        st.session_state.ingredients_text = ""
                
                # 關鍵修正：圖片處理完畢，執行重跑，讓 Text Area 顯示新的 session state 值
                st.rerun() 
            else:
                # 圖片已上傳但沒有新的檔案，僅顯示圖片
                image = Image.open(uploaded_file)
                st.image(image, caption="您上傳的食材圖片", use_container_width=True)


    # ----- 輸入與偏好設定區 -----
    with col_input:
        st.subheader("📝 食材與偏好設定")
        
        # 關鍵修正：使用 session state 變數名作為 key。
        # Text Area 會自動將其值寫入 st.session_state.ingredients_text
        st.text_area(
            "1. 現有食材清單 (用逗號分隔)", 
            value=st.session_state.ingredients_text, # 使用 session state 中的值來初始化
            placeholder="例如：雞蛋, 牛絞肉, 洋蔥, 番茄, 醬油",
            key="ingredients_text", # <-- 關鍵修正：將 key 設為 'ingredients_text'
            help="AI 辨識後會自動填入，您也可以在此手動修改或新增。"
        )
        
        # 飲食偏好
        preference_input = st.text_area(
            "2. 飲食偏好與客製化要求",
            placeholder="例如：低碳水、少油少鹽、無麩質、不使用烤箱，烹飪時間 20 分鐘內完成。",
            key="preference_text"
        )
        
        # 讀取當前食材 (從 key="ingredients_text" 自動更新後的 state)
        current_ingredients = st.session_state.get('ingredients_text', '')

        # 處理按鈕點擊
        if st.button("✨ 生成客製化食譜 ✨", type="primary", use_container_width=True):
            if not current_ingredients: 
                st.warning("不輸入就吃空氣去吧！")
                return
                
            # 確保 AI 在運行時會顯示進度
            with st.spinner("🔄 iChef 正在為您客製化食譜中，請稍候..."):
                # 呼叫核心邏輯，使用從 st.session_state 讀取的值
                recipe_result = generate_recipe_from_ai(current_ingredients, preference_input)
                
                # 將結果儲存在 session state 中以便顯示
                st.session_state.recipe_output = recipe_result
    
    # ----- 輸出結果區 (放在下方，讓介面更清晰) -----
    if st.session_state.get("recipe_output"):
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("---")
        st.header("✅ 客製化食譜結果")
        
        # 使用 st.markdown 渲染 LLM 輸出的 Markdown 格式
        st.markdown(st.session_state.recipe_output)
        st.markdown("---")

if __name__ == "__main__":
    main_app()