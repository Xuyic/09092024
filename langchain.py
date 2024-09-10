import warnings
import sys
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder

# 抑制所有弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 確保輸出編碼為 UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# 載入環境變數
load_dotenv()

# 創建聊天歷史
hist = ChatMessageHistory()

# 設定 OpenAI 模型
llm = ChatOpenAI(temperature=0)

# 定義提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你會說中文，是聰明的助理"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

output_parser = StrOutputParser()

# 創建 LLMChain
chain = prompt | llm | output_parser

# 不斷讀取用戶輸入並回應
while True:
    user_mess = input("User: ").strip()  # 用戶輸入，去除首尾空白
    
    # 檢查是否停止對話
    if user_mess.lower() == "stop":
        print("對話已停止。")
        break
    
    # 更新聊天歷史
    hist.add_user_message(user_mess)
    
    # 創建上下文
    context = {
        "input": user_mess,
        "chat_history": hist.messages[-5:]  # 最近的 5 條消息
    }

    # 呼叫链条获取最终回應
    response = chain.invoke(context)

    # 假設 response 是字符串，直接處理它
    cleaned_response = response.replace("\n\n", "\n")
    print(cleaned_response)
    
    # 更新聊天歷史
    hist.add_ai_message(cleaned_response)
