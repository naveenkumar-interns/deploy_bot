
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from datetime import datetime
import pymongo
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from typing import List

load_dotenv()

GOOGLE_API_KEY = "AIzaSyDR5hSTYjo6jbiTpHw8AEKZsuRVEEFcAJk"
hf_token = "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"
MONGO_URI = "mongodb+srv://jsckson_store:jsckson_store@cluster0.9a981.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

model_1 = "gemini-1.5-flash"
model_2 = "gemini-2.0-pro-exp-02-05"
model_3 = "gemini-2.0-flash-lite"




client = pymongo.MongoClient(MONGO_URI)
db = client["jacksonHardwareDB"]
collection = db["inventory"]
user_client = pymongo.MongoClient("mongodb+srv://sudhakaran:URvEVWjORGTkaeaq@cluster0.znyhl.mongodb.net/chatbot?retryWrites=true&w=majority&appName=Cluster0")
user_db = user_client["chatbot"]
chat_history_collection = user_db["chats"]




app = Flask(__name__)
CORS(app)


llm = ChatGoogleGenerativeAI(
    model=model_1,
    temperature=0.7,
    max_tokens=60,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,
)

embedding_cache ={}

def generate_embedding(text: str) -> List[float]:
    if text in embedding_cache:
        return embedding_cache[text]
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name="sentence-transformers/all-MiniLM-l6-v2")
    response = embeddings.embed_query(text)
    embedding_cache[text]=response
    return response


def convert_to_json(data):
    result = []
    forai = []
    for product in data:
        # Filter out unnecessary keys from metadata
        product_info = {
        'id': product.get('id'),
        'title': product.get('title'),
        'description': product.get('description'),
        'product_type': product.get('product_type'),
        'link': product.get('link'),
        'image_list': product.get('image_list'),
        'price': product.get('price'),
        'inventory_quantity': product.get('inventory_quantity'),
        'vendor': product.get('vendor')
        }
        # iteminfo = {
        # 'title': product.get('title'),
        # 'product_type': product.get('product_type'),
        # 'description': product.get('description'),
        # 'vendor': product.get('vendor'),
        # 'price': product.get('price'),
        # 'inventory_quantity': product.get('inventory_quantity')
        # }
        # forai.append(iteminfo)
        result.append(product_info)

    print(result)

    return result,forai


def get_product_search(query):
    results = collection.aggregate([
    {"$vectorSearch": {
        "queryVector": generate_embedding(query),
        "path": "embeddings",
        "numCandidates": 100,
        "limit": 4,
        # "index": "vector_search_index",
        "index": "vx",
        }}
    ])
    return convert_to_json(results)

def research_intent(chat_history):
    llm = ChatGoogleGenerativeAI(
    model=model_2,
    temperature=0.7,
    max_tokens=60,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY
    )

    try:
        prompt = ChatPromptTemplate.from_messages([
        (
        "system",
            """You are a senior research assistant. 
            Analyze the chat history to track the user's current topic and predict their request.
              Accumulate filters (e.g., specifications) until the topic changes, then reset context. 
              Respond with only a phrase summarizing the current request, prioritizing the latest input. No explanations or additional text.

                Examples:
                1. User: I need a heater.
                Bot: Heater
                User: I need a 220V one.
                Bot: Heater 220V
                User: I need it in black.
                Bot: Heater 220V Black
                User: Can you list tables?
                Bot: Table

                2. User: Show me smartphones.
                Bot: Smartphone
                User: I need one with 128GB storage.
                Bot: Smartphone 128GB
                User: Show me refrigerators.
                Bot: Refrigerator

                Analyze the conversation and return the summarizing word or phrase."""
        ),
        ("human", "{chat_history}")
    ])

        chain = prompt | llm

        response = chain.invoke({"chat_history": chat_history})

        return response.content.strip()
    except Exception as e:
        print(f"Error in research_intent: {str(e)}")
        raise

def prioritize_products(user_intent, products):

    llm = ChatGoogleGenerativeAI(
    model=model_3,
    temperature=0.7,
    max_tokens=50000,
    timeout=None,
    max_retries=2,
    google_api_key=GOOGLE_API_KEY,
)

    input_str = f"User asks for : '{user_intent}'\n Products we have: {json.dumps(products, indent=2)}"
    try:
        prompt = """ Role: You are a Product Prioritization Expert specializing in ranking products based on user intent, price constraints, and relevance. 
            Your task is to filter, reorder, and return the most relevant products that match the user's intent and budget.

            Rules for Prioritization:
            1. **Match User Intent**: 
            - Prioritize products that contain keywords from the user's intent in the title, description, or product type.
            - Stronger keyword matches (e.g., exact matches in the title) should rank higher.

            2. **Apply Price Constraints**:
            - If a price limit is specified (e.g., "under $30"), exclude products exceeding this threshold.
            - If no price limit is provided, ignore this rule.

            3. **Sort Order**:
            - First, sort by **intent relevance** (strongest keyword matches first).
            - Then, sort by **price** (low to high) within products of equal relevance.

            4. **Output Format**:
            - Return a JSON array of the top 8 most relevant products.
            - Do not modify any values inside the input data.** Only reorder and filter based on the rules.



        Examples:
        Example 1
        Intent: 'waterproof gloves under $20'
        Products:
        [
        {"id": 1, "title": "Waterproof Gloves", "price": "19.99", "inventory_quantity": 5, "description": "Waterproof"},
        {"id": 2, "title": "Leather Gloves", "price": "25.00", "inventory_quantity": 3, "description": "Durable"}
        ]
        Output:
        [
        {"id": 1, "title": "Waterproof Gloves", "price": "19.99", "inventory_quantity": 5, "description": "Waterproof"}
        ]

        Example 2
        Intent: 'touchscreen gloves'
        Products:
        [
        {"id": 4, "title": "Touchscreen Gloves", "price": "29.99", "inventory_quantity": 2, "description": "Touchscreen"},
        {"id": 5, "title": "Work Gloves", "price": "15.00", "inventory_quantity": 4, "description": "Rugged"}
        ]
        Output:
        [
        {"id": 4, "title": "Touchscreen Gloves", "price": "29.99", "inventory_quantity": 2, "description": "Touchscreen"},
        {"id": 5, "title": "Work Gloves", "price": "15.00", "inventory_quantity": 4, "description": "Rugged"}
        ]

        Task Execution:
        Now, apply these rules to the following product dataset and return the top 8 most relevant products in sorted JSON format:

        """ + input_str



        # Format the input string correctly and pass it as the 'input' variable
  
        response = llm.invoke(prompt)
        prompt = ""
  
        print("AI product result :",response.content.replace("\n", "").replace("```json", "").replace("```", "").strip())

        return json.loads(response.content.replace("\n", "").replace("```json", "").replace("```", "").strip())
    
    except Exception as e:
        print(f"Error in prioritize_products: {str(e)}")
        raise


def get_response(input_text,related_products,user_intent):
    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Jackson Hardware Store's AI assistant. Your role is to help customers find tools, hardware, or equipment,
          suggest relevant products based on their needs, and provide key details like brand, features, or availability. 
          Respond in 1-2 short, direct sentences (max 20 tokens) with no technical formatting, explanations, or symbols. 
          avoid preambles. and talk in a friendly manner.
          Actual user intention: {user_intent}
          Use related products from: {related_products}."""
    ),
    ("human", "{input}"),
])

        chain = prompt | llm

        response = chain.invoke({"input": input_text, "related_products":related_products})

        return response.content
    
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 
        
   
@app.route('/chat', methods=['POST'])
def chat_product_search():
    try:
        message = request.json
        email = message.get('email')
        if email is None:
            return jsonify({'error': 'email is required'}), 400

        query = {"Email": email} if email else {"Email": "guest_69dd2db7-11bf-49cc-934c-14fa2811bb4c"}
        chat_history = list(chat_history_collection.find(query))
        # Extract just sender and text from chat history
        chat_history = [{'sender': msg['sender'], 'text': msg['text']} 
                   for chat_doc in chat_history 
                   for msg in chat_doc.get('messages', [])]
        

        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        chat_history.append({'sender': message.get('sender'), 'text': message.get('content')})

        message.update({
            'timestamp': datetime.now().isoformat()
        })

        research_intent_response = research_intent(chat_history)
        chat_history = []

        # print("\n\nchat_history : ", chat_history)
        # print("\n\nresearch_intent_response : ", research_intent_response)

        
        related_product = get_product_search(research_intent_response)

        prioritize_products_response = prioritize_products(research_intent_response,related_product)
        related_product = ""

        # print("\n\nprioritize_products_response : ", prioritize_products_response)

        ai_response = get_response(input_text = message['content'], user_intent = research_intent_response,related_products=prioritize_products_response)
  
        
        response = {
            'content': ai_response,
            'sender': 'bot',
            'timestamp': datetime.now().isoformat(),
            'related_products_for_query':prioritize_products_response
        }        
        ai_response = ""
        return jsonify(response)
    
    except Exception as e:
        error_response = {
            "error_response" : str(e),
            'content': "I apologize, but I encountered an error. Please try again.",
            'sender': 'bot',
            'error': True,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_response), 500
    
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "working"})

if __name__ == "__main__":
    app.run(debug=True)
