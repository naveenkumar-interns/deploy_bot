
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

client = pymongo.MongoClient("mongodb+srv://jsckson_store:jsckson_store@cluster0.9a981.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["jacksonHardwareDB"]
collection = db["inventory"]
user_client = pymongo.MongoClient("mongodb+srv://sudhakaran:URvEVWjORGTkaeaq@cluster0.znyhl.mongodb.net/chatbot?retryWrites=true&w=majority&appName=Cluster0")
user_db = user_client["chatbot"]
chat_history_collection = user_db["chats"]


app = Flask(__name__)
CORS(app)


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    max_tokens=60,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

embedding_cache ={}

def generate_embedding(text: str) -> List[float]:
    if text in embedding_cache:
        return embedding_cache[text]
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=os.getenv("hf_token"), model_name="sentence-transformers/all-MiniLM-l6-v2")
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
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a senior research assistant. Your job is to analyze the user's chat history, "
             "track relevant topics, and predict exactly what the user is looking for now. "
             "If the user changes the topic, clear old context and respond accordingly.\n\n"
             "You should return only a single word or phrase summarizing the request based on cumulative filters.\n\n"
             
             "Example 1:\n"
             "User: I need a heater.\n"
             "Bot: Heater\n"
             "User: I need a 220V one.\n"
             "Bot: Heater 220V\n"
             "User: I need it in black.\n"
             "Bot: Heater 220V Black\n"
             "User: Can you list tables?\n"
             "Bot: Table (Context reset!)\n\n"

             "Example 2:\n"
             "User: Show me smartphones.\n"
             "Bot: Smartphone\n"
             "User: I need one with 128GB storage.\n"
             "Bot: Smartphone 128GB\n"
             "User: Show me refrigerators.\n"
             "Bot: Refrigerator (Context reset!)\n\n"

             "Now, analyze the following conversation and return the cumulative filtered request."),
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
    model="gemini-1.5-flash",
    temperature=0.7,
    max_tokens=10000,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

    input_str = f"Intent: '{user_intent}'\nProducts: {json.dumps(products, indent=2)}"
    try:
        prompt = """You are a Product Prioritization Expert specializing in ranking products based on user intent, price constraints, and relevance. Your goal is to filter, reorder, and return the most relevant, price-appropriate products first.

        Rules for Prioritization:
        1 Match User Intent: Prioritize products that contain keywords from the intent in the title, description, or product type.
        2 Apply Price Constraints: If a price limit is provided (e.g., 'under $30'), exclude items exceeding this threshold.
        3 Sort Order:
        - First, by intent relevance (strongest keyword matches first).
        - Then, by price (low to high) within relevant matches.
        4 Output Format: Return a JSON array of the top 8 most relevant products.

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


def get_response(input_text,related_products,chat_history=None):
    try:
        prompt = ChatPromptTemplate.from_messages([
        (
        "system",
        """You are Jackson Hardware Store's AI assistant. Your job is to:
        1. Help customers find the right tools, hardware, or equipment.
        2. Suggest relevant products based on customer needs and related items.
        3. Share key product details like brand, features, use cases, and availability.
        5. note important: Avoid using technical formatting like new line symbols, markdown symbols *, _, etc., or bullet points.
        6.  Respond in a short, direct manner <maximum 1-2 brief sentences> with only the most relevant information. and max tokens 20

        note: act as a chatbot 
        Deliver the response here in plain text without any formatting.
        related products for the recent user query: {related_products}
        """,
    ),
    ("human", "{input}"),
])

        chain = prompt | llm

        response = chain.invoke({"input": input_text, "related_products":related_products, "chathistory": chat_history})

        return response.content
    
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 
        


   
@app.route('/chat', methods=['POST'])
def chat_product_search():
    try:
        message = request.json
        email = message.get('email')
        # if email is None:
        #     return jsonify({'error': 'email is required'}), 400

        query = {"Email": email} if email else {"Email": "guest_69dd2db7-11bf-49cc-934c-14fa2811bb4c"}
        chat_history = list(chat_history_collection.find(query))
        # Extract just sender and text from chat history
        chat_history = [{'sender': msg['sender'], 'text': msg['text']} 
                   for chat_doc in chat_history 
                   for msg in chat_doc.get('messages', [])]
        

        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

        chat_history.append(message)

        message.update({
            'timestamp': datetime.now().isoformat()
        })

        research_intent_response = research_intent(chat_history)

        print("chat_history : ", chat_history)
        print("\n\nresearch_intent_response : ", research_intent_response)

        
        related_product = get_product_search(research_intent_response)

        prioritize_products_response = prioritize_products(research_intent_response,related_product)
        related_product = ""

        print("\n\nprioritize_products_response : ", prioritize_products_response)

        ai_response = get_response(input_text = message['content'], related_products=prioritize_products_response, chat_history=chat_history)
        chat_history = []
        
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
