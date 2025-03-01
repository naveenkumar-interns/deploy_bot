from flask import Flask, request, jsonify
from flask_cors import CORS
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
    print(type(text))
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
        iteminfo = {
        'title': product.get('title'),
        'product_type': product.get('product_type'),
        'description': product.get('description'),
        'vendor': product.get('vendor'),
        'price': product.get('price'),
        'inventory_quantity': product.get('inventory_quantity')
        }
        forai.append(iteminfo)
        result.append(product_info)

    print(result)

    return result,forai


def get_product_search(query):
    results = collection.aggregate([
    {"$vectorSearch": {
        "queryVector": generate_embedding(query),
        "path": "embeddings",
        "numCandidates": 100,
        "limit": 5,
        # "index": "vector_search_index",
        "index": "vx",
        }}
    ])
    return convert_to_json(results)


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
        chat history: {chathistory}
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

        message.update({
            'timestamp': datetime.now().isoformat()
        })
        chat_history.append(message)
        
        related_product,for_ai = get_product_search(message['content'])

        ai_response = get_response(input_text = message['content'], related_products=for_ai, chat_history=chat_history)
        
        response = {
            'content': ai_response,
            'sender': 'bot',
            'timestamp': datetime.now().isoformat(),
            'related_products_for_query':related_product
        }        
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
