import io
import base64
import os
import requests
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification, pipeline
import google.generativeai as genai
from wordcloud import WordCloud, STOPWORDS
import pymongo
from fastapi.encoders import jsonable_encoder
from bson import ObjectId
from google.cloud import storage
from contextlib import asynccontextmanager
from fastapi.middleware import cors

# Environment setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.path.dirname(__file__), "sacred-truck-473222-n2-30bd10f14101.json")

# Configuration
GCS_BUCKET = "janmatra-storage-bucket"
MONGO_URL = "mongodb+srv://anshvahini16:Curet24.Nelll@volume-logs.iwoipqu.mongodb.net/?retryWrites=true&w=majority&appName=volume-logs"
SENTIMENT_MODEL_PATH = "xlm-roberta-zero-shot"
TOKENIZER_PATH = "xlm-roberta-base"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDBRkT19u526TWs1w2_s4rE1-DgTu5fLbg")

# Global variables - EXACTLY as original
tokenizer = None
sentiment_model = None
gemini_model = None
fallback_pipeline = None
mongo_client = None
db = None
collection = None
SENTIMENT_LABELS = ["Positive", "Negative", "Neutral", "Suggestive"]
SOURCE_TITLE = ""

# Initialize FastAPI app
app = FastAPI(title="Comment Processing API", version="1.0")

# CORS middleware - EXACTLY as original
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CommentRequest(BaseModel):
    comments: List[str]
    use_gemini_summary: Optional[bool] = True
    max_summary_length: Optional[int] = 1500
    min_summary_length: Optional[int] = 1000

class ExternalDatabaseConfig(BaseModel):
    """Configuration for connecting to external user databases"""
    database_type: str  # "mongodb", "mysql", "postgresql", "api"
    connection_string: Optional[str] = None
    api_endpoint: Optional[str] = None
    headers: Optional[Dict[str, str]] = {}
    query_params: Optional[Dict[str, str]] = {}

class ImportExternalDataRequest(BaseModel):
    """Import data from external user databases"""
    database_config: ExternalDatabaseConfig
    query: Optional[str] = None  # SQL query or collection name
    use_gemini_summary: Optional[bool] = True
    max_summary_length: Optional[int] = 1500
    min_summary_length: Optional[int] = 1000

# ORIGINAL analyze-extension function - PRESERVED EXACTLY
@app.post("/analyze-extension")
async def analyze_extension_data(extension_data: List[Any]):
    global SOURCE_TITLE
    
    print(f"Received extension data: {len(extension_data)} items")
    
    if not extension_data:
        raise HTTPException(status_code=400, detail="No comments provided")
    
    SOURCE_TITLE = extension_data[0].get('source_title', 'Unknown Source')
    comment_texts = []
    for i, item in enumerate(extension_data):
        print(f"Item {i}: {type(item)} - {item.keys() if isinstance(item, dict) else 'Not a dict'}")
        if isinstance(item, dict) and 'text' in item:
            text = item['text']
            if text and text.strip():
                comment_texts.append(text.strip())
                print(f"Added comment: {text[:50]}...")
    
    print(f"Extracted {len(comment_texts)} valid comments")
    
    if not comment_texts:
        raise HTTPException(status_code=400, detail="No valid comment texts found")

    request = CommentRequest(comments=comment_texts)
    return await analyze_comments(request)

# ORIGINAL download_from_gcs function - PRESERVED EXACTLY
def download_from_gcs(gcs_path: str, local_dir: str = "./models/"):
    print(f"📥 Downloading files from GCS: {gcs_path}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)

    blobs = list(bucket.list_blobs(prefix=gcs_path))
    if not blobs:
        print(f"⚠ No files found in bucket path: {gcs_path}")
        return

    for blob in blobs:
        local_path = os.path.join(local_dir, blob.name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            print(f"✅ Already exists, skipping: {local_path}")
            continue
        
        print(f"⬇ Downloading: {blob.name}")
        blob.download_to_filename(local_path)
        print(f"✅ Downloaded: {blob.name} -> {local_path}")

# ORIGINAL load_models function - PRESERVED EXACTLY
def load_models():
    global tokenizer, sentiment_model, gemini_model, fallback_pipeline

    try:
        print("Attempting to load custom sentiment analysis model...")

        if not (os.path.exists(SENTIMENT_MODEL_PATH) and os.path.exists(TOKENIZER_PATH)):
            print("Local models not found. Downloading from GCS...")
            download_from_gcs(f"xlm-roberta-zero-shot")
            download_from_gcs(f"xlm-roberta-base")

        tokenizer = XLMRobertaTokenizerFast.from_pretrained(TOKENIZER_PATH)
        sentiment_model = XLMRobertaForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
        sentiment_model.eval()
        print("Custom models loaded successfully!")

        print("Configuring Gemini API...")
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("Gemini API configured!")

        print("All models loaded successfully!")

    except Exception as e:
        print(f"Error loading models: {e}")

# ORIGINAL predict_sentiment function - PRESERVED EXACTLY
def predict_sentiment(comment: str) -> dict:
    global tokenizer, sentiment_model, fallback_pipeline
    
    try:
        # Try custom model first
        if tokenizer and sentiment_model:
            inputs = tokenizer(
                comment, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            
            with torch.no_grad():
                outputs = sentiment_model(**inputs)
                logits = outputs.logits
                
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
                predicted_label = SENTIMENT_LABELS[predicted_class_idx]
                confidence = probabilities[0][predicted_class_idx].item()
            
            return {
                "sentiment": predicted_label,
                "confidence": round(confidence, 4),
                "all_probabilities": {
                    label: round(prob.item(), 4) 
                    for label, prob in zip(SENTIMENT_LABELS, probabilities[0])
                }
            }
        
        # Use fallback pipeline
        elif fallback_pipeline:
            results = fallback_pipeline(comment)
            
            # Map the fallback results to our format
            sentiment_map = {
                "LABEL_0": "Negative",
                "LABEL_1": "Neutral", 
                "LABEL_2": "Positive",
                "negative": "Negative",
                "neutral": "Neutral",
                "positive": "Positive"
            }
            
            best_result = max(results[0], key=lambda x: x['score'])
            mapped_sentiment = sentiment_map.get(best_result['label'], best_result['label'])
            
            # Create probabilities dict
            all_probs = {
                "Positive": 0.0,
                "Negative": 0.0, 
                "Neutral": 0.0,
                "Suggestive": 0.0
            }
            
            for result in results[0]:
                mapped_label = sentiment_map.get(result['label'], result['label'])
                if mapped_label in all_probs:
                    all_probs[mapped_label] = round(result['score'], 4)
            
            return {
                "sentiment": mapped_sentiment,
                "confidence": round(best_result['score'], 4),
                "all_probabilities": all_probs
            }
        
        else:
            # Basic fallback
            return {
                "sentiment": "Neutral",
                "confidence": 0.5,
                "all_probabilities": {
                    "Positive": 0.25,
                    "Negative": 0.25,
                    "Neutral": 0.5,
                    "Suggestive": 0.0
                }
            }
    
    except Exception as e:
        print(f"Error in predict_sentiment: {e}")
        return {
            "sentiment": "Error",
            "confidence": 0.0,
            "error": str(e)
        }

# ORIGINAL suggestions function - PRESERVED EXACTLY
@app.post("/suggestions")
async def generate_suggestions(summary: str = Body(..., embed=True)):
    try:
        if not gemini_model:
            return "Gemini API not available"
        
        prompt = f"""
        Based on the following summary of comments, provide 4 actionable suggestions for improvement. 
        Each suggestion should be concise and relevant to the themes identified in the summary.
        
        Summary:
        {summary}
        
        Suggestions:
        1.
        2.
        3.
        4.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return f"Error generating suggestions with Gemini: {str(e)}"

# ORIGINAL generate_gemini_summary function - PRESERVED EXACTLY
def generate_gemini_summary(comments: List[str], max_length: int = 150, min_length: int = 50) -> str:
    try:
        if not gemini_model:
            return "Gemini API not available"
            
        combined_text = "\n\n".join(comments)
        
        prompt = f"""
        Please provide a comprehensive summary of the following comments in {min_length}-{max_length} words. 
        Identify the main themes, key points, and common topics across all comments. 
        Create a cohesive summary that captures the essence of the feedback:
        
        Comments:
        {combined_text}
        
        Summary (in {min_length}-{max_length} words):
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        return f"Error generating summary with Gemini: {str(e)}"

# ORIGINAL generate_wordcloud_base64 function - PRESERVED EXACTLY
def generate_wordcloud_base64(text: str) -> str:
    try:
        stopwords = set(STOPWORDS)
        stopwords.update(["said", "also", "one", "two", "like", "will", "may", "comment", "feedback"])
        
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            stopwords=stopwords,
            colormap="Blues",
            max_words=100,
            collocations=False
        ).generate(text)
        
        img = wordcloud.to_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        buf.close()
        
        return base64.b64encode(img_bytes).decode('utf-8')
        
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return ""

# ORIGINAL analyze_sentiment_distribution function - PRESERVED EXACTLY
def analyze_sentiment_distribution(sentiments: List[dict]) -> dict:
    distribution = {label: 0 for label in SENTIMENT_LABELS}
    total_confidence = 0
    
    for sentiment_result in sentiments:
        if sentiment_result["sentiment"] in distribution:
            distribution[sentiment_result["sentiment"]] += 1
            total_confidence += sentiment_result.get("confidence", 0)
    
    total_comments = len(sentiments)
    avg_confidence = total_confidence / total_comments if total_comments > 0 else 0
    
    percentages = {
        label: round((count / total_comments) * 100, 2) if total_comments > 0 else 0
        for label, count in distribution.items()
    }
    
    return {
        "counts": distribution,
        "percentages": percentages,
        "average_confidence": round(avg_confidence, 4),
        "total_comments": total_comments
    }

# NEW FUNCTION: External database connection
async def connect_to_external_database(config: ExternalDatabaseConfig) -> List[str]:
    """Connect to external user database and extract comments"""
    comments = []
    
    try:
        if config.database_type.lower() == "api":
            # Handle API endpoints
            headers = config.headers or {}
            params = config.query_params or {}
            
            response = requests.get(
                config.api_endpoint, 
                headers=headers, 
                params=params, 
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract comments from API response
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Try common field names for comment text
                        text = item.get('text') or item.get('comment') or item.get('content') or item.get('message')
                        if text:
                            comments.append(str(text))
                    elif isinstance(item, str):
                        comments.append(item)
            elif isinstance(data, dict):
                # Handle nested structures
                if 'comments' in data:
                    comments.extend([str(c) for c in data['comments']])
                elif 'data' in data:
                    for item in data['data']:
                        if isinstance(item, dict):
                            text = item.get('text') or item.get('comment') or item.get('content')
                            if text:
                                comments.append(str(text))
        
        elif config.database_type.lower() == "mongodb":
            # MongoDB connection (if user has MongoDB)
            try:
                import pymongo
                client = pymongo.MongoClient(config.connection_string)
                db_name = config.query_params.get('database', 'comments')
                collection_name = config.query_params.get('collection', 'feedback')
                
                db = client[db_name]
                collection = db[collection_name]
                
                # Find documents
                cursor = collection.find({})
                for doc in cursor:
                    text = doc.get('text') or doc.get('comment') or doc.get('content')
                    if text:
                        comments.append(str(text))
                        
                client.close()
            except ImportError:
                raise HTTPException(status_code=400, detail="pymongo not installed for MongoDB support")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported database type: {config.database_type}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to external database: {str(e)}")
    
    return comments

# ORIGINAL root function - PRESERVED AND EXTENDED
@app.get("/")
async def root():
    return {
        "message": "Comment Processing API",
        "version": "1.0",
        "endpoints": {
            "analyze": "POST /analyze - Analyze comments (expects {comments: [string]})",
            "analyze-extension": "POST /analyze-extension - Analyze browser extension data",
            "import-external": "POST /import-external - Import from external databases",
            "suggestions": "POST /suggestions - Generate suggestions from summary",
            "sources": "GET /sources - Get all available source titles",
            "records": "GET /records/{source_title} - Get records by source title",
            "health": "GET /health - Check API health status"
        }
    }

# ORIGINAL analyze function - PRESERVED EXACTLY
@app.post("/analyze")
async def analyze_comments(payload: CommentRequest):    
    if not payload.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    
    try:
        print(f"Analyzing sentiment for {len(payload.comments)} comments...")
        sentiment_results = []
        
        for i, comment in enumerate(payload.comments):
            if not comment or not comment.strip():
                sentiment_results.append({
                    "comment": comment,
                    "sentiment": "Neutral",
                    "confidence": 0.0,
                    "error": "Empty comment"
                })
                continue
                
            sentiment_result = predict_sentiment(comment.strip())
            sentiment_results.append({
                "comment": comment,
                **sentiment_result
            })
        
        # Generate Summary
        print("Generating summary...")
        summary_text = ""
        if payload.use_gemini_summary and gemini_model:
            summary_text = generate_gemini_summary(
                payload.comments, 
                payload.max_summary_length, 
                payload.min_summary_length
            )
        else:
            summary_text = f"Analysis of {len(payload.comments)} comments with various sentiments."
        
        print("Generating word cloud...")
        wordcloud_text = summary_text if summary_text and not summary_text.startswith("Error") else " ".join(payload.comments)
        wordcloud_b64 = generate_wordcloud_base64(wordcloud_text)
        
        sentiment_analysis = analyze_sentiment_distribution(sentiment_results)
        
        return {
            "Source Title": SOURCE_TITLE,
            "sentiment_analysis": {
                "individual_results": sentiment_results,
                "distribution": sentiment_analysis
            },
            "summary": {
                "text": summary_text,
                "method": "gemini" if payload.use_gemini_summary and gemini_model else "fallback",
                "word_count": len(summary_text.split()) if summary_text else 0
            },
            "wordcloud": {
                "image_base64": wordcloud_b64,
                "status": "success" if wordcloud_b64 else "failed"
            },
            "metadata": {
                "total_comments": len(payload.comments),
                "processed_comments": len([r for r in sentiment_results if not r.get("error")]),
                "processing_errors": len([r for r in sentiment_results if r.get("error")])
            }
        }
        
    except Exception as e:
        print(f"Error in analyze_comments: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# NEW ENDPOINT: Import from external database
@app.post("/import-external")
async def import_from_external_database(payload: ImportExternalDataRequest):
    """Import and analyze comments from external user databases"""
    global SOURCE_TITLE
    
    try:
        print(f"🔗 Connecting to external {payload.database_config.database_type} database...")
        
        # Connect to external database and extract comments
        comments = await connect_to_external_database(payload.database_config)
        
        if not comments:
            raise HTTPException(status_code=404, detail="No comments found in external database")
        
        print(f"📥 Imported {len(comments)} comments from external database")
        
        # Set SOURCE_TITLE for external data
        SOURCE_TITLE = f"External {payload.database_config.database_type.upper()}"
        
        # Analyze the imported comments using original analyze flow
        request = CommentRequest(
            comments=comments,
            use_gemini_summary=payload.use_gemini_summary,
            max_summary_length=payload.max_summary_length,
            min_summary_length=payload.min_summary_length
        )
        
        return await analyze_comments(request)
        
    except Exception as e:
        print(f"❌ Error importing from external database: {e}")
        raise HTTPException(status_code=500, detail=f"Error importing from external database: {str(e)}")

# ORIGINAL records function - PRESERVED WITH FIX
@app.get("/records/{source_title}")
async def get_records_by_source(source_title: str):
    try:
        if collection is None:
            raise HTTPException(status_code=503, detail="Database not connected")
            
        records = list(collection.find({"SourceTitle": source_title}))
        if not records:
            raise HTTPException(status_code=404, detail="No records found for this source title")

        # Convert ObjectIds to strings
        for record in records:
            record["_id"] = str(record["_id"])

        return jsonable_encoder(
            {"source_title": source_title, "records": records},
            custom_encoder={ObjectId: str}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching records: {e}")

# ORIGINAL sources function - PRESERVED WITH FIX
@app.get("/sources")
async def get_all_source_titles():
    try:
        if collection is None:
            raise HTTPException(status_code=503, detail="Database not connected")
            
        titles = collection.distinct("SourceTitle")
        return {"available_source_titles": titles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching source titles: {e}")

# ADD health endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_connected = False
    if mongo_client is not None and collection is not None:
        try:
            mongo_client.admin.command('ping')
            db_connected = True
        except Exception:
            db_connected = False
    
    health_status = {
        "status": "healthy",
        "models": {
            "custom_sentiment": tokenizer is not None and sentiment_model is not None,
            "fallback_pipeline": fallback_pipeline is not None,
            "gemini": gemini_model is not None
        },
        "database": {
            "connected": db_connected
        }
    }
    return health_status

# ORIGINAL startup event - PRESERVED EXACTLY
@app.on_event("startup")
async def startup_event():
    global mongo_client, db, collection
    load_models()
    try:
        mongo_client = pymongo.MongoClient(MONGO_URL)
        mongo_client.admin.command('ping')
        db = mongo_client["vtqube"]
        collection = db["sentiment_records"]
        print("✅ MongoDB connected!")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        mongo_client = None
        db = None
        collection = None

# ORIGINAL main check - PRESERVED EXACTLY
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
