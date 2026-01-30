"""
FastAPI Backend: Product Comparison API (UPDATED for GCP)
- Exposes REST endpoints for React frontend
- Orchestrates Agent 1 → Agent 2 → Agent 4 pipeline
- Securely handles GCP Vertex AI credentials via Secret Manager
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables (for local development)
load_dotenv()

# ============================================================================
# GCP CREDENTIALS CONFIGURATION
# ============================================================================
# This path matches your GCP Screenshot: /MountPath/Path1
GCP_SECRET_PATH = "/VERTEX_JSON/VERTEX_JSON"

if os.path.exists(GCP_SECRET_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_SECRET_PATH
    logging.info(f"✅ GCP Secret detected. Setting credentials to: {GCP_SECRET_PATH}")
else:
    logging.info("ℹ️ Running in local mode or GCP Secret path not found.")

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import services and agents
from backend.services.search_service import SearchService
from backend.services.llm_service import LLMService
from backend.services.scraper_service import ScraperService
from backend.agents.agent1_product_discovery import Agent1ProductDiscovery
from backend.agents.agent2_data_retrieval import Agent2DataRetrieval
from backend.agents.agent4_spec_extraction import Agent4SpecExtraction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
search_service = None
llm_service = None
scraper_service = None
agent1 = None
agent2 = None
agent4 = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global search_service, llm_service, scraper_service, agent1, agent2, agent4
    
    logger.info("🚀 Starting Product Comparison API...")
    
    # Get environment variables (Ensure these are in your GCP "Variables" tab)
    serper_key = os.getenv('SERPER_API_KEY')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
    
    # This will now use GCP_SECRET_PATH if it exists
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
    if not serper_key:
        logger.error("❌ SERPER_API_KEY not found")
        raise ValueError("SERPER_API_KEY not found in environment")
    
    if not project_id:
        logger.error("❌ GOOGLE_CLOUD_PROJECT_ID not found")
        raise ValueError("GOOGLE_CLOUD_PROJECT_ID not found in environment")
    
    # Initialize services
    search_service = SearchService(serper_key)
    llm_service = LLMService(project_id, credentials_path)
    scraper_service = ScraperService()
    
    # Initialize agents
    agent1 = Agent1ProductDiscovery(search_service, llm_service)
    agent2 = Agent2DataRetrieval(scraper_service)
    agent4 = Agent4SpecExtraction(llm_service)
    
    logger.info("✅ All services initialized successfully")
    
    yield
    
    # Cleanup on shutdown
    logger.info("👋 Shutting down Product Comparison API...")


# Create FastAPI app
app = FastAPI(
    title="Product Comparison API",
    description="AI-powered product comparison system",
    version="1.1.0",
    lifespan=lifespan
)

# Configure CORS (Updated origins for flexible deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class CompareRequest(BaseModel):
    """Request model for product comparison."""
    user_query: str
    conversation_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    service: str


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "product-comparison-api"
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Product Comparison API",
        "version": "1.1.0",
        "docs": "/docs",
        "endpoints": {
            "compare": "POST /api/compare",
            "health": "GET /health"
        }
    }


@app.post("/api/compare")
async def compare_products(request: CompareRequest):
    """
    Compare products based on user query.
    """
    logger.info(f"📝 Received comparison request: '{request.user_query}'")
    
    try:
        # AGENT 1: PRODUCT DISCOVERY
        agent1_result = agent1.execute(request.user_query)
        if agent1_result.get('status') != 'success':
            raise HTTPException(status_code=400, detail="Product discovery failed")
        
        # AGENT 2: DATA RETRIEVAL
        agent2_result = agent2.execute(agent1_result)
        
        # AGENT 4: SPEC EXTRACTION
        agent4_result = agent4.execute(agent2_result)
        
        # BUILD RESPONSE
        response = build_comparison_response(
            request.user_query,
            agent1_result,
            agent4_result
        )
        
        return response
    
    except Exception as e:
        logger.error(f"❌ Comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HELPER FUNCTIONS (KEEP THESE FROM YOUR ORIGINAL CODE)
# ============================================================================

def build_comparison_response(user_query: str, agent1_result: dict, agent4_result: dict) -> dict:
    products = agent4_result.get('products', [])
    agent1_products = {p.get('name'): p for p in agent1_result.get('products', [])}
    comparison_table = build_comparison_table(products)
    
    formatted_products = []
    for product in products:
        product_name = product.get('name', '')
        agent1_product = agent1_products.get(product_name, {})
        specs = product.get('specifications', {})
        price = specs.get('price', '') or product.get('price', '') or agent1_product.get('price', '')
        
        if price and '₹' in str(price):
            specs['price'] = price
        
        image = product.get('image', '') or agent1_product.get('image', '') or extract_product_image(product)
        
        formatted_products.append({
            'id': product.get('id', ''),
            'name': product.get('name', 'Unknown'),
            'category': product.get('category', 'general'),
            'specifications': specs,
            'price': price,
            'image': image,
            'urls': product.get('urls', agent1_product.get('urls', [])),
        })
    
    return {
        'status': 'success',
        'user_query': user_query,
        'products': formatted_products,
        'comparison_table': comparison_table,
        'timestamp': datetime.now().isoformat()
    }

def build_comparison_table(products: list) -> dict:
    if not products: return {'headers': [], 'rows': []}
    all_keys = set()
    for product in products: all_keys.update(product.get('specifications', {}).keys())
    headers = ['Specification'] + [p.get('name', 'Unknown') for p in products]
    rows = []
    for key in sorted(all_keys):
        row = {'spec': key.replace('_', ' ').title(), 'values': [str(p.get('specifications', {}).get(key, 'N/A')) for p in products]}
        rows.append(row)
    return {'headers': headers, 'rows': rows}

def extract_product_image(product: dict) -> str:
    if product.get('image'): return product['image']
    return ''

@app.get("/api/examples")
async def get_example_queries():
    return {'examples': [{'query': 'Compare iPhone 15 vs Samsung Galaxy S24'}]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8080)) # GCP default is 8080
    uvicorn.run("main:app", host="0.0.0.0", port=port)
