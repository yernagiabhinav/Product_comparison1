"""
FastAPI Backend: Product Comparison API (UPDATED)
- Exposes REST endpoints for React frontend
- Orchestrates Agent 1 → Agent 2 → Agent 4 pipeline
- Better handling of ₹ prices from Shopping API
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

# Load environment variables
load_dotenv()

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
    
    # Get environment variables
    serper_key = os.getenv('SERPER_API_KEY')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
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

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # React dev server
        "http://localhost:5173",      # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"                           # Allow all (for development)
    ],
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
    
    Pipeline:
    1. Agent 1: Product Discovery (extract products, get prices)
    2. Agent 2: Data Retrieval (fetch product data from web)
    3. Agent 4: Spec Extraction (extract specs, generate recommendations)
    """
    logger.info(f"📝 Received comparison request: '{request.user_query}'")
    
    try:
        # ==================== AGENT 1: PRODUCT DISCOVERY ====================
        logger.info("🔍 Running Agent 1: Product Discovery...")
        
        agent1_result = agent1.execute(request.user_query)
        
        if agent1_result.get('status') != 'success':
            logger.warning(f"⚠️ Agent 1 failed: {agent1_result.get('message', 'Unknown error')}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Product discovery failed",
                    "message": agent1_result.get('message', 'Could not find products in query'),
                    "suggestion": "Try a query like: 'Compare iPhone 15 vs Samsung Galaxy S24'"
                }
            )
        
        products_found = len(agent1_result.get('products', []))
        logger.info(f"✅ Agent 1 complete: Found {products_found} products")
        
        # Log prices found
        for p in agent1_result.get('products', []):
            logger.info(f"   💰 {p.get('name')}: {p.get('price', 'No price')}")
        
        if products_found < 2:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Not enough products",
                    "message": f"Found only {products_found} product(s). Need at least 2 to compare.",
                    "suggestion": "Try: 'Compare Product A vs Product B'"
                }
            )
        
        # ==================== AGENT 2: DATA RETRIEVAL ====================
        logger.info("📥 Running Agent 2: Data Retrieval...")
        
        agent2_result = agent2.execute(agent1_result)
        
        if agent2_result.get('status') != 'success':
            logger.warning(f"⚠️ Agent 2 failed: {agent2_result.get('error', 'Unknown error')}")
            # Continue anyway - Agent 4 will use LLM knowledge
        
        urls_fetched = agent2_result.get('stats', {}).get('urls_fetched', 0)
        logger.info(f"✅ Agent 2 complete: Fetched {urls_fetched} URLs")
        
        # ==================== AGENT 4: SPEC EXTRACTION ====================
        logger.info("🔬 Running Agent 4: Specification Extraction...")
        
        agent4_result = agent4.execute(agent2_result)
        
        if agent4_result.get('status') != 'success':
            logger.warning(f"⚠️ Agent 4 failed: {agent4_result.get('error', 'Unknown error')}")
        
        logger.info("✅ Agent 4 complete")
        
        # ==================== BUILD RESPONSE ====================
        response = build_comparison_response(
            request.user_query,
            agent1_result,
            agent4_result
        )
        
        logger.info(f"✅ Comparison complete for: '{request.user_query}'")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Comparison error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Comparison failed",
                "message": str(e)
            }
        )


def build_comparison_response(user_query: str, agent1_result: dict, agent4_result: dict) -> dict:
    """
    Build the final comparison response for frontend.
    """
    products = agent4_result.get('products', [])
    agent1_products = {p.get('name'): p for p in agent1_result.get('products', [])}
    
    # Build comparison table
    comparison_table = build_comparison_table(products)
    
    # Format products for frontend
    formatted_products = []
    for product in products:
        # Get price from Agent 1 (Shopping API) if not in specs
        product_name = product.get('name', '')
        agent1_product = agent1_products.get(product_name, {})
        
        # Priority: specs.price > product.price > agent1.price
        specs = product.get('specifications', {})
        price = specs.get('price', '')
        if not price or price in ['N/A', '', '$,']:
            price = product.get('price', '')
        if not price or price in ['N/A', '', '$,']:
            price = agent1_product.get('price', '')
        
        # Ensure price is in specifications
        if price and '₹' in str(price):
            specs['price'] = price
        
        # Get image
        image = product.get('image', '')
        if not image:
            image = agent1_product.get('image', '')
        if not image:
            image = extract_product_image(product)
        
        formatted_products.append({
            'id': product.get('id', ''),
            'name': product.get('name', 'Unknown'),
            'category': product.get('category', 'general'),
            'specifications': specs,
            'extraction_status': product.get('extraction_status', 'unknown'),
            'price': price,
            'image': image,
            'urls': product.get('urls', agent1_product.get('urls', [])),
        })
    
    return {
        'status': 'success',
        'user_query': user_query,
        'product_type': agent4_result.get('product_type', 'general'),
        'user_preferences': agent4_result.get('user_preferences', []),
        'products': formatted_products,
        'comparison_table': comparison_table,
        'recommendation_summary': agent4_result.get('recommendation_summary', {}),
        'stats': {
            'products_compared': len(products),
            'specs_extracted': sum(
                1 for p in products 
                if p.get('extraction_status') == 'success'
            ),
        },
        'timestamp': datetime.now().isoformat()
    }


def build_comparison_table(products: list) -> dict:
    """Build a comparison table from products."""
    if not products:
        return {'headers': [], 'rows': []}
    
    # Display name mappings
    display_names = {
        'brand': 'Brand',
        'model': 'Model',
        'product_name': 'Product Name',
        'display_size': 'Display Size',
        'processor': 'Processor',
        'ram': 'RAM',
        'storage': 'Storage',
        'battery': 'Battery',
        'camera_main': 'Main Camera',
        'camera_front': 'Front Camera',
        'os': 'Operating System',
        'weight': 'Weight',
        'price': 'Price',
        'refresh_rate': 'Refresh Rate',
        '5g_support': '5G Support',
    }
    
    # Get all specification keys
    all_keys = set()
    for product in products:
        specs = product.get('specifications', {})
        all_keys.update(specs.keys())
    
    # Build headers (product names)
    headers = ['Specification'] + [p.get('name', 'Unknown') for p in products]
    
    # Build rows
    rows = []
    for key in sorted(all_keys):
        display_key = display_names.get(key, key.replace('_', ' ').title())
        row = {
            'spec': display_key,
            'key': key,
            'values': []
        }
        
        for product in products:
            specs = product.get('specifications', {})
            value = specs.get(key, 'N/A')
            row['values'].append(str(value))
        
        rows.append(row)
    
    return {
        'headers': headers,
        'rows': rows,
        'product_names': [p.get('name', 'Unknown') for p in products]
    }


def extract_product_image(product: dict) -> str:
    """Extract product image URL if available."""
    # Check direct image field
    if product.get('image'):
        return product['image']
    
    # Check URLs for image
    urls = product.get('urls', [])
    for url_info in urls:
        if isinstance(url_info, dict) and url_info.get('image'):
            return url_info['image']
    
    # Check fetched data
    fetched_data = product.get('fetched_data', [])
    for data in fetched_data:
        if data.get('image'):
            return data['image']
    
    return ''


# ============================================================================
# ADDITIONAL ENDPOINTS
# ============================================================================

@app.get("/api/examples")
async def get_example_queries():
    """Get example comparison queries."""
    return {
        'examples': [
            {
                'query': 'Compare iPhone 15 vs Samsung Galaxy S24',
                'category': 'smartphone'
            },
            {
                'query': 'Compare Vivo Y73 vs Realme 8 Pro for gaming',
                'category': 'smartphone'
            },
            {
                'query': 'MacBook Pro vs Dell XPS 15 for video editing',
                'category': 'laptop'
            },
            {
                'query': 'Compare OnePlus 12 vs iPhone 15 Pro',
                'category': 'smartphone'
            },
            {
                'query': 'Samsung Galaxy S24 Ultra vs Google Pixel 8 Pro',
                'category': 'smartphone'
            }
        ]
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv('PORT', 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
