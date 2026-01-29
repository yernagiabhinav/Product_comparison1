"""
Agent 1: Product Discovery (UPDATED)
- Uses LLM to identify product names and types
- Gets ₹ prices from Shopping API
- Gets spec URLs from Search API
- Properly passes price data to downstream agents
"""

from typing import List, Dict, Any, Optional
import re
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Agent1ProductDiscovery:
    """
    Discovers products from user query using LLM.
    Gets prices (₹) and spec URLs for each product.
    """
    
    # Category detection keywords
    CATEGORY_KEYWORDS = {
        'smartphone': ['phone', 'mobile', 'iphone', 'galaxy', 'pixel', 'oneplus', 'xiaomi', 'redmi', 'poco', 'oppo', 'vivo', 'realme', 'nothing', 'motorola', 'nokia', 'samsung'],
        'laptop': ['laptop', 'macbook', 'thinkpad', 'xps', 'zenbook', 'notebook', 'chromebook', 'surface', 'pavilion', 'inspiron', 'legion', 'rog'],
        'tablet': ['ipad', 'tab', 'tablet'],
        'headphones': ['airpods', 'headphone', 'earbuds', 'earphone', 'buds', 'headset'],
        'smartwatch': ['watch', 'fitbit', 'garmin', 'band'],
        'tv': ['tv', 'television', 'oled', 'qled'],
    }
    
    def __init__(self, search_service, llm_service):
        """Initialize Agent 1."""
        self.search_service = search_service
        self.llm_service = llm_service
        logger.info("✅ Agent 1 (Product Discovery - Updated) initialized")
    
    def execute(self, user_query: str) -> Dict[str, Any]:
        """
        Execute product discovery.
        
        Args:
            user_query: User's query (e.g., "compare vivo y73 vs realme 8 pro")
            
        Returns:
            Dict with products, types, URLs, and PRICES
        """
        logger.info(f"🔍 Agent 1: Processing '{user_query}'")
        
        try:
            # Step 1: Extract product names using LLM
            extraction = self._extract_products_llm(user_query)
            
            product_names = extraction.get('product_names', [])
            product_type = extraction.get('product_type', 'general')
            
            if not product_names:
                # Fallback to regex
                extraction = self._extract_products_regex(user_query)
                product_names = extraction.get('product_names', [])
                product_type = extraction.get('product_type', 'general')
            
            logger.info(f"📦 Found {len(product_names)} products: {product_names}")
            logger.info(f"📱 Product type: {product_type}")
            
            if not product_names:
                return {
                    'status': 'no_products',
                    'products': [],
                    'product_type': 'unknown',
                    'user_query': user_query,
                    'message': 'No products found in query'
                }
            
            # Step 2: Search for each product
            products = []
            for idx, name in enumerate(product_names):
                product = self._search_product(name, product_type, idx)
                products.append(product)
            
            return {
                'status': 'success',
                'products': products,
                'product_type': product_type,
                'user_query': user_query,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"❌ Agent 1 error: {str(e)}")
            return {
                'status': 'error',
                'products': [],
                'error': str(e),
                'user_query': user_query
            }
    
    def _extract_products_llm(self, query: str) -> Dict[str, Any]:
        """Extract products using LLM."""
        try:
            prompt = f"""Extract product names from this comparison query. Return ONLY JSON.

Query: "{query}"

Return this exact format:
{{"product_names": ["Product 1", "Product 2"], "product_type": "smartphone"}}

Product types: smartphone, laptop, tablet, headphones, smartwatch, tv, general

Examples:
- "compare iPhone 15 vs Samsung S24" → {{"product_names": ["iPhone 15", "Samsung Galaxy S24"], "product_type": "smartphone"}}
- "vivo y73 vs realme 8 pro" → {{"product_names": ["Vivo Y73", "Realme 8 Pro"], "product_type": "smartphone"}}

Return ONLY JSON, no other text."""

            response = self.llm_service.extract_products(prompt)
            
            # Parse JSON
            json_match = re.search(r'\{[\s\S]*?\}', response)
            if json_match:
                result = json.loads(json_match.group())
                if result.get('product_names'):
                    return {
                        'product_names': result['product_names'],
                        'product_type': result.get('product_type', 'smartphone')
                    }
            
            return {}
        
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return {}
    
    def _extract_products_regex(self, query: str) -> Dict[str, Any]:
        """Fallback regex extraction."""
        # Split by comparison keywords
        text = re.sub(r'\s+(vs\.?|versus|or|and)\s+', ' ||| ', query, flags=re.IGNORECASE)
        parts = text.split(' ||| ')
        
        products = []
        for part in parts:
            # Clean up
            cleaned = re.sub(r'\b(compare|which|best|better|for|gaming|my|is|the)\b', '', part, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            if len(cleaned) >= 3:
                products.append(cleaned)
        
        # Detect product type
        query_lower = query.lower()
        product_type = 'general'
        for ptype, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                product_type = ptype
                break
        
        return {
            'product_names': products,
            'product_type': product_type
        }
    
    def _search_product(self, name: str, product_type: str, idx: int) -> Dict[str, Any]:
        """
        Search for a single product.
        Gets prices from Shopping API and spec URLs from Search API.
        """
        logger.info(f"🔎 Searching for '{name}'...")
        
        try:
            # Call search service (combines Shopping + Search APIs)
            search_query = f"{name} {product_type}"
            results = self.search_service.search_products(search_query, num_results=3)
            
            # Extract data
            urls = []
            best_price = None
            best_image = None
            best_rating = None
            all_prices = []
            
            for r in results:
                url_info = {
                    'url': r.get('url') or r.get('link', ''),
                    'title': r.get('title', ''),
                    'snippet': r.get('snippet', ''),
                    'source': r.get('source', ''),
                    'price': r.get('price', ''),
                    'image': r.get('image', ''),
                    'rating': r.get('rating'),
                }
                urls.append(url_info)
                
                # Capture price
                if r.get('price') and '₹' in str(r.get('price', '')):
                    if not best_price:
                        best_price = r['price']
                    all_prices.append({
                        'price': r['price'],
                        'source': r.get('source', r.get('price_source', ''))
                    })
                
                # Capture image
                if not best_image and r.get('image'):
                    best_image = r['image']
                
                # Capture rating
                if not best_rating and r.get('rating'):
                    best_rating = r['rating']
            
            product = {
                'id': f'product_{idx + 1}',
                'name': name,
                'category': product_type,
                'urls': urls,
                'price': best_price,           # ₹ price from Shopping API
                'all_prices': all_prices,      # All prices found
                'image': best_image,
                'rating': best_rating,
            }
            
            logger.info(f"   ✅ Found {len(urls)} URLs | Price: {best_price}")
            return product
        
        except Exception as e:
            logger.error(f"   ❌ Search failed: {str(e)}")
            return {
                'id': f'product_{idx + 1}',
                'name': name,
                'category': product_type,
                'urls': [],
                'price': None,
                'error': str(e)
            }


def create_agent1(search_service, llm_service):
    """Factory function."""
    return Agent1ProductDiscovery(search_service, llm_service)
