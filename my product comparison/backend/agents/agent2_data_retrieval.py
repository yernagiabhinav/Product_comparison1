"""
Agent 2: Data Retrieval (UPDATED)
- Takes products from Agent 1 (each with multiple URLs)
- PASSES USER QUERY through the pipeline
- Fetches HTML content from each URL using ScraperService
- Combines and enriches product data
- Returns products ready for Agent 4 (Spec Extraction)
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from backend.services.scraper_service import ScraperService

logger = logging.getLogger(__name__)


class Agent2DataRetrieval:
    """
    Agent 2: Data Retrieval
    
    Fetches detailed product information from URLs discovered by Agent 1.
    Uses ScraperService to fetch and parse web pages.
    
    Input: Products from Agent 1 (each with multiple URLs) + user_query
    Output: Enriched products with HTML content, specs text, and metadata + user_query
    """
    
    def __init__(self, scraper_service: ScraperService = None):
        """
        Initialize Agent 2 with scraper service.
        
        Args:
            scraper_service: Optional ScraperService instance (creates new if not provided)
        """
        self.scraper = scraper_service or ScraperService()
        logger.info("✅ Agent 2 (Data Retrieval) initialized")
    
    def execute(self, agent1_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute data retrieval for all products.
        
        Args:
            agent1_result: Full result from Agent 1 containing products and user_query
            
        Returns:
            Dict with enriched products, metadata, and user_query passed through
        """
        logger.info("🔄 Agent 2: Starting data retrieval...")
        
        # Handle both dict and list input formats
        if isinstance(agent1_result, dict):
            products = agent1_result.get('products', [])
            product_type = agent1_result.get('product_type', 'unknown')
            # IMPORTANT: Pass through the user query
            user_query = agent1_result.get('user_query', '')
        elif isinstance(agent1_result, list):
            products = agent1_result
            product_type = 'unknown'
            user_query = ''
        else:
            logger.error("❌ Agent 2: Invalid input format")
            return {
                'status': 'error',
                'error': 'Invalid input format',
                'products': [],
                'user_query': ''
            }
        
        if not products:
            logger.warning("⚠️ Agent 2: No products to process")
            return {
                'status': 'no_products',
                'products': [],
                'product_type': product_type,
                'user_query': user_query
            }
        
        logger.info(f"📦 Agent 2: Processing {len(products)} products...")
        logger.info(f"📝 Agent 2: User query: '{user_query}'")
        
        enriched_products = []
        total_urls_fetched = 0
        total_urls_failed = 0
        
        for product in products:
            enriched = self._process_product(product)
            enriched_products.append(enriched)
            
            # Track stats
            total_urls_fetched += enriched.get('urls_fetched', 0)
            total_urls_failed += enriched.get('urls_failed', 0)
        
        logger.info(f"✅ Agent 2: Completed - {len(enriched_products)} products enriched")
        logger.info(f"📊 Agent 2: URLs fetched: {total_urls_fetched}, failed: {total_urls_failed}")
        
        return {
            'status': 'success',
            'products': enriched_products,
            'product_type': product_type,
            'user_query': user_query,  # PASS THROUGH
            'stats': {
                'products_processed': len(enriched_products),
                'urls_fetched': total_urls_fetched,
                'urls_failed': total_urls_failed,
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _process_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single product - fetch all its URLs.
        
        Args:
            product: Product dict from Agent 1
            
        Returns:
            Enriched product with fetched content
        """
        product_name = product.get('name', 'Unknown')
        urls = product.get('urls', [])
        
        logger.info(f"🔍 Agent 2: Processing '{product_name}' ({len(urls)} URLs)")
        
        # Start with original product data
        enriched = {
            **product,
            'fetched_data': [],
            'combined_specs_text': '',
            'combined_clean_text': '',
            'best_source': None,
            'urls_fetched': 0,
            'urls_failed': 0,
        }
        
        if not urls:
            logger.warning(f"⚠️ Agent 2: No URLs for '{product_name}'")
            return enriched
        
        # Fetch each URL
        all_specs_text = []
        all_clean_text = []
        fetched_data = []
        
        for url_info in urls:
            # Handle both dict format and string format
            if isinstance(url_info, dict):
                url = url_info.get('url', '')
            else:
                url = url_info
            
            if not url:
                continue
            
            # Fetch the page
            result = self.scraper.fetch_page(url)
            
            if result['status'] == 'success':
                enriched['urls_fetched'] += 1
                
                # Store fetched data
                fetched_data.append({
                    'url': url,
                    'site_type': result.get('site_type', 'unknown'),
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'specs_text': result.get('specs_text', ''),
                    'specs_tables': result.get('specs_tables', []),
                    'price': result.get('price', ''),
                    'html_content': result.get('html_content', ''),
                    'clean_text': result.get('clean_text', ''),
                })
                
                # Collect specs text
                if result.get('specs_text'):
                    all_specs_text.append(f"--- Source: {result.get('site_type', 'unknown')} ---")
                    all_specs_text.append(result['specs_text'])
                
                # Collect clean text
                if result.get('clean_text'):
                    all_clean_text.append(result['clean_text'][:3000])
                
                logger.info(f"   ✅ Fetched: {url[:60]}...")
            else:
                enriched['urls_failed'] += 1
                logger.warning(f"   ❌ Failed: {url[:60]}... ({result.get('error', 'unknown')})")
        
        # Combine all fetched data
        enriched['fetched_data'] = fetched_data
        enriched['combined_specs_text'] = '\n\n'.join(all_specs_text)
        enriched['combined_clean_text'] = '\n\n'.join(all_clean_text)
        
        # Determine best source (prefer GSMArena, then official sites)
        enriched['best_source'] = self._determine_best_source(fetched_data)
        
        # Extract best price found
        enriched['price'] = self._extract_best_price(fetched_data)
        
        logger.info(f"   📊 '{product_name}': {enriched['urls_fetched']} fetched, {enriched['urls_failed']} failed")
        
        return enriched
    
    def _determine_best_source(self, fetched_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine the best source from fetched data.
        Prioritizes: GSMArena > Official sites > Wikipedia > Others
        
        Args:
            fetched_data: List of fetched page data
            
        Returns:
            Best source data or None
        """
        if not fetched_data:
            return None
        
        # Priority order
        priority = {
            'gsmarena': 1,
            'apple': 2,
            'samsung': 2,
            'oneplus': 2,
            'amazon': 3,
            'bestbuy': 3,
            'flipkart': 3,
            'wikipedia': 4,
            'generic': 5,
        }
        
        # Sort by priority and specs content length
        sorted_data = sorted(
            fetched_data,
            key=lambda x: (
                priority.get(x.get('site_type', 'generic'), 5),
                -len(x.get('specs_text', ''))
            )
        )
        
        return sorted_data[0] if sorted_data else None
    
    def _extract_best_price(self, fetched_data: List[Dict[str, Any]]) -> str:
        """
        Extract the best price from fetched data.
        
        Args:
            fetched_data: List of fetched page data
            
        Returns:
            Price string or empty
        """
        for data in fetched_data:
            price = data.get('price', '')
            if price:
                return price
        return ''


# Factory function for easy creation
def create_agent2(scraper_service: ScraperService = None) -> Agent2DataRetrieval:
    """
    Factory function to create Agent 2.
    
    Args:
        scraper_service: Optional ScraperService instance
            
    Returns:
        Agent2DataRetrieval instance
    """
    return Agent2DataRetrieval(scraper_service)