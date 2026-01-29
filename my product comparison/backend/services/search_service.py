"""
Search Service: Handles product searches using Serper API (IMPROVED)
- Shopping API for prices in INR (₹)
- Regular Search API for specification pages
- Source diversification to ensure multiple sources
- Category-aware search queries
"""

import http.client
import json
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SearchService:
    """
    Handles product searches using Serper APIs.
    - Shopping API: Get prices from Indian e-commerce sites (INR)
    - Search API: Get specification pages
    - Source diversification: Ensures results from multiple sources
    """
    
    # Category-specific search configurations
    CATEGORY_SEARCH_CONFIG = {
        'smartphone': {
            'spec_sites': ['gsmarena.com', '91mobiles.com', 'smartprix.com', 'gadgets360.com'],
            'shop_sites': ['amazon.in', 'flipkart.com', 'croma.com', 'vijaysales.com'],
            'search_suffix': 'specifications features',
        },
        'laptop': {
            'spec_sites': ['notebookcheck.net', 'laptopmag.com', '91mobiles.com', 'gadgets360.com'],
            'shop_sites': ['amazon.in', 'flipkart.com', 'croma.com', 'dell.com', 'hp.com'],
            'search_suffix': 'specifications features review',
        },
        'toothpaste': {
            'spec_sites': ['amazon.in', 'flipkart.com', '1mg.com', 'netmeds.com', 'bigbasket.com'],
            'shop_sites': ['amazon.in', 'flipkart.com', 'bigbasket.com', 'jiomart.com'],
            'search_suffix': 'ingredients benefits review',
        },
        'fmcg': {
            'spec_sites': ['amazon.in', 'flipkart.com', 'bigbasket.com', 'jiomart.com'],
            'shop_sites': ['amazon.in', 'flipkart.com', 'bigbasket.com', '1mg.com'],
            'search_suffix': 'review features',
        },
        'general': {
            'spec_sites': ['amazon.in', 'flipkart.com'],
            'shop_sites': ['amazon.in', 'flipkart.com', 'croma.com'],
            'search_suffix': 'review specifications',
        },
    }
    
    # Priority order for e-commerce sources (lower = higher priority)
    SOURCE_PRIORITY = {
        'amazon.in': 1,
        'amazon.com': 2,
        'flipkart.com': 3,
        'croma.com': 4,
        'vijaysales.com': 5,
        'reliancedigital.in': 6,
        'bigbasket.com': 7,
        'jiomart.com': 8,
        '1mg.com': 9,
        'netmeds.com': 10,
        'gsmarena.com': 11,
        '91mobiles.com': 12,
        'smartprix.com': 13,
        'gadgets360.com': 14,
    }
    
    def __init__(self, api_key: str):
        """
        Initialize search service.
        
        Args:
            api_key: Serper API key from environment
        """
        if not api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables")
        
        self.api_key = api_key
        self.host = "google.serper.dev"
        logger.info("✅ Search Service initialized (Shopping + Search APIs - Improved)")
    
    def search_products(
        self, 
        query: str, 
        num_results: int = 3,
        product_category: str = 'general'
    ) -> List[Dict[str, Any]]:
        """
        Search for products - combines Shopping API (prices) + Search API (specs).
        Ensures source diversification.
        
        Args:
            query: Product search query
            num_results: Number of results to return
            product_category: Category for tailored search (smartphone, laptop, toothpaste, etc.)
            
        Returns:
            List of results with prices AND spec page URLs from diverse sources
        """
        # Clean query for better results
        product_name = self._extract_product_name(query)
        
        # Detect category if not provided or is 'general'
        if product_category == 'general':
            product_category = self._detect_category(product_name)
        
        logger.info(f"🔍 Searching for: {product_name} (category: {product_category})")
        
        # Get prices from Shopping API
        shopping_results = self._search_shopping(product_name, num_results + 2)
        
        # Get spec pages from regular Search API (category-aware)
        spec_results = self._search_specs(product_name, num_results + 2, product_category)
        
        # Combine results with source diversification
        combined = self._combine_results_diversified(
            product_name, 
            shopping_results, 
            spec_results, 
            num_results
        )
        
        return combined
    
    def _detect_category(self, query: str) -> str:
        """Detect product category from query."""
        query_lower = query.lower()
        
        # Category detection keywords
        category_keywords = {
            'toothpaste': ['toothpaste', 'colgate', 'pepsodent', 'sensodyne', 'closeup', 'oral-b', 'dabur red', 'patanjali dant'],
            'smartphone': ['phone', 'iphone', 'galaxy', 'pixel', 'oneplus', 'redmi', 'realme', 'vivo', 'oppo', 'poco'],
            'laptop': ['laptop', 'macbook', 'thinkpad', 'xps', 'zenbook', 'notebook'],
            'fmcg': ['soap', 'shampoo', 'detergent', 'cream', 'lotion', 'face wash', 'body wash'],
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return category
        
        return 'general'
    
    def _search_shopping(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search Shopping API for prices.
        
        Args:
            query: Product query
            num_results: Number of results
            
        Returns:
            Shopping results with prices
        """
        try:
            conn = http.client.HTTPSConnection(self.host)
            
            payload = json.dumps({
                "q": f"{query} price India",
                "gl": "in",
                "hl": "en",
                "num": num_results
            })
            
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            conn.request("POST", "/shopping", payload, headers)
            
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            
            results = data.get('shopping', [])
            
            formatted = []
            for r in results[:num_results]:
                url = r.get('link', '')
                domain = self._extract_domain(url)
                
                formatted.append({
                    'title': r.get('title', ''),
                    'price': r.get('price', ''),
                    'source': r.get('source', domain),
                    'domain': domain,
                    'image': r.get('imageUrl', ''),
                    'rating': r.get('rating'),
                    'url': url,
                    'api_source': 'shopping',  # Track which API this came from
                })
            
            logger.info(f"   💰 Shopping API: Found {len(formatted)} results with prices")
            
            # Log sources found
            sources = set(r['domain'] for r in formatted if r.get('domain'))
            if sources:
                logger.info(f"      Sources: {', '.join(sources)}")
            
            return formatted
            
        except Exception as e:
            logger.error(f"❌ Shopping API error: {str(e)}")
            return []
        finally:
            try:
                conn.close()
            except:
                pass
    
    def _search_specs(
        self, 
        query: str, 
        num_results: int = 5,
        category: str = 'general'
    ) -> List[Dict[str, Any]]:
        """
        Search regular Search API for specification pages.
        Category-aware search queries.
        
        Args:
            query: Product query
            num_results: Number of results
            category: Product category for tailored search
            
        Returns:
            Search results with spec page URLs
        """
        try:
            conn = http.client.HTTPSConnection(self.host)
            
            # Get category-specific config
            config = self.CATEGORY_SEARCH_CONFIG.get(category, self.CATEGORY_SEARCH_CONFIG['general'])
            spec_sites = config['spec_sites']
            search_suffix = config['search_suffix']
            
            # Build search query with OR for multiple sites
            site_query = ' OR '.join([f'site:{site}' for site in spec_sites[:4]])
            search_query = f"{query} {search_suffix} ({site_query})"
            
            payload = json.dumps({
                "q": search_query,
                "gl": "in",
                "hl": "en",
                "num": num_results + 3  # Get extra to filter
            })
            
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            conn.request("POST", "/search", payload, headers)
            
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            
            results = data.get('organic', [])
            
            formatted = []
            for r in results:
                url = r.get('link', '')
                domain = self._extract_domain(url)
                
                # Calculate priority score
                priority = self.SOURCE_PRIORITY.get(domain, 99)
                
                formatted.append({
                    'title': r.get('title', ''),
                    'url': url,
                    'snippet': r.get('snippet', ''),
                    'source': domain,
                    'domain': domain,
                    'priority': priority,
                    'api_source': 'search',  # Track which API this came from
                })
            
            logger.info(f"   📋 Search API: Found {len(formatted)} spec pages")
            
            # Log sources found
            sources = set(r['domain'] for r in formatted if r.get('domain'))
            if sources:
                logger.info(f"      Sources: {', '.join(sources)}")
            
            return formatted
            
        except Exception as e:
            logger.error(f"❌ Search API error: {str(e)}")
            return []
        finally:
            try:
                conn.close()
            except:
                pass
    
    def _combine_results_diversified(
        self, 
        product_name: str, 
        shopping: List[Dict], 
        specs: List[Dict],
        num_results: int
    ) -> List[Dict[str, Any]]:
        """
        Combine shopping (prices) and search (specs) results with SOURCE DIVERSIFICATION.
        Ensures results come from multiple sources (Amazon, Flipkart, etc.)
        
        Args:
            product_name: Product name
            shopping: Shopping API results
            specs: Search API results
            num_results: Number of results to return
            
        Returns:
            Combined and diversified results
        """
        # Get best price info from shopping results
        best_price = None
        best_image = None
        best_rating = None
        price_source = None
        
        for s in shopping:
            if s.get('price') and '₹' in str(s.get('price', '')):
                if best_price is None:
                    best_price = s['price']
                    price_source = s.get('source', '')
                if s.get('image') and not best_image:
                    best_image = s['image']
                if s.get('rating') and not best_rating:
                    best_rating = s['rating']
        
        # Merge all results into a single pool
        all_results = []
        seen_urls = set()
        
        # Add spec results first (they have better snippets)
        for spec in specs:
            url = spec.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append({
                    'title': spec.get('title', ''),
                    'url': url,
                    'link': url,
                    'snippet': spec.get('snippet', ''),
                    'source': spec.get('source', ''),
                    'domain': spec.get('domain', ''),
                    'price': best_price,
                    'image': best_image,
                    'rating': best_rating,
                    'price_source': price_source,
                    'api_source': 'search',
                    'priority': spec.get('priority', 99),
                })
        
        # Add shopping results (they have direct product pages with prices)
        for shop in shopping:
            url = shop.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                domain = shop.get('domain', self._extract_domain(url))
                all_results.append({
                    'title': shop.get('title', ''),
                    'url': url,
                    'link': url,
                    'snippet': '',
                    'source': shop.get('source', domain),
                    'domain': domain,
                    'price': shop.get('price', best_price),
                    'image': shop.get('image', best_image),
                    'rating': shop.get('rating', best_rating),
                    'price_source': shop.get('source', ''),
                    'api_source': 'shopping',
                    'priority': self.SOURCE_PRIORITY.get(domain, 99),
                })
        
        # DIVERSIFICATION: Select results from different sources
        diversified = self._diversify_by_source(all_results, num_results)
        
        logger.info(f"   ✅ Combined: {len(diversified)} results (Price: {best_price})")
        
        # Log final sources
        final_sources = set(r.get('domain', '') for r in diversified)
        logger.info(f"   📦 Final sources: {', '.join(filter(None, final_sources))}")
        
        return diversified
    
    def _diversify_by_source(
        self, 
        results: List[Dict], 
        num_results: int
    ) -> List[Dict]:
        """
        Select results ensuring diversity across sources.
        
        Strategy:
        1. Group results by source domain
        2. Round-robin select from each source
        3. Prioritize Amazon > Flipkart > Others
        
        Args:
            results: All results
            num_results: Number to return
            
        Returns:
            Diversified results
        """
        if not results:
            return []
        
        # Group by domain
        by_domain: Dict[str, List[Dict]] = {}
        for r in results:
            domain = r.get('domain', 'unknown')
            # Normalize domain
            domain = self._normalize_domain(domain)
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(r)
        
        # Sort each domain's results by priority
        for domain in by_domain:
            by_domain[domain].sort(key=lambda x: x.get('priority', 99))
        
        # Define source priority order
        priority_order = ['amazon.in', 'flipkart.com', 'croma.com', 'bigbasket.com', 'jiomart.com', '1mg.com']
        
        # Sort domains by priority
        sorted_domains = sorted(
            by_domain.keys(),
            key=lambda d: next((i for i, p in enumerate(priority_order) if p in d), 99)
        )
        
        # Round-robin selection
        diversified = []
        domain_indices = {d: 0 for d in sorted_domains}
        
        # First pass: Take one from each unique domain (prioritized)
        for domain in sorted_domains:
            if len(diversified) >= num_results:
                break
            if by_domain[domain]:
                diversified.append(by_domain[domain][0])
                domain_indices[domain] = 1
        
        # Second pass: Fill remaining slots
        round_robin_idx = 0
        while len(diversified) < num_results:
            domain = sorted_domains[round_robin_idx % len(sorted_domains)]
            idx = domain_indices[domain]
            
            if idx < len(by_domain[domain]):
                result = by_domain[domain][idx]
                if result not in diversified:
                    diversified.append(result)
                domain_indices[domain] += 1
            
            round_robin_idx += 1
            
            # Prevent infinite loop
            if round_robin_idx > len(results) + num_results:
                break
        
        return diversified[:num_results]
    
    def _normalize_domain(self, domain: str) -> str:
        """Normalize domain name."""
        if not domain:
            return 'unknown'
        
        domain = domain.lower().strip()
        domain = domain.replace('www.', '')
        
        # Normalize common variations
        if 'amazon' in domain:
            return 'amazon.in'
        if 'flipkart' in domain:
            return 'flipkart.com'
        
        return domain
    
    def _extract_product_name(self, query: str) -> str:
        """Extract clean product name from query."""
        import re
        
        # Remove common suffixes
        clean = re.sub(r'\s*(price|specifications?|specs?|features?|general|smartphone|laptop)\s*$', '', query, flags=re.IGNORECASE)
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        return clean
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            domain = urlparse(url).netloc
            return domain.replace('www.', '').lower()
        except:
            return 'unknown'


# For testing
if __name__ == "__main__":
    import os
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    api_key = os.getenv('SERPER_API_KEY', 'test_key')
    
    if api_key != 'test_key':
        service = SearchService(api_key)
        
        # Test smartphone
        print("\n=== Testing Smartphone ===")
        results = service.search_products("iPhone 15 Pro", num_results=3, product_category='smartphone')
        for r in results:
            print(f"  - {r['domain']}: {r['title'][:50]}...")
        
        # Test toothpaste
        print("\n=== Testing Toothpaste ===")
        results = service.search_products("Colgate", num_results=3, product_category='toothpaste')
        for r in results:
            print(f"  - {r['domain']}: {r['title'][:50]}...")
    else:
        print("Set SERPER_API_KEY to test")