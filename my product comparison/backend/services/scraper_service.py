"""
Scraper Service: Fetch and parse web pages (IMPROVED v2)
- Uses cloudscraper to bypass Cloudflare protection
- Rotates User-Agent strings
- Realistic browser headers
- Random delays between requests
- Better error handling with actual HTTP status codes
- Site-specific extractors for GSMArena, Amazon, official sites
"""

import re
import time
import random
import logging
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

import requests
import cloudscraper
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ScraperService:
    """
    Service for fetching and parsing web pages.
    Uses cloudscraper + BeautifulSoup for scraping protected sites.
    """
    
    # Rotating User-Agent strings (realistic browsers)
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
    ]
    
    # Timeout settings
    DEFAULT_TIMEOUT = 20
    MAX_RETRIES = 3
    MIN_DELAY = 0.5  # seconds
    MAX_DELAY = 2.0  # seconds
    
    def __init__(self):
        """Initialize the scraper service with cloudscraper."""
        # Create cloudscraper session (bypasses Cloudflare)
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True,
            },
            delay=10  # Delay for JS challenges
        )
        
        # Also keep a regular requests session for sites that don't need cloudscraper
        self.session = requests.Session()
        
        logger.info("✅ ScraperService initialized with cloudscraper")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get realistic browser headers with rotating User-Agent."""
        return {
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
    
    def _random_delay(self):
        """Add random delay between requests."""
        delay = random.uniform(self.MIN_DELAY, self.MAX_DELAY)
        time.sleep(delay)
    
    def fetch_page(self, url: str) -> Dict[str, Any]:
        """
        Fetch a web page and return parsed content.
        Automatically uses cloudscraper for protected sites.
        
        Args:
            url: URL to fetch
            
        Returns:
            Dict with status, html, text, and extracted data
        """
        logger.info(f"🌐 Fetching: {url}")
        
        # Determine if site needs cloudscraper
        domain = urlparse(url).netloc.lower()
        needs_cloudscraper = self._needs_cloudscraper(domain)
        
        for attempt in range(self.MAX_RETRIES):
            try:
                # Add random delay
                if attempt > 0:
                    self._random_delay()
                
                headers = self._get_headers()
                
                # Choose scraper based on site
                if needs_cloudscraper:
                    logger.debug(f"Using cloudscraper for {domain}")
                    response = self.scraper.get(
                        url,
                        headers=headers,
                        timeout=self.DEFAULT_TIMEOUT,
                        allow_redirects=True
                    )
                else:
                    logger.debug(f"Using regular requests for {domain}")
                    response = self.session.get(
                        url,
                        headers=headers,
                        timeout=self.DEFAULT_TIMEOUT,
                        allow_redirects=True
                    )
                
                response.raise_for_status()
                
                # Check if we got blocked anyway
                if self._is_blocked_response(response):
                    logger.warning(f"⚠️ Blocked response detected, retrying with cloudscraper...")
                    needs_cloudscraper = True
                    continue
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract content based on site type
                site_type = self._detect_site_type(url)
                extracted = self._extract_content(soup, site_type, url)
                
                logger.info(f"✅ Successfully fetched: {url} (status: {response.status_code})")
                
                return {
                    'status': 'success',
                    'url': url,
                    'site_type': site_type,
                    'http_status': response.status_code,
                    'html_content': response.text[:50000],  # Limit HTML size
                    'title': extracted.get('title', ''),
                    'description': extracted.get('description', ''),
                    'specs_text': extracted.get('specs_text', ''),
                    'specs_tables': extracted.get('specs_tables', []),
                    'price': extracted.get('price', ''),
                    'clean_text': extracted.get('clean_text', ''),
                }
            
            except requests.exceptions.Timeout:
                logger.warning(f"⏱️ Timeout on attempt {attempt + 1}/{self.MAX_RETRIES} for {url}")
                if attempt < self.MAX_RETRIES - 1:
                    self._random_delay()
                    continue
                return self._error_response(url, "Request timeout", 408)
            
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else 0
                logger.warning(f"❌ HTTP {status_code} for {url}")
                
                # Retry on certain status codes
                if status_code in [403, 429, 503] and attempt < self.MAX_RETRIES - 1:
                    logger.info(f"   Retrying with cloudscraper...")
                    needs_cloudscraper = True
                    self._random_delay()
                    continue
                
                return self._error_response(url, f"HTTP error: {status_code}", status_code)
            
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"🔌 Connection error for {url}: {str(e)[:50]}")
                if attempt < self.MAX_RETRIES - 1:
                    self._random_delay()
                    continue
                return self._error_response(url, "Connection error", 0)
            
            except cloudscraper.exceptions.CloudflareChallengeError as e:
                logger.warning(f"🛡️ Cloudflare challenge failed for {url}")
                if attempt < self.MAX_RETRIES - 1:
                    self._random_delay()
                    continue
                return self._error_response(url, "Cloudflare protection", 403)
            
            except Exception as e:
                logger.error(f"❌ Unexpected error for {url}: {str(e)}")
                return self._error_response(url, str(e), 0)
        
        return self._error_response(url, "Max retries exceeded", 0)
    
    def _needs_cloudscraper(self, domain: str) -> bool:
        """Determine if a domain needs cloudscraper."""
        # Known protected sites
        protected_domains = [
            'amazon.com',
            'walmart.com',
            'target.com',
            'bestbuy.com',
            'unilever.com',
            'pepsodent.com',
            'close-up.com',
            'colgate.com',
            'samsung.com',
            'apple.com',
            'flipkart.com',
            'ebay.com',
            'newegg.com',
        ]
        
        for protected in protected_domains:
            if protected in domain:
                return True
        
        return False
    
    def _is_blocked_response(self, response) -> bool:
        """Check if response indicates blocking."""
        # Check for common block indicators
        text_lower = response.text[:5000].lower()
        
        block_indicators = [
            'access denied',
            'blocked',
            'captcha',
            'robot',
            'unusual traffic',
            'verify you are human',
            'please enable javascript',
            'challenge-running',
        ]
        
        for indicator in block_indicators:
            if indicator in text_lower:
                return True
        
        # Check for very short responses (might be block page)
        if len(response.text) < 500 and response.status_code == 200:
            return True
        
        return False
    
    def _error_response(self, url: str, error: str, status_code: int = 0) -> Dict[str, Any]:
        """Return a standardized error response."""
        return {
            'status': 'error',
            'url': url,
            'error': error,
            'http_status': status_code,
            'html_content': '',
            'title': '',
            'description': '',
            'specs_text': '',
            'specs_tables': [],
            'price': '',
            'clean_text': '',
        }
    
    def _detect_site_type(self, url: str) -> str:
        """
        Detect the type of website for specialized parsing.
        
        Args:
            url: URL to analyze
            
        Returns:
            Site type string
        """
        url_lower = url.lower()
        domain = urlparse(url).netloc.lower()
        
        # Tech specs sites
        if 'gsmarena.com' in domain:
            return 'gsmarena'
        if 'amazon' in domain:
            return 'amazon'
        if 'apple.com' in domain:
            return 'apple'
        if 'samsung.com' in domain:
            return 'samsung'
        if 'oneplus.com' in domain:
            return 'oneplus'
        if 'wikipedia.org' in domain:
            return 'wikipedia'
        if 'bestbuy.com' in domain:
            return 'bestbuy'
        if 'flipkart.com' in domain:
            return 'flipkart'
        if 'walmart.com' in domain:
            return 'walmart'
        if 'target.com' in domain:
            return 'target'
        if 'rtings.com' in domain:
            return 'rtings'
        
        # Generic
        return 'generic'
    
    def _extract_content(self, soup: BeautifulSoup, site_type: str, url: str) -> Dict[str, Any]:
        """
        Extract content from parsed HTML based on site type.
        
        Args:
            soup: BeautifulSoup object
            site_type: Type of website
            url: Original URL
            
        Returns:
            Dict with extracted content
        """
        # Get basic info first
        title = self._extract_title(soup)
        description = self._extract_description(soup)
        price = self._extract_price(soup, site_type)
        
        # Site-specific extraction
        if site_type == 'gsmarena':
            specs_text, specs_tables = self._extract_gsmarena_specs(soup)
        elif site_type == 'amazon':
            specs_text, specs_tables = self._extract_amazon_specs(soup)
        elif site_type == 'walmart':
            specs_text, specs_tables = self._extract_walmart_specs(soup)
        elif site_type == 'target':
            specs_text, specs_tables = self._extract_target_specs(soup)
        elif site_type == 'apple':
            specs_text, specs_tables = self._extract_apple_specs(soup)
        elif site_type == 'samsung':
            specs_text, specs_tables = self._extract_samsung_specs(soup)
        elif site_type == 'wikipedia':
            specs_text, specs_tables = self._extract_wikipedia_specs(soup)
        elif site_type == 'rtings':
            specs_text, specs_tables = self._extract_rtings_specs(soup)
        else:
            specs_text, specs_tables = self._extract_generic_specs(soup)
        
        # Get clean text from entire page
        clean_text = self._extract_clean_text(soup)
        
        return {
            'title': title,
            'description': description,
            'price': price,
            'specs_text': specs_text,
            'specs_tables': specs_tables,
            'clean_text': clean_text,
        }
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try og:title first
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()
        
        # Try regular title
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        # Try h1
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        
        return ''
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page description."""
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content'].strip()
        
        # Try og:description
        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            return og_desc['content'].strip()
        
        # Try first paragraph
        first_p = soup.find('p')
        if first_p:
            text = first_p.get_text(strip=True)
            return text[:500] if len(text) > 500 else text
        
        return ''
    
    def _extract_price(self, soup: BeautifulSoup, site_type: str) -> str:
        """Extract price from page."""
        price_patterns = []
        
        if site_type == 'amazon':
            price_patterns = [
                soup.find('span', class_='a-price-whole'),
                soup.find('span', id='priceblock_ourprice'),
                soup.find('span', class_='a-offscreen'),
                soup.find('span', class_='a-price'),
            ]
        elif site_type == 'walmart':
            price_patterns = [
                soup.find('span', {'itemprop': 'price'}),
                soup.find('span', class_=re.compile(r'price', re.I)),
            ]
        elif site_type == 'target':
            price_patterns = [
                soup.find('span', {'data-test': 'product-price'}),
                soup.find('span', class_=re.compile(r'price', re.I)),
            ]
        elif site_type == 'bestbuy':
            price_patterns = [
                soup.find('div', class_='priceView-customer-price'),
                soup.find('span', class_='sr-only', string=re.compile(r'\$')),
            ]
        else:
            # Generic price patterns
            price_patterns = [
                soup.find('span', class_=re.compile(r'price', re.I)),
                soup.find('div', class_=re.compile(r'price', re.I)),
                soup.find('span', {'itemprop': 'price'}),
            ]
        
        for elem in price_patterns:
            if elem:
                if hasattr(elem, 'get_text'):
                    text = elem.get_text(strip=True)
                else:
                    text = str(elem).strip()
                
                # Extract price with regex
                price_match = re.search(r'[\$€£₹][\d,]+\.?\d*', text)
                if price_match:
                    return price_match.group()
        
        # Fallback: search entire page for price
        text = soup.get_text()
        price_match = re.search(r'\$[\d,]+\.?\d*', text)
        if price_match:
            return price_match.group()
        
        return ''
    
    def _extract_gsmarena_specs(self, soup: BeautifulSoup) -> tuple:
        """Extract specifications from GSMArena."""
        specs_text = []
        specs_tables = []
        
        # Find specs tables
        spec_tables = soup.find_all('table', class_='specs-brief') or soup.find_all('table')
        
        for table in spec_tables:
            table_data = []
            rows = table.find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    if key and value:
                        table_data.append({'key': key, 'value': value})
                        specs_text.append(f"{key}: {value}")
            
            if table_data:
                specs_tables.append(table_data)
        
        # Also get spec list items
        spec_items = soup.find_all('li', class_='specs-brief-accent')
        for item in spec_items:
            text = item.get_text(strip=True)
            if text:
                specs_text.append(text)
        
        # Get nfo divs (GSMArena specific)
        nfo_divs = soup.find_all('div', class_='nfo')
        for nfo in nfo_divs:
            text = nfo.get_text(strip=True)
            if text:
                specs_text.append(text)
        
        return '\n'.join(specs_text), specs_tables
    
    def _extract_amazon_specs(self, soup: BeautifulSoup) -> tuple:
        """Extract specifications from Amazon."""
        specs_text = []
        specs_tables = []
        
        # Product details table
        detail_table = soup.find('table', id='productDetails_detailBullets_sections1')
        if detail_table:
            rows = detail_table.find_all('tr')
            table_data = []
            for row in rows:
                th = row.find('th')
                td = row.find('td')
                if th and td:
                    key = th.get_text(strip=True)
                    value = td.get_text(strip=True)
                    table_data.append({'key': key, 'value': value})
                    specs_text.append(f"{key}: {value}")
            if table_data:
                specs_tables.append(table_data)
        
        # Technical details
        tech_table = soup.find('table', id='productDetails_techSpec_section_1')
        if tech_table:
            rows = tech_table.find_all('tr')
            table_data = []
            for row in rows:
                th = row.find('th')
                td = row.find('td')
                if th and td:
                    key = th.get_text(strip=True)
                    value = td.get_text(strip=True)
                    table_data.append({'key': key, 'value': value})
                    specs_text.append(f"{key}: {value}")
            if table_data:
                specs_tables.append(table_data)
        
        # Feature bullets
        bullets = soup.find('div', id='feature-bullets')
        if bullets:
            items = bullets.find_all('li')
            for item in items:
                text = item.get_text(strip=True)
                if text:
                    specs_text.append(text)
        
        # Product description
        desc = soup.find('div', id='productDescription')
        if desc:
            text = desc.get_text(strip=True)
            if text:
                specs_text.append(text[:1000])
        
        return '\n'.join(specs_text), specs_tables
    
    def _extract_walmart_specs(self, soup: BeautifulSoup) -> tuple:
        """Extract specifications from Walmart."""
        specs_text = []
        specs_tables = []
        
        # Product highlights
        highlights = soup.find('div', {'data-testid': 'product-highlights'})
        if highlights:
            items = highlights.find_all('li')
            for item in items:
                text = item.get_text(strip=True)
                if text:
                    specs_text.append(text)
        
        # Specifications section
        specs_section = soup.find('div', {'data-testid': 'specifications'})
        if specs_section:
            rows = specs_section.find_all(['tr', 'div'])
            table_data = []
            for row in rows:
                text = row.get_text(separator=': ', strip=True)
                if text:
                    specs_text.append(text)
        
        # About this item
        about = soup.find('div', class_=re.compile(r'about', re.I))
        if about:
            text = about.get_text(strip=True)
            if text:
                specs_text.append(text[:1000])
        
        return '\n'.join(specs_text), specs_tables
    
    def _extract_target_specs(self, soup: BeautifulSoup) -> tuple:
        """Extract specifications from Target."""
        specs_text = []
        specs_tables = []
        
        # Highlights
        highlights = soup.find('div', {'data-test': 'item-highlights'})
        if highlights:
            items = highlights.find_all('li')
            for item in items:
                text = item.get_text(strip=True)
                if text:
                    specs_text.append(text)
        
        # Specifications
        specs_div = soup.find('div', {'data-test': 'item-details-specifications'})
        if specs_div:
            text = specs_div.get_text(separator='\n', strip=True)
            if text:
                specs_text.append(text)
        
        # Description
        desc = soup.find('div', {'data-test': 'item-details-description'})
        if desc:
            text = desc.get_text(strip=True)
            if text:
                specs_text.append(text[:1000])
        
        return '\n'.join(specs_text), specs_tables
    
    def _extract_apple_specs(self, soup: BeautifulSoup) -> tuple:
        """Extract specifications from Apple."""
        specs_text = []
        specs_tables = []
        
        # Tech specs sections
        spec_sections = soup.find_all('div', class_=re.compile(r'techspecs|specs', re.I))
        
        for section in spec_sections:
            # Get all text
            text = section.get_text(separator='\n', strip=True)
            specs_text.append(text)
            
            # Try to find tables
            tables = section.find_all('table')
            for table in tables:
                table_data = self._parse_table(table)
                if table_data:
                    specs_tables.append(table_data)
        
        # Also look for dl (definition lists)
        dls = soup.find_all('dl')
        for dl in dls:
            dts = dl.find_all('dt')
            dds = dl.find_all('dd')
            table_data = []
            for dt, dd in zip(dts, dds):
                key = dt.get_text(strip=True)
                value = dd.get_text(strip=True)
                if key and value:
                    table_data.append({'key': key, 'value': value})
                    specs_text.append(f"{key}: {value}")
            if table_data:
                specs_tables.append(table_data)
        
        return '\n'.join(specs_text), specs_tables
    
    def _extract_samsung_specs(self, soup: BeautifulSoup) -> tuple:
        """Extract specifications from Samsung."""
        specs_text = []
        specs_tables = []
        
        # Spec sections
        spec_sections = soup.find_all('div', class_=re.compile(r'spec|feature', re.I))
        
        for section in spec_sections:
            text = section.get_text(separator='\n', strip=True)
            if text and len(text) > 10:
                specs_text.append(text)
        
        # Tables
        tables = soup.find_all('table')
        for table in tables:
            table_data = self._parse_table(table)
            if table_data:
                specs_tables.append(table_data)
        
        return '\n'.join(specs_text), specs_tables
    
    def _extract_wikipedia_specs(self, soup: BeautifulSoup) -> tuple:
        """Extract specifications from Wikipedia."""
        specs_text = []
        specs_tables = []
        
        # Infobox (common in Wikipedia)
        infobox = soup.find('table', class_=re.compile(r'infobox', re.I))
        if infobox:
            table_data = self._parse_table(infobox)
            if table_data:
                specs_tables.append(table_data)
                for item in table_data:
                    specs_text.append(f"{item['key']}: {item['value']}")
        
        # Also get the first few paragraphs
        content_div = soup.find('div', class_='mw-parser-output')
        if content_div:
            paragraphs = content_div.find_all('p', limit=5)
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text and len(text) > 50:
                    specs_text.append(text)
        
        return '\n'.join(specs_text), specs_tables
    
    def _extract_rtings_specs(self, soup: BeautifulSoup) -> tuple:
        """Extract specifications from Rtings.com."""
        specs_text = []
        specs_tables = []
        
        # Test results
        test_results = soup.find_all('div', class_=re.compile(r'test-result|score', re.I))
        for result in test_results:
            text = result.get_text(strip=True)
            if text:
                specs_text.append(text)
        
        # Specifications tables
        tables = soup.find_all('table')
        for table in tables:
            table_data = self._parse_table(table)
            if table_data:
                specs_tables.append(table_data)
                for item in table_data:
                    specs_text.append(f"{item['key']}: {item['value']}")
        
        return '\n'.join(specs_text), specs_tables
    
    def _extract_generic_specs(self, soup: BeautifulSoup) -> tuple:
        """Extract specifications from generic websites."""
        specs_text = []
        specs_tables = []
        
        # Look for common spec-related elements
        spec_keywords = ['spec', 'feature', 'detail', 'technical', 'overview', 'ingredient', 'nutrition', 'about']
        
        for keyword in spec_keywords:
            # Find divs/sections with spec-related classes
            elements = soup.find_all(['div', 'section', 'article'], 
                                     class_=re.compile(keyword, re.I))
            for elem in elements:
                text = elem.get_text(separator='\n', strip=True)
                if text and len(text) > 20:
                    specs_text.append(text[:2000])  # Limit size
        
        # Extract all tables
        tables = soup.find_all('table')
        for table in tables[:5]:  # Limit to 5 tables
            table_data = self._parse_table(table)
            if table_data:
                specs_tables.append(table_data)
                for item in table_data:
                    specs_text.append(f"{item['key']}: {item['value']}")
        
        # Extract definition lists
        dls = soup.find_all('dl')
        for dl in dls[:3]:
            dts = dl.find_all('dt')
            dds = dl.find_all('dd')
            for dt, dd in zip(dts, dds):
                key = dt.get_text(strip=True)
                value = dd.get_text(strip=True)
                if key and value:
                    specs_text.append(f"{key}: {value}")
        
        # Extract list items that might be specs/features
        lists = soup.find_all(['ul', 'ol'], class_=re.compile(r'feature|spec|detail|benefit', re.I))
        for lst in lists[:3]:
            items = lst.find_all('li')
            for item in items:
                text = item.get_text(strip=True)
                if text and len(text) > 10:
                    specs_text.append(text)
        
        return '\n'.join(specs_text), specs_tables
    
    def _parse_table(self, table) -> List[Dict[str, str]]:
        """Parse an HTML table into key-value pairs."""
        table_data = []
        rows = table.find_all('tr')
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                
                if key and value and len(key) < 100 and len(value) < 500:
                    table_data.append({'key': key, 'value': value})
            elif len(cells) == 1:
                # Could be a header or single-cell row
                text = cells[0].get_text(strip=True)
                if text and len(text) < 200:
                    table_data.append({'key': 'info', 'value': text})
        
        return table_data
    
    def _extract_clean_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from the entire page."""
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        # Limit size
        return text[:10000] if len(text) > 10000 else text
    
    def fetch_multiple(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch multiple URLs with delays between requests.
        
        Args:
            urls: List of URLs to fetch
            
        Returns:
            List of fetch results
        """
        results = []
        for url in urls:
            result = self.fetch_page(url)
            results.append(result)
            # Add delay between requests
            self._random_delay()
        return results