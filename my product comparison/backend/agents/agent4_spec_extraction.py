"""
Agent 4: Specification Extraction (FULLY DYNAMIC - FIXED)
- Uses ₹ prices from Shopping API
- Converts USD to INR if needed
- Flattens nested JSON responses
- Normalizes spec keys for consistency
- LLM decides comparison aspects based on product type
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class Agent4SpecExtraction:
    """
    Extracts product specifications and generates recommendations.
    Fully dynamic - LLM decides aspects based on product type.
    """
    
    # USD to INR conversion rate (approximate)
    USD_TO_INR = 83.0
    
    # Key normalization mapping
    KEY_NORMALIZATION = {
        'resolution': 'display_resolution',
        'screen_size': 'display_size',
        'screen': 'display_size',
        'display': 'display_size',
        'camera': 'camera_main',
        'rear_camera': 'camera_main',
        'back_camera': 'camera_main',
        'selfie_camera': 'camera_front',
        'front_cam': 'camera_front',
        'chip': 'processor',
        'chipset': 'processor',
        'cpu': 'processor',
        'soc': 'processor',
        'memory': 'ram',
        'internal_storage': 'storage',
        'rom': 'storage',
        'battery_capacity': 'battery',
        'os_version': 'os',
        'operating_system': 'os',
        'weight_g': 'weight',
        'mass': 'weight',
    }
    
    def __init__(self, llm_service):
        """Initialize Agent 4."""
        self.llm_service = llm_service
        logger.info("✅ Agent 4 (Spec Extraction - Fully Dynamic Fixed) initialized")
    
    def execute(self, agent2_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute specification extraction.
        """
        logger.info("🔄 Agent 4: Starting specification extraction...")
        
        # Get data from agent2
        if isinstance(agent2_result, dict):
            products = agent2_result.get('products', [])
            product_type = agent2_result.get('product_type', 'general')
            user_query = agent2_result.get('user_query', '')
        else:
            products = agent2_result
            product_type = 'general'
            user_query = ''
        
        logger.info(f"📦 Product type: {product_type}")
        
        if not products:
            return {
                'status': 'no_products',
                'products': [],
                'user_query': user_query
            }
        
        # Detect user preferences from query
        user_preferences = self._detect_preferences(user_query, product_type)
        logger.info(f"🎯 User preferences: {user_preferences}")
        
        # Extract specs for each product (dynamic based on product type)
        for product in products:
            self._extract_specs(product, product_type)
        
        # Generate recommendations (dynamic aspects based on product type)
        logger.info("📊 Generating recommendations...")
        recommendation = self._generate_recommendations(products, user_query, user_preferences, product_type)
        
        return {
            'status': 'success',
            'products': products,
            'product_type': product_type,
            'user_query': user_query,
            'user_preferences': user_preferences,
            'recommendation_summary': recommendation,
            'stats': {
                'total': len(products),
                'successful': sum(1 for p in products if p.get('extraction_status') == 'success'),
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _detect_preferences(self, query: str, product_type: str) -> List[str]:
        """
        Detect user preferences from query using LLM.
        """
        if not query:
            return []
        
        try:
            prompt = f"""Analyze this user query and identify their priorities/preferences.

Query: "{query}"
Product Type: {product_type}

Return ONLY a JSON array of 1-3 preference keywords that matter to this user.
Examples:
- For "best camera phone under 20000" → ["camera", "price"]
- For "toothpaste for sensitive teeth" → ["sensitivity protection"]
- For "gaming laptop with good battery" → ["gaming performance", "battery"]
- For "running shoes for marathon" → ["comfort", "durability"]

Return ONLY the JSON array, nothing else:
["preference1", "preference2"]"""

            response = self.llm_service.extract_products(prompt)
            
            json_match = re.search(r'\[[\s\S]*?\]', response)
            if json_match:
                preferences = json.loads(json_match.group())
                if isinstance(preferences, list):
                    return preferences[:3]
            
            return []
        
        except Exception as e:
            logger.warning(f"Preference detection failed: {e}")
            return []
    
    def _extract_specs(self, product: Dict[str, Any], product_type: str) -> None:
        """
        Extract specifications for a product using LLM.
        """
        product_name = product.get('name', 'Unknown')
        logger.info(f"📋 Extracting specs for '{product_name}' (type: {product_type})")
        
        # Get price from Shopping API
        shopping_price = product.get('price', '')
        
        # Get any scraped content
        scraped_content = self._get_scraped_content(product)
        
        # Build dynamic LLM prompt
        prompt = self._build_extraction_prompt(product_name, product_type, scraped_content, shopping_price)
        
        try:
            response = self.llm_service.extract_products(prompt)
            specs = self._parse_specs(response)
            
            if specs:
                # Handle price with fallback and conversion
                specs = self._handle_price(specs, shopping_price, product)
                
                # Normalize keys
                specs = self._normalize_keys(specs)
                
                product['specifications'] = specs
                product['extraction_status'] = 'success'
                logger.info(f"   ✅ Extracted {len(specs)} specs")
            else:
                product['specifications'] = {'product_name': product_name}
                product['extraction_status'] = 'partial'
        
        except Exception as e:
            logger.error(f"   ❌ Extraction error: {e}")
            product['specifications'] = {'product_name': product_name}
            product['extraction_status'] = 'failed'
        
        # Final price check
        if shopping_price and '₹' in str(shopping_price):
            product['specifications']['price'] = shopping_price
    
    def _handle_price(self, specs: Dict[str, Any], shopping_price: str, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle price with multiple fallbacks:
        1. Use Shopping API price (₹)
        2. Convert USD to INR if price is in $
        3. Use all_prices from product
        """
        # Priority 1: Shopping API price in INR
        if shopping_price and '₹' in str(shopping_price):
            specs['price'] = shopping_price
            return specs
        
        # Priority 2: Check if specs has price
        current_price = specs.get('price', '')
        
        if current_price:
            # Check if it's in USD and convert
            if '$' in str(current_price):
                inr_price = self._convert_usd_to_inr(current_price)
                if inr_price:
                    specs['price'] = inr_price
                    logger.info(f"   💱 Converted {current_price} to {inr_price}")
                    return specs
            
            # If already in INR, keep it
            if '₹' in str(current_price):
                return specs
        
        # Priority 3: Try all_prices from product
        all_prices = product.get('all_prices', [])
        for price_info in all_prices:
            price = price_info.get('price', '')
            if price and '₹' in str(price):
                specs['price'] = price
                return specs
        
        # Priority 4: If still USD, convert it
        if current_price and '$' in str(current_price):
            inr_price = self._convert_usd_to_inr(current_price)
            if inr_price:
                specs['price'] = inr_price
        
        return specs
    
    def _convert_usd_to_inr(self, usd_price: str) -> Optional[str]:
        """
        Convert USD price to INR.
        Examples: "$350", "$350.00", "$ 350", "$1,299.99"
        """
        try:
            # Remove $ and commas, extract number
            clean_price = usd_price.replace('$', '').replace(',', '').strip()
            
            # Extract numeric value
            match = re.search(r'[\d.]+', clean_price)
            if match:
                usd_value = float(match.group())
                inr_value = usd_value * self.USD_TO_INR
                
                # Format as INR
                if inr_value >= 1000:
                    return f"₹{inr_value:,.0f}"
                else:
                    return f"₹{inr_value:.2f}"
            
            return None
        
        except Exception as e:
            logger.warning(f"Price conversion failed for {usd_price}: {e}")
            return None
    
    def _normalize_keys(self, specs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize specification keys to standard names.
        """
        normalized = {}
        
        for key, value in specs.items():
            # Skip None or empty values
            if value is None or value == '':
                continue
            
            # Convert key to lowercase for matching
            key_lower = key.lower().replace(' ', '_')
            
            # Check if key needs normalization
            normalized_key = self.KEY_NORMALIZATION.get(key_lower, key_lower)
            
            # Store with normalized key
            normalized[normalized_key] = value
        
        return normalized
    
    def _get_scraped_content(self, product: Dict[str, Any]) -> str:
        """Get any available scraped content."""
        parts = []
        
        if product.get('combined_specs_text'):
            parts.append(product['combined_specs_text'][:2000])
        
        for data in product.get('fetched_data', []):
            if data.get('specs_text'):
                parts.append(data['specs_text'][:1000])
            if data.get('clean_text'):
                parts.append(data['clean_text'][:1000])
        
        return '\n'.join(parts)
    
    def _build_extraction_prompt(self, product_name: str, product_type: str, scraped_content: str, price: str) -> str:
        """
        Build DYNAMIC LLM prompt for spec extraction.
        """
        
        content_section = ""
        if scraped_content:
            content_section = f"""
Scraped content (use if helpful):
---
{scraped_content[:3000]}
---
"""
        
        price_note = ""
        if price and '₹' in str(price):
            price_note = f"Current Price: {price} (USE THIS EXACT PRICE)"
        
        return f"""You are a product specification expert. Extract relevant specifications for this product.

Product: {product_name}
Product Type: {product_type}
{price_note}
{content_section}

IMPORTANT RULES:
1. Return a FLAT JSON object (no nested objects!)
2. Use CONSISTENT key names from this list:
   - brand, model, product_name
   - display_size, display_resolution, refresh_rate
   - processor, ram, storage
   - battery, charging
   - camera_main, camera_front
   - os, weight, price
   - 5g_support, nfc, bluetooth
3. Values must be simple strings or numbers (NOT objects or arrays)
4. If price is known, include it in INR (₹) format
5. Omit any field you don't know (don't write "N/A" or "Unknown")

For {product_type}, extract the RELEVANT specifications only.

CORRECT FORMAT EXAMPLE:
{{
    "brand": "Samsung",
    "model": "Galaxy S24",
    "display_size": "6.2 inches",
    "processor": "Snapdragon 8 Gen 3",
    "ram": "8GB",
    "storage": "256GB",
    "battery": "4000 mAh",
    "camera_main": "50 MP",
    "camera_front": "12 MP",
    "os": "Android 14",
    "price": "₹79,999"
}}

WRONG FORMAT (DO NOT DO THIS):
{{
    "smartphone": {{
        "brand": "Samsung"
    }}
}}

Return ONLY the flat JSON object, no explanation or markdown."""
    
    def _parse_specs(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from LLM response.
        Handles nested JSON by flattening it.
        """
        if not response:
            return None
        
        try:
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return None
            
            specs = json.loads(json_match.group())
            
            # Flatten if nested
            specs = self._flatten_specs(specs)
            
            # Clean and validate values
            cleaned = {}
            for k, v in specs.items():
                # Skip None, empty, or placeholder values
                if v is None:
                    continue
                if isinstance(v, str) and v.upper() in ['UNKNOWN', 'N/A', '', 'NOT APPLICABLE', 'NONE']:
                    continue
                
                # Skip if value is still a dict or list (shouldn't happen after flatten)
                if isinstance(v, (dict, list)):
                    logger.warning(f"Skipping non-scalar value for key '{k}': {type(v)}")
                    continue
                
                # Convert boolean to Yes/No for display
                if isinstance(v, bool):
                    v = "Yes" if v else "No"
                
                cleaned[k] = v
            
            return cleaned if cleaned else None
        
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return None
    
    def _flatten_specs(self, specs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested JSON specs.
        If LLM returns {"smartphone": {...}}, extract the inner object.
        """
        if not isinstance(specs, dict):
            return {}
        
        # Check if it's a single-key nested structure
        keys = list(specs.keys())
        
        if len(keys) == 1:
            inner = specs[keys[0]]
            # If the single value is a dict, it's likely nested - flatten it
            if isinstance(inner, dict):
                logger.info(f"   🔄 Flattening nested JSON (key: {keys[0]})")
                return self._flatten_specs(inner)  # Recursive in case of deep nesting
        
        # Check for any nested dicts and flatten them
        flattened = {}
        for key, value in specs.items():
            if isinstance(value, dict):
                # If value is a dict, try to extract its values with prefixed keys
                # But first check if it looks like a product category wrapper
                if key.lower() in ['smartphone', 'phone', 'mobile', 'laptop', 'tablet', 
                                   'toothpaste', 'shampoo', 'product', 'specs', 'specifications']:
                    # It's a wrapper - extract inner values directly
                    for inner_key, inner_value in value.items():
                        if not isinstance(inner_value, dict):
                            flattened[inner_key] = inner_value
                else:
                    # It's a nested object we can't flatten nicely - skip it
                    logger.warning(f"   ⚠️ Skipping nested object: {key}")
            else:
                flattened[key] = value
        
        return flattened
    
    def _generate_recommendations(self, products: List[Dict], user_query: str, 
                                   preferences: List[str], product_type: str) -> Dict[str, Any]:
        """
        Generate comparison recommendations using LLM.
        FULLY DYNAMIC - LLM decides comparison aspects based on product type.
        """
        
        if len(products) < 2:
            return {'status': 'insufficient_products'}
        
        # Prepare product data
        product_data = []
        for p in products:
            specs = p.get('specifications', {})
            product_data.append({
                'name': p.get('name', 'Unknown'),
                'price': p.get('price') or specs.get('price', 'N/A'),
                'specifications': specs
            })
        
        priority = preferences[0] if preferences else 'overall value'
        product_names = [p['name'] for p in product_data]
        
        prompt = f"""You are a product comparison expert. Compare these {product_type} products and provide recommendations.

User Query: "{user_query}"
User Priority: {priority}
Product Type: {product_type}

Products to Compare:
{json.dumps(product_data, indent=2)}

IMPORTANT INSTRUCTIONS:
1. First, determine 5-6 relevant comparison aspects for "{product_type}" products.
   - For smartphones: price, performance, camera, battery, display, storage
   - For toothpaste: price, whitening, cavity protection, freshness, sensitivity protection, ingredients
   - For laptops: price, performance, display, portability, battery, build quality
   - For shampoo: price, effectiveness, hair type suitability, ingredients, fragrance
   - For shoes: price, comfort, durability, style, material quality
   - Choose aspects that make sense for {product_type}!

2. Compare the products on each of these RELEVANT aspects.

3. Use ACTUAL data from the specifications provided above.

Return ONLY this JSON format:
{{
    "user_priority": "{priority}",
    "priority_recommendation": {{
        "aspect": "{priority}",
        "winner": "product name from {product_names}",
        "reason": "specific reason based on actual specs"
    }},
    "aspect_recommendations": [
        {{
            "aspect": "first relevant aspect for {product_type}",
            "winner": "product name",
            "reason": "specific reason with actual values",
            "details": "comparison details"
        }},
        {{
            "aspect": "second relevant aspect for {product_type}",
            "winner": "product name",
            "reason": "specific reason with actual values",
            "details": "comparison details"
        }},
        {{
            "aspect": "third relevant aspect for {product_type}",
            "winner": "product name",
            "reason": "specific reason with actual values",
            "details": "comparison details"
        }},
        {{
            "aspect": "fourth relevant aspect for {product_type}",
            "winner": "product name",
            "reason": "specific reason with actual values",
            "details": "comparison details"
        }},
        {{
            "aspect": "fifth relevant aspect for {product_type}",
            "winner": "product name",
            "reason": "specific reason with actual values",
            "details": "comparison details"
        }}
    ],
    "overall_recommendation": {{
        "winner": "best product name",
        "summary": "2-3 sentence summary explaining why this {product_type} is recommended based on the user's needs"
    }}
}}

CRITICAL RULES:
1. The aspects MUST be relevant to {product_type} - do NOT use smartphone aspects for toothpaste!
2. Use ACTUAL specifications from the data provided
3. Be specific with comparisons (e.g., "contains fluoride vs fluoride-free" for toothpaste)
4. Return ONLY valid JSON, no markdown or explanation"""

        try:
            response = self.llm_service.extract_products(prompt)
            
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                result['status'] = 'success'
                return result
            
            return self._fallback_recommendation(products, preferences, product_type)
        
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return self._fallback_recommendation(products, preferences, product_type)
    
    def _fallback_recommendation(self, products: List[Dict], preferences: List[str], product_type: str) -> Dict[str, Any]:
        """Fallback when LLM fails."""
        names = [p.get('name', 'Unknown') for p in products]
        priority = preferences[0] if preferences else 'overall value'
        
        return {
            'status': 'fallback',
            'user_priority': priority,
            'priority_recommendation': {
                'aspect': priority,
                'winner': names[0],
                'reason': 'Based on available data'
            },
            'aspect_recommendations': [
                {
                    'aspect': 'overall',
                    'winner': 'See comparison table',
                    'reason': f'Compare the {product_type} specifications above to make your decision',
                    'details': ''
                }
            ],
            'overall_recommendation': {
                'winner': names[0],
                'summary': f'Compare {" and ".join(names)} using the specification table above to find the best {product_type} for your needs.'
            }
        }


def create_agent4(llm_service):
    """Factory function."""
    return Agent4SpecExtraction(llm_service)