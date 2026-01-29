"""
CLI Interface: Test agents from terminal (UPDATED)
- Passes user query through entire pipeline
- Displays smart recommendation summary
Run: python cli.py
"""

import os
import sys
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.utils.logger import setup_logger
from backend.services.search_service import SearchService
from backend.services.llm_service import LLMService
from backend.services.scraper_service import ScraperService
from backend.agents.agent1_product_discovery import Agent1ProductDiscovery
from backend.agents.agent2_data_retrieval import Agent2DataRetrieval
from backend.agents.agent4_spec_extraction import Agent4SpecExtraction

logger = setup_logger("product-comparison-cli")


class ProductComparisonCLI:
    """Command-line interface for product comparison."""
    
    def __init__(self):
        """Initialize CLI and services."""
        logger.info("🚀 Initializing Product Comparison System...")
        
        # Check environment variables
        serper_key = os.getenv('SERPER_API_KEY')
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if not serper_key:
            logger.error("❌ SERPER_API_KEY not found in .env file")
            sys.exit(1)
        
        if not project_id:
            logger.error("❌ GOOGLE_CLOUD_PROJECT_ID not found in .env file")
            sys.exit(1)
        
        logger.info("✅ Environment variables loaded")
        
        # Initialize services
        self.search_service = SearchService(serper_key)
        self.llm_service = LLMService(project_id, credentials_path)
        self.scraper_service = ScraperService()
        
        # Initialize agents
        self.agent1 = Agent1ProductDiscovery(self.search_service, self.llm_service)
        self.agent2 = Agent2DataRetrieval(self.scraper_service)
        self.agent4 = Agent4SpecExtraction(self.llm_service)
        
        logger.info("✅ All services initialized\n")
    
    def run(self):
        """Run the CLI interface."""
        print("=" * 70)
        print("🛍️  PRODUCT COMPARISON SYSTEM - CLI")
        print("=" * 70)
        print("\nType your query to compare products.")
        print("Examples:")
        print("  - 'Compare iPhone 15 vs Samsung Galaxy S24'")
        print("  - 'Which is cheaper: Close Up or Pepsodent?'")
        print("  - 'MacBook Pro vs Dell XPS for video editing'")
        print("  - 'Best phone for gaming: OnePlus vs Samsung'")
        print("\nType 'quit' to exit\n")
        
        while True:
            try:
                user_input = input("📝 Enter your query: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    print("⚠️  Please enter a query\n")
                    continue
                
                # Run Agent 1
                agent1_result = self.agent1.execute(user_input)
                
                # Display Agent 1 results
                self._display_agent1_results(agent1_result)
                
                # Check if Agent 1 was successful before running Agent 2
                if agent1_result.get('status') == 'success' and agent1_result.get('products'):
                    # Run Agent 2
                    print("\n⏳ Running Agent 2: Data Retrieval...")
                    agent2_result = self.agent2.execute(agent1_result)
                    
                    # Display Agent 2 results
                    self._display_agent2_results(agent2_result)
                    
                    # Check if Agent 2 was successful before running Agent 4
                    if agent2_result.get('status') == 'success' and agent2_result.get('products'):
                        # Run Agent 4
                        print("\n⏳ Running Agent 4: Specification Extraction & Analysis...")
                        agent4_result = self.agent4.execute(agent2_result)
                        
                        # Display Agent 4 results
                        self._display_agent4_results(agent4_result)
                        
                        # Display Smart Summary
                        self._display_smart_summary(agent4_result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                print(f"❌ Error: {str(e)}\n")
    
    def _display_agent1_results(self, result: dict):
        """Display Agent 1 results in a nice format."""
        print("\n" + "=" * 80)
        
        if result.get('status') != 'success':
            print(f"❌ Status: {result.get('status', 'unknown')}")
            if 'message' in result:
                print(f"   {result['message']}")
            if 'error' in result:
                print(f"   Error: {result['error']}")
            print("=" * 80 + "\n")
            return
        
        print("✅ AGENT 1: PRODUCT DISCOVERY - SUCCESS")
        print("=" * 80)
        
        product_type = result.get('product_type', 'unknown')
        confidence = result.get('confidence', 0)
        extraction_method = result.get('extraction_method', 'unknown')
        
        print(f"\n📦 Product Type: {product_type.upper()}")
        print(f"🎯 Confidence: {confidence:.0%}")
        print(f"🔧 Extraction Method: {extraction_method.upper()}")
        
        products = result.get('products', [])
        print(f"\n📊 Found {len(products)} product(s):\n")
        
        for i, product in enumerate(products, 1):
            name = product.get('name', 'Unknown')
            category = product.get('category', 'unknown')
            url_count = product.get('url_count', 0)
            
            print(f"{i}. {name.upper()}")
            print(f"   Category: {category}")
            print(f"   URLs Found: {url_count}")
            
            urls = product.get('urls', [])
            
            for j, url_info in enumerate(urls, 1):
                if isinstance(url_info, dict):
                    url = url_info.get('url', '')
                    title = url_info.get('title', 'No title')
                    source = url_info.get('source', 'unknown')
                    snippet = url_info.get('snippet', '')
                else:
                    url = url_info
                    title = 'No title'
                    source = 'unknown'
                    snippet = ''
                
                print(f"\n   URL {j}:")
                print(f"      Title: {title}")
                print(f"      Source: {source}")
                print(f"      Link: {url}")
                if snippet:
                    snippet_display = snippet[:100] + "..." if len(snippet) > 100 else snippet
                    print(f"      Snippet: {snippet_display}")
            
            print()
        
        print("=" * 80)
        print(f"⏱️  Timestamp: {result.get('timestamp', 'N/A')}")
        print("=" * 80 + "\n")
    
    def _display_agent2_results(self, result: dict):
        """Display Agent 2 results in a nice format."""
        print("\n" + "=" * 80)
        
        if result.get('status') != 'success':
            print(f"❌ AGENT 2: DATA RETRIEVAL - FAILED")
            print(f"   Status: {result.get('status', 'unknown')}")
            if 'error' in result:
                print(f"   Error: {result['error']}")
            print("=" * 80 + "\n")
            return
        
        print("✅ AGENT 2: DATA RETRIEVAL - SUCCESS")
        print("=" * 80)
        
        stats = result.get('stats', {})
        print(f"\n📊 Statistics:")
        print(f"   Products Processed: {stats.get('products_processed', 0)}")
        print(f"   URLs Fetched: {stats.get('urls_fetched', 0)}")
        print(f"   URLs Failed: {stats.get('urls_failed', 0)}")
        
        products = result.get('products', [])
        print(f"\n📦 Enriched Products:\n")
        
        for i, product in enumerate(products, 1):
            name = product.get('name', 'Unknown')
            urls_fetched = product.get('urls_fetched', 0)
            urls_failed = product.get('urls_failed', 0)
            price = product.get('price', '')
            best_source = product.get('best_source')
            
            print(f"{i}. {name.upper()}")
            print(f"   URLs Fetched: {urls_fetched} ✅ | Failed: {urls_failed} ❌")
            
            if price:
                print(f"   💰 Price Found: {price}")
            
            if best_source:
                print(f"   🏆 Best Source: {best_source.get('site_type', 'unknown')}")
                print(f"      Title: {best_source.get('title', 'N/A')[:60]}...")
            
            fetched_data = product.get('fetched_data', [])
            if fetched_data:
                print(f"\n   📄 Fetched Content Summary:")
                for j, data in enumerate(fetched_data, 1):
                    site_type = data.get('site_type', 'unknown')
                    title = data.get('title', 'No title')[:50]
                    specs_len = len(data.get('specs_text', ''))
                    html_len = len(data.get('html_content', ''))
                    
                    print(f"      [{j}] {site_type.upper()}")
                    print(f"          Title: {title}...")
                    print(f"          Specs Text: {specs_len} chars")
                    print(f"          HTML Content: {html_len} chars")
            
            specs_text = product.get('combined_specs_text', '')
            if specs_text:
                preview = specs_text[:300].replace('\n', ' ')
                print(f"\n   📋 Specs Preview:")
                print(f"      {preview}...")
            
            print()
        
        print("=" * 80)
        print(f"⏱️  Timestamp: {result.get('timestamp', 'N/A')}")
        print("=" * 80 + "\n")
    
    def _display_agent4_results(self, result: dict):
        """Display Agent 4 results in a nice format."""
        print("\n" + "=" * 80)
        
        if result.get('status') != 'success':
            print(f"❌ AGENT 4: SPECIFICATION EXTRACTION - FAILED")
            print(f"   Status: {result.get('status', 'unknown')}")
            if 'error' in result:
                print(f"   Error: {result['error']}")
            print("=" * 80 + "\n")
            return
        
        print("✅ AGENT 4: SPECIFICATION EXTRACTION - SUCCESS")
        print("=" * 80)
        
        stats = result.get('stats', {})
        product_type = result.get('product_type', 'unknown')
        user_preferences = result.get('user_preferences', [])
        
        print(f"\n📊 Extraction Statistics:")
        print(f"   Product Type: {product_type.upper()}")
        print(f"   Total Products: {stats.get('total', 0)}")
        print(f"   Successful: {stats.get('successful', 0)} ✅")
        print(f"   Failed: {stats.get('failed', 0)} ❌")
        
        if user_preferences:
            print(f"\n🎯 Detected User Preferences: {', '.join(user_preferences).upper()}")
        
        products = result.get('products', [])
        schema = result.get('schema', {})
        display_names = schema.get('display_names', {})
        
        print(f"\n{'='*80}")
        print("📋 EXTRACTED SPECIFICATIONS")
        print("=" * 80)
        
        for i, product in enumerate(products, 1):
            name = product.get('name', 'Unknown')
            status = product.get('extraction_status', 'unknown')
            specs = product.get('specifications', {})
            
            print(f"\n┌{'─'*78}┐")
            print(f"│ {i}. {name.upper():<74} │")
            print(f"│    Extraction Status: {status.upper():<52} │")
            print(f"├{'─'*78}┤")
            
            if specs:
                for key, value in specs.items():
                    display_key = display_names.get(key, key.replace('_', ' ').title())
                    
                    if isinstance(value, str) and len(value) > 50:
                        value = value[:47] + "..."
                    
                    print(f"│   {display_key:<20}: {str(value):<53} │")
            else:
                print(f"│   {'No specifications extracted':<74} │")
            
            print(f"└{'─'*78}┘")
        
        # Show comparison table if we have 2+ products
        if len(products) >= 2:
            self._display_comparison_table(products, schema)
        
        print("\n" + "=" * 80)
        print(f"⏱️  Timestamp: {result.get('timestamp', 'N/A')}")
        print("=" * 80 + "\n")
    
    def _display_comparison_table(self, products: list, schema: dict):
        """Display a side-by-side comparison table."""
        print(f"\n{'='*80}")
        print("📊 COMPARISON TABLE")
        print("=" * 80)
        
        all_keys = set()
        for product in products:
            specs = product.get('specifications', {})
            all_keys.update(specs.keys())
        
        if not all_keys:
            print("   No specifications to compare")
            return
        
        display_names = schema.get('display_names', {})
        product_names = [p.get('name', 'Unknown')[:20] for p in products]
        col_width = max(25, max(len(n) for n in product_names) + 2)
        
        header = f"{'Specification':<25}"
        for name in product_names:
            header += f" │ {name:<{col_width}}"
        print(f"\n{header}")
        print("─" * len(header))
        
        for key in sorted(all_keys):
            display_key = display_names.get(key, key.replace('_', ' ').title())
            row = f"{display_key:<25}"
            
            for product in products:
                specs = product.get('specifications', {})
                value = specs.get(key, 'N/A')
                
                if isinstance(value, str) and len(value) > col_width - 2:
                    value = value[:col_width - 5] + "..."
                
                row += f" │ {str(value):<{col_width}}"
            
            print(row)
        
        print("─" * len(header))
    
    def _display_smart_summary(self, result: dict):
        """Display the smart recommendation summary."""
        summary = result.get('recommendation_summary', {})
        
        if not summary or summary.get('status') == 'insufficient_products':
            return
        
        user_query = result.get('user_query', '')
        user_preferences = result.get('user_preferences', [])
        
        print("\n" + "=" * 80)
        print("🎯 SMART RECOMMENDATION SUMMARY")
        print("=" * 80)
        
        # Show user query
        if user_query:
            print(f"\n📝 Your Query: \"{user_query}\"")
        
        # Show detected preferences
        if user_preferences:
            print(f"🔍 Detected Priority: {', '.join(user_preferences).upper()}")
        
        # Priority recommendation (if user specified preference)
        priority_rec = summary.get('priority_recommendation', {})
        if priority_rec and priority_rec.get('winner'):
            print(f"\n{'─'*80}")
            aspect = priority_rec.get('aspect', 'overall').upper()
            winner = priority_rec.get('winner', 'Unknown')
            reason = priority_rec.get('reason', '')
            
            if user_preferences:
                print(f"\n⭐ YOUR PRIORITY ({aspect}):")
            else:
                print(f"\n⭐ TOP RECOMMENDATION:")
            
            print(f"   🏆 Winner: {winner.upper()}")
            if reason:
                print(f"   📌 Reason: {reason}")
        
        # All aspect recommendations
        aspect_recs = summary.get('aspect_recommendations', [])
        if aspect_recs:
            print(f"\n{'─'*80}")
            print("\n📊 RECOMMENDATIONS BY ASPECT:\n")
            
            for rec in aspect_recs:
                aspect = rec.get('aspect', 'Unknown').upper()
                winner = rec.get('winner', 'Unknown')
                reason = rec.get('reason', '')
                details = rec.get('details', '')
                
                # Choose emoji based on aspect
                emoji = self._get_aspect_emoji(aspect.lower())
                
                print(f"   {emoji} For {aspect}:")
                print(f"      → Choose {winner.upper()}")
                if reason:
                    print(f"      • {reason}")
                if details:
                    print(f"      • Details: {details}")
                print()
        
        # Overall recommendation
        overall = summary.get('overall_recommendation', {})
        if overall and overall.get('winner'):
            print(f"{'─'*80}")
            print(f"\n🏆 OVERALL VERDICT:")
            print(f"\n   ✅ RECOMMENDED: {overall.get('winner', 'Unknown').upper()}")
            
            summary_text = overall.get('summary', '')
            if summary_text:
                # Word wrap the summary
                words = summary_text.split()
                lines = []
                current_line = []
                for word in words:
                    current_line.append(word)
                    if len(' '.join(current_line)) > 65:
                        lines.append(' '.join(current_line[:-1]))
                        current_line = [word]
                if current_line:
                    lines.append(' '.join(current_line))
                
                print(f"\n   📋 Summary:")
                for line in lines:
                    print(f"      {line}")
        
        print("\n" + "=" * 80 + "\n")
    
    def _get_aspect_emoji(self, aspect: str) -> str:
        """Get emoji for different aspects."""
        emoji_map = {
            'price': '💰',
            'performance': '⚡',
            'battery': '🔋',
            'camera': '📷',
            'display': '📱',
            'quality': '✨',
            'portability': '🎒',
            'features': '🛠️',
            'health': '❤️',
            'whitening': '✨',
            'sensitivity': '🦷',
            'freshness': '🌿',
            'cavity_protection': '🛡️',
            'taste': '👅',
            'calories': '🔥',
            'sugar': '🍬',
            'caffeine': '☕',
            'comfort': '😊',
            'durability': '💪',
            'style': '👟',
            'sound_quality': '🎵',
            'noise_cancellation': '🔇',
            'value': '💎',
            'ingredients': '🧪',
            'benefits': '✅',
        }
        return emoji_map.get(aspect, '📌')


def main():
    """Main entry point."""
    try:
        cli = ProductComparisonCLI()
        cli.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()  