#!/usr/bin/env python
"""Test script for portfolio endpoint debugging"""

import sys
sys.path.insert(0, '.')

print("Testing portfolio endpoint registration...")

try:
    # Import the main module
    import main_simple_for_frontend as m
    
    # Get all routes
    all_routes = []
    for r in m.app.routes:
        if hasattr(r, 'path') and hasattr(r, 'methods'):
            all_routes.append((r.methods, r.path))
    
    print(f"\nTotal routes: {len(all_routes)}")
    
    # Find portfolio routes
    print("\nPortfolio routes:")
    for methods, path in all_routes:
        if 'portfolio' in path.lower() or 'generate' in path.lower():
            print(f"  {methods}: {path}")
    
    # Check if generate_portfolio function is accessible
    print("\n\nChecking generate_portfolio function...")
    if hasattr(m, 'generate_portfolio'):
        print("  ✓ generate_portfolio function exists")
    else:
        print("  ✗ generate_portfolio function NOT FOUND")
    
    if hasattr(m, 'generate_portfolio_legacy'):
        print("  ✓ generate_portfolio_legacy function exists")
    else:
        print("  ✗ generate_portfolio_legacy function NOT FOUND")
        
    if hasattr(m, '_generate_portfolio_impl'):
        print("  ✓ _generate_portfolio_impl function exists")
    else:
        print("  ✗ _generate_portfolio_impl function NOT FOUND")
    
    # Check PortfolioGenerateRequest
    if hasattr(m, 'PortfolioGenerateRequest'):
        print("  ✓ PortfolioGenerateRequest class exists")
    else:
        print("  ✗ PortfolioGenerateRequest class NOT FOUND")

except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
