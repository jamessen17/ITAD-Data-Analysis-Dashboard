#!/usr/bin/env python3
"""
Synthetic IT Asset Disposition (ITAD) Dataset Generator
Generates realistic data for refurbishment, second-hand sales, and sustainability analysis
Based on DEFRA and WRAP methodologies for carbon footprint calculations
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import csv

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_itad_dataset(num_records=10000):
    """
    Generate synthetic ITAD dataset with comprehensive metrics
    """
    
    # Define equipment categories and specifications
    equipment_types = {
        'Laptop': {'weight_kg': (1.2, 2.8), 'lifespan_years': (4, 6), 'carbon_kg_new': (200, 400)},
        'Desktop': {'weight_kg': (8, 15), 'lifespan_years': (5, 8), 'carbon_kg_new': (350, 600)},
        'Monitor': {'weight_kg': (3, 8), 'lifespan_years': (6, 10), 'carbon_kg_new': (150, 300)},
        'Server': {'weight_kg': (15, 30), 'lifespan_years': (8, 12), 'carbon_kg_new': (800, 1500)},
        'Tablet': {'weight_kg': (0.3, 0.8), 'lifespan_years': (3, 5), 'carbon_kg_new': (50, 120)},
        'Smartphone': {'weight_kg': (0.15, 0.25), 'lifespan_years': (2, 4), 'carbon_kg_new': (20, 80)},
        'Printer': {'weight_kg': (5, 25), 'lifespan_years': (5, 8), 'carbon_kg_new': (180, 400)}
    }
    
    # Regional data with market characteristics
    regions = {
        'North America': {'market_size': 0.35, 'price_multiplier': 1.2, 'grid_intensity_kg_co2_kwh': 0.42},
        'Europe': {'market_size': 0.28, 'price_multiplier': 1.1, 'grid_intensity_kg_co2_kwh': 0.31},
        'Asia Pacific': {'market_size': 0.25, 'price_multiplier': 0.85, 'grid_intensity_kg_co2_kwh': 0.68},
        'Latin America': {'market_size': 0.08, 'price_multiplier': 0.7, 'grid_intensity_kg_co2_kwh': 0.35},
        'Middle East & Africa': {'market_size': 0.04, 'price_multiplier': 0.8, 'grid_intensity_kg_co2_kwh': 0.55}
    }
    
    # Sales channels with different characteristics
    sales_channels = {
        'B2B Direct': {'commission': 0.05, 'volume_share': 0.45, 'avg_order_value_mult': 2.5},
        'Online Marketplace': {'commission': 0.12, 'volume_share': 0.30, 'avg_order_value_mult': 1.0},
        'Retail Partner': {'commission': 0.20, 'volume_share': 0.15, 'avg_order_value_mult': 1.2},
        'Refurb Specialist': {'commission': 0.08, 'volume_share': 0.10, 'avg_order_value_mult': 1.8}
    }
    
    # Generate base dataset
    data = []
    
    for i in range(num_records):
        # Basic asset information
        equipment_type = np.random.choice(list(equipment_types.keys()))
        eq_specs = equipment_types[equipment_type]
        
        # Asset characteristics
        asset_id = f"ITAD-{i+1:06d}"
        asset_age_years = np.random.exponential(3.5)  # Most assets are relatively new
        asset_age_years = min(asset_age_years, 15)  # Cap at 15 years
        
        original_purchase_price = np.random.normal(800, 400) * eq_specs['carbon_kg_new'][1] / 400
        original_purchase_price = max(original_purchase_price, 100)  # Minimum price
        
        weight_kg = np.random.uniform(*eq_specs['weight_kg'])
        
        # Regional assignment
        region = np.random.choice(list(regions.keys()), 
                                p=[regions[r]['market_size'] for r in regions.keys()])
        
        # Asset condition assessment (1-10 scale, higher is better)
        # Older equipment generally has lower condition scores
        base_condition = 10 - (asset_age_years / eq_specs['lifespan_years'][1]) * 6
        condition_score = np.random.normal(base_condition, 1.5)
        condition_score = np.clip(condition_score, 1, 10)
        
        # Refurbishment decision and metrics
        refurb_threshold = 4.5  # Minimum condition for refurbishment
        can_refurbish = condition_score >= refurb_threshold
        
        if can_refurbish:
            refurb_cost = np.random.normal(150, 50) * (equipment_types[equipment_type]['carbon_kg_new'][1] / 400)
            refurb_cost = max(refurb_cost, 20)
            refurb_success_prob = min(0.95, 0.6 + (condition_score - 4) * 0.08)
            refurb_success = np.random.binomial(1, refurb_success_prob)
        else:
            refurb_cost = 0
            refurb_success = 0
            refurb_success_prob = 0
        
        # Second-hand market value
        depreciation_rate = 1 - (asset_age_years / (eq_specs['lifespan_years'][1] * 1.5))
        depreciation_rate = max(depreciation_rate, 0.05)  # Minimum residual value
        
        if refurb_success:
            # Refurbished items command higher prices
            market_value = original_purchase_price * depreciation_rate * 1.3
            product_condition = "Refurbished"
        elif condition_score >= 6:
            market_value = original_purchase_price * depreciation_rate
            product_condition = "Good"
        elif condition_score >= 3:
            market_value = original_purchase_price * depreciation_rate * 0.7
            product_condition = "Fair"
        else:
            market_value = original_purchase_price * depreciation_rate * 0.3
            product_condition = "Poor"
        
        # Apply regional pricing multiplier
        market_value *= regions[region]['price_multiplier']
        
        # Sales channel selection
        sales_channel = np.random.choice(list(sales_channels.keys()),
                                       p=[sales_channels[sc]['volume_share'] for sc in sales_channels.keys()])
        
        # Final sale price (including channel effects)
        channel_mult = sales_channels[sales_channel]['avg_order_value_mult']
        final_sale_price = market_value * channel_mult * np.random.normal(1, 0.1)
        final_sale_price = max(final_sale_price, 10)
        
        # Carbon footprint calculations (based on DEFRA methodology)
        carbon_intensity_region = regions[region]['grid_intensity_kg_co2_kwh']
        
        # Carbon emissions for new equipment
        carbon_new_manufacturing = np.random.uniform(*eq_specs['carbon_kg_new'])
        
        # Carbon saved by refurbishment/reuse instead of new purchase
        if refurb_success or condition_score >= 3:  # Sold as second-hand
            # Avoided emissions from not buying new
            carbon_avoided_manufacturing = carbon_new_manufacturing * 0.85  # 85% of new emissions avoided
            
            # Refurbishment process emissions
            if refurb_success:
                refurb_energy_kwh = weight_kg * 2.5  # Energy intensive refurb process
                carbon_refurb_process = refurb_energy_kwh * carbon_intensity_region
            else:
                carbon_refurb_process = 0
            
            # Net carbon savings
            carbon_savings_kg = carbon_avoided_manufacturing - carbon_refurb_process
        else:
            # Item goes to recycling/disposal
            carbon_savings_kg = carbon_new_manufacturing * 0.15  # Only material recovery
        
        # Extended lifecycle impact
        if refurb_success:
            lifecycle_extension_years = np.random.uniform(1.5, 3.5)
        elif condition_score >= 6:
            lifecycle_extension_years = np.random.uniform(1, 2.5)
        elif condition_score >= 3:
            lifecycle_extension_years = np.random.uniform(0.5, 1.5)
        else:
            lifecycle_extension_years = 0
        
        # Seasonal demand patterns
        month = np.random.randint(1, 13)
        if month in [9, 10, 11]:  # Back-to-school/fiscal year-end
            seasonal_demand_mult = 1.15
        elif month in [12, 1]:  # Holiday season
            seasonal_demand_mult = 1.08
        else:
            seasonal_demand_mult = 1.0
        
        # Processing date
        processing_date = datetime.now() - timedelta(days=np.random.randint(0, 365))
        
        # Compile record
        record = {
            'asset_id': asset_id,
            'equipment_type': equipment_type,
            'asset_age_years': round(asset_age_years, 2),
            'weight_kg': round(weight_kg, 2),
            'region': region,
            'original_purchase_price_usd': round(original_purchase_price, 2),
            'condition_score': round(condition_score, 1),
            'product_condition': product_condition,
            'can_refurbish': can_refurbish,
            'refurbishment_cost_usd': round(refurb_cost, 2) if can_refurbish else 0,
            'refurbishment_success': bool(refurb_success),
            'refurbishment_success_probability': round(refurb_success_prob, 3),
            'sales_channel': sales_channel,
            'market_value_usd': round(market_value, 2),
            'final_sale_price_usd': round(final_sale_price, 2),
            'channel_commission_rate': sales_channels[sales_channel]['commission'],
            'net_revenue_usd': round(final_sale_price * (1 - sales_channels[sales_channel]['commission']), 2),
            'carbon_new_manufacturing_kg': round(carbon_new_manufacturing, 2),
            'carbon_refurb_process_kg': round(carbon_refurb_process, 2) if 'carbon_refurb_process' in locals() else 0,
            'carbon_savings_kg': round(carbon_savings_kg, 2),
            'lifecycle_extension_years': round(lifecycle_extension_years, 2),
            'grid_carbon_intensity_kg_co2_kwh': carbon_intensity_region,
            'processing_date': processing_date.strftime('%Y-%m-%d'),
            'processing_month': month,
            'seasonal_demand_multiplier': seasonal_demand_mult,
            'roi_percent': round(((final_sale_price - refurb_cost) / max(refurb_cost, original_purchase_price * 0.1)) * 100, 2)
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

def export_to_csv(df, filename='synthetic_itad_dataset.csv'):
    """
    Export dataset to CSV with proper formatting
    """
    df.to_csv(filename, index=False, float_format='%.2f')
    print(f"Dataset exported to {filename}")
    print(f"Total records: {len(df)}")
    
    # Print summary statistics
    print("\n=== DATASET SUMMARY ===")
    print(f"Equipment types: {df['equipment_type'].value_counts().to_dict()}")
    print(f"Regions: {df['region'].value_counts().to_dict()}")
    print(f"Sales channels: {df['sales_channel'].value_counts().to_dict()}")
    print(f"Refurbishment success rate: {df['refurbishment_success'].mean():.1%}")
    print(f"Average carbon savings: {df['carbon_savings_kg'].mean():.1f} kg CO2e")
    print(f"Average final sale price: ${df['final_sale_price_usd'].mean():.2f}")
    print(f"Total carbon savings: {df['carbon_savings_kg'].sum():,.0f} kg CO2e")

def validate_and_clean_data(df):
    """
    Validate and clean the ITAD dataset.
    Removes duplicates, fills missing values, and enforces valid ranges.
    """
    # Remove duplicate asset IDs
    df = df.drop_duplicates(subset=['asset_id'])
    # Fill missing values with reasonable defaults
    df = df.fillna({
        'condition_score': 5,
        'refurbishment_success_probability': 0,
        'final_sale_price_usd': 10,
        'carbon_savings_kg': 0,
        'lifecycle_extension_years': 0
    })
    # Clip condition scores and probabilities to valid ranges
    df['condition_score'] = df['condition_score'].clip(1, 10)
    df['refurbishment_success_probability'] = df['refurbishment_success_probability'].clip(0, 1)
    return df

def generate_data_quality_report(df):
    """
    Generate a simple data quality report.
    """
    print("\n=== DATA QUALITY REPORT ===")
    print(f"Null values per column:\n{df.isnull().sum()}")
    print(f"Duplicate asset IDs: {df['asset_id'].duplicated().sum()}")
    print(f"Condition scores out of range: {(~df['condition_score'].between(1, 10)).sum()}")
    print(f"Probabilities out of range: {(~df['refurbishment_success_probability'].between(0, 1)).sum()}")

if __name__ == "__main__":
    # Generate the dataset
    print("Generating synthetic ITAD dataset...")
    
    # You can adjust the number of records here
    itad_dataset = generate_itad_dataset(num_records=10000)
    
    # Validate and clean the data
    itad_dataset = validate_and_clean_data(itad_dataset)
    
    # Generate data quality report
    generate_data_quality_report(itad_dataset)
    
    # Export to CSV
    export_to_csv(itad_dataset, 'synthetic_itad_dataset.csv')
    
    # Optional: Display sample records
    print("\n=== SAMPLE RECORDS ===")
    print(itad_dataset.head())
    
    # Optional: Create additional segmented datasets
    print("\n=== CREATING SEGMENTED DATASETS ===")
    
    # High-value refurbishment opportunities
    high_value_refurb = itad_dataset[
        (itad_dataset['can_refurbish'] == True) & 
        (itad_dataset['final_sale_price_usd'] > 500)
    ]
    if len(high_value_refurb) > 0:
        high_value_refurb.to_csv('high_value_refurbishment_opportunities.csv', index=False)
        print(f"High-value refurbishment dataset: {len(high_value_refurb)} records")
    else:
        print("No high-value refurbishment opportunities found")
    
    # Carbon impact analysis dataset
    carbon_analysis = itad_dataset[['asset_id', 'equipment_type', 'region', 'carbon_savings_kg', 
                                   'lifecycle_extension_years', 'refurbishment_success']]
    carbon_analysis.to_csv('carbon_impact_analysis.csv', index=False)
    print(f"Carbon impact analysis dataset: {len(carbon_analysis)} records")
    
    # Regional sales performance dataset
    try:
        regional_sales = itad_dataset.groupby(['region', 'sales_channel']).agg({
            'final_sale_price_usd': ['mean', 'sum', 'count'],
            'carbon_savings_kg': 'sum',
            'refurbishment_success': 'mean'
        }).round(2)
        regional_sales.to_csv('regional_sales_performance.csv')
        print("Regional sales performance dataset created")
    except Exception as e:
        print(f"Warning: Could not create regional sales dataset: {e}")
    
    # Final validation summary
    print(f"\n=== FINAL DATA VALIDATION ===")
    print(f"✓ Dataset contains {len(itad_dataset):,} clean records")
    print(f"✓ No null values: {itad_dataset.isnull().sum().sum() == 0}")
    print(f"✓ No duplicate asset IDs: {itad_dataset['asset_id'].duplicated().sum() == 0}")
    print(f"✓ All condition scores in range 1-10: {itad_dataset['condition_score'].between(1, 10).all()}")
    print(f"✓ All probabilities in range 0-1: {itad_dataset['refurbishment_success_probability'].between(0, 1).all()}")
    
    print("\n=== DATASET GENERATION COMPLETE ===")
    print("Files created:")
    print("1. synthetic_itad_dataset.csv (main dataset)")
    if len(high_value_refurb) > 0:
        print("2. high_value_refurbishment_opportunities.csv")
    print("3. carbon_impact_analysis.csv") 
    print("4. regional_sales_performance.csv")