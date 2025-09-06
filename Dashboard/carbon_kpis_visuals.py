
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_and_visualize(csv_path="carbon_impact_analysis.csv"):
    df = pd.read_csv(csv_path)

    # KPIs
    total_co2e_tons = df['carbon_savings_kg'].sum() / 1000
    avg_per_asset_kg = df['carbon_savings_kg'].mean()
    success_rate = df['refurbishment_success'].mean() * 100
    avg_success_kg = df[df['refurbishment_success'] == True]['carbon_savings_kg'].mean()
    avg_fail_kg = df[df['refurbishment_success'] == False]['carbon_savings_kg'].mean()
    uplift_success_vs_fail_pct = ((avg_success_kg - avg_fail_kg) / avg_fail_kg) * 100 if avg_fail_kg != 0 else np.nan

    df_eff = df[df['lifecycle_extension_years'] > 0].copy()
    df_eff['kg_per_year'] = df_eff['carbon_savings_kg'] / df_eff['lifecycle_extension_years']
    median_eff_kg_per_year = df_eff['kg_per_year'].median()

    region_totals = df.groupby('region')['carbon_savings_kg'].sum().sort_values(ascending=False)

    print("=== KEY KPIs ===")
    print(f"Total CO2e avoided: {total_co2e_tons:.2f} tons")
    print(f"Average CO2e per asset: {avg_per_asset_kg:.2f} kg")
    print(f"Refurbishment success rate: {success_rate:.2f}%")
    print(f"Avg CO2e (success): {avg_success_kg:.2f} kg | Avg CO2e (fail): {avg_fail_kg:.2f} kg | Uplift: {uplift_success_vs_fail_pct:.2f}%")
    print(f"Median CO2e per added year: {median_eff_kg_per_year:.2f} kg/year")

    # Charts
    device_totals = df.groupby('equipment_type')['carbon_savings_kg'].sum().sort_values(ascending=False) / 1000
    plt.figure(figsize=(10,6)); plt.bar(device_totals.index, device_totals.values)
    plt.title("Total CO2e Avoided by Device Type (tons)"); plt.xlabel("Device Type"); plt.ylabel("Tons CO2e avoided"); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.show()

    avg_by_device_kg = df.groupby('equipment_type')['carbon_savings_kg'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10,6)); plt.bar(avg_by_device_kg.index, avg_by_device_kg.values)
    plt.title("Average CO2e Avoided per Asset by Device Type (kg)"); plt.xlabel("Device Type"); plt.ylabel("Average kg CO2e avoided"); plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,6)); plt.scatter(df['lifecycle_extension_years'], df['carbon_savings_kg'], alpha=0.4)
    plt.title("Carbon Savings vs Lifecycle Extension"); plt.xlabel("Lifecycle Extension (years)"); plt.ylabel("CO2e avoided (kg)"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8,6)); plt.hist(df_eff['kg_per_year'].dropna(), bins=30)
    plt.title("Distribution of CO2e Avoided per Added Year (kg/year)"); plt.xlabel("kg CO2e avoided per year"); plt.ylabel("Count of assets"); plt.tight_layout(); plt.show()

    plt.figure(figsize=(7,7)); plt.pie(region_totals.values, labels=region_totals.index, autopct="%1.1f%%", startangle=90)
    plt.title("Regional Share of Total CO2e Avoided"); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    analyze_and_visualize()
