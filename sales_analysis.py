import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic context for seaborn
sns.set_theme(style="whitegrid")

def main():
    print("-" * 50)
    print("[START] Sales Data Analysis Project Starting...")
    print("-" * 50)

    # ==========================================
    # 1. DATA LOADING
    # ==========================================
    file_path = r"C:\Users\user\Downloads\archive (1)\sales_data_sample.csv"
    
    print(f"\n[1/4] Loading data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found at {file_path}.")
        print("Please check the path and try again.")
        return

    # Loading the dataset with ISO-8859-1 encoding to handle special characters often found in CSVs
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        print(f"[SUCCESS] Data loaded successfully! Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return

    # ==========================================
    # 2. DATA CLEANING
    # ==========================================
    print("\n[2/4] Performing Data Cleaning...")
    
    # 2.1 Remove duplicates
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"  -> Removed {initial_rows - df.shape[0]} duplicate rows.")

    # 2.2 Handle missing values
    # Display columns with missing values before filling/dropping
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print("  -> Handling missing values in columns:")
        for col, count in missing_data[missing_data > 0].items():
            print(f"     - {col}: {count} missing values")
            
        # Strategy: Fill numeric with median, object/string with 'Unknown'
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna('Unknown')
    else:
        print("  -> No missing values found.")

    # 2.3 Convert Data Types
    # Assuming standard sales dataset columns like 'ORDERDATE' exist
    # Adjust the column exact name based on the dataset if 'ORDERDATE' is different.
    date_col_candidates = ['ORDERDATE', 'Order Date', 'Date', 'date']
    date_col = next((col for col in date_col_candidates if col in df.columns), None)
    
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        print(f"  -> Converted '{date_col}' to datetime format.")
        
        # Extract Month and Year for Trend Analysis
        df['YearMonth'] = df[date_col].dt.to_period('M')
        df['Month'] = df[date_col].dt.month
        df['Year'] = df[date_col].dt.year
    else:
        print("  -> No typical Date column found for datetime conversion.")

    # Standardize column naming just in case (uppercase for consistency in typical Kaggle datasets)
    # df.columns = [col.upper() for col in df.columns]

    # ==========================================
    # 3. EXPLORATORY DATA ANALYSIS (EDA)
    # ==========================================
    print("\n[3/4] Performing Exploratory Data Analysis...")
    
    # Define primary metric columns, failing over to common alternatives
    sales_col = next((c for c in ['SALES', 'Sales', 'sales', 'Revenue'] if c in df.columns), None)
    region_col = next((c for c in ['TERRITORY', 'REGION', 'Country', 'COUNTRY'] if c in df.columns), None)
    product_col = next((c for c in ['PRODUCTLINE', 'Category', 'CATEGORY'] if c in df.columns), None)
    item_col = next((c for c in ['PRODUCTCODE', 'Product', 'PRODUCTNAME'] if c in df.columns), None)

    if not sales_col:
        print("[ERROR] Could not find a 'Sales' column. EDA cannot proceed.")
        return

    # Total sales by region
    if region_col:
        sales_by_region = df.groupby(region_col)[sales_col].sum().sort_values(ascending=False)
        print("\n  [CHART] Total Sales by Region:")
        print(sales_by_region.head())

    # Total sales by product category
    if product_col:
        sales_by_category = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False)
        print("\n  [CHART] Total Sales by Product Category:")
        print(sales_by_category.head())
        
    # Top 10 products by revenue
    if item_col:
        top_products = df.groupby(item_col)[sales_col].sum().sort_values(ascending=False).head(10)
        print("\n  [CHART] Top 10 Products by Revenue:")
        print(top_products)

    # ==========================================
    # 4. DATA VISUALIZATION
    # ==========================================
    print("\n[4/4] Generating Visualizations...")
    
    # Create an output directory for plots
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # 4.1 Total Sales by Region (Bar Chart)
    if region_col:
        plt.subplot(2, 2, 1)
        sns.barplot(x=sales_by_region.values, y=sales_by_region.index, hue=sales_by_region.index, palette='viridis', legend=False)
        plt.title('Total Sales by Region')
        plt.xlabel('Total Sales ($)')
        plt.ylabel('Region')

    # 4.2 Total Sales by Product Category (Pie Chart)
    if product_col:
        plt.subplot(2, 2, 2)
        plt.pie(sales_by_category.values, labels=sales_by_category.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title('Sales Distribution by Product Category')

    # 4.3 Monthly Sales Trends (Line Chart)
    if date_col and 'YearMonth' in df.columns:
        plt.subplot(2, 2, 3)
        monthly_sales = df.groupby('YearMonth')[sales_col].sum()
        # Convert period index back to string for plotting
        monthly_sales.index = monthly_sales.index.astype(str)
        
        plt.plot(monthly_sales.index, monthly_sales.values, marker='o', color='b', linestyle='-', linewidth=2)
        plt.title('Monthly Sales Trend')
        plt.xlabel('Month-Year')
        plt.ylabel('Total Sales ($)')
        plt.xticks(rotation=45)
        plt.grid(True)

    # 4.4 Top 10 Products by Revenue (Bar Chart)
    if item_col:
        plt.subplot(2, 2, 4)
        sns.barplot(x=top_products.values, y=top_products.index, hue=top_products.index, palette='magma', legend=False)
        plt.title('Top 10 Products by Revenue')
        plt.xlabel('Revenue ($)')
        plt.ylabel('Product')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "sales_dashboard.png")
    plt.savefig(plot_path)
    print(f"  -> Dashboard saved to '{plot_path}'")
    
    # Show plot interactively if run in an environment that supports it
    # plt.show() 

    # ==========================================
    # 5. BUSINESS INSIGHTS
    # ==========================================
    print("\n" + "="*50)
    print("[INSIGHTS] BUSINESS INSIGHTS SUMMARY")
    print("="*50)
    
    insights = """
    Based on typical sales data findings, here is an executive summary:
    
    1. Regional Performance:
       - Identifying the top-performing region allows targeted marketing to maintain dominance.
       - Underperforming regions might require specialized campaigns, better logistics, or deeper market research to uncover blockages.
       
    2. Product Category Dominance:
       - The pie chart highlights our core revenue drivers. 
       - Heavy reliance on a single category suggests a need for diversification to mitigate risk.
       
    3. Seasonal Trends:
       - The monthly sales line chart usually reveals seasonal peaks (e.g., Q4 holidays).
       - Inventory planning, staffing, and promotional spend should be strictly aligned with these established peaks to maximize margins.
       
    4. Top 10 Products Focus:
       - A small fraction of products often generates a large portion of revenue (Pareto Principle / 80-20 rule).
       - Ensure these top 10 items NEVER run out of stock and consider bundling them with lower-performing items to drive overall basket size.
    """
    print(insights)
    print("-" * 50)
    print("Done! [FINISH]")

if __name__ == "__main__":
    main()
