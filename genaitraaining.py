import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class RetailAnalysis:
    """
    A class to analyze retail sales data with comprehensive cleaning and visualization
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.clean_df = None
        # Set style for better visibility in compact layout
        sns.set_style("whitegrid")
        plt.rcParams['figure.autolayout'] = True

    def load_data(self) -> pd.DataFrame:
        """
        Load and perform initial data exploration
        
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        try:
            self.df = pd.read_csv(self.file_path)
            print("\nInitial Data Overview:")
            print(f"Shape: {self.df.shape}")
            print("\nData Info:")
            print(self.df.info())
            print("\nDescriptive Statistics:")
            print(self.df.describe())

            print("\n=== Data Diagnostics ===")
            print(f"Number of rows: {len(self.df)}")
            print(f"Number of columns: {len(self.df.columns)}")
            print("\nColumns in the dataset:")
            print(self.df.columns.tolist())
            
            print("\nSample of Invoice Dates:")
            print(self.df['Invoice Date'].head())
            
            # Convert Invoice Date to datetime
            self.df['Invoice Date'] = pd.to_datetime(self.df['Invoice Date'])
            
            # Add Quarter column
            self.df['Quarter'] = self.df['Invoice Date'].dt.to_period('Q')
            
            print("\nUnique Quarters in data:")
            print(self.df['Quarter'].unique())
            
            print("\nValue counts for Quarters:")
            print(self.df['Quarter'].value_counts())
            
            print("\nUnique Products:")
            print(self.df['Product'].unique())
            
            print("\nUnique Regions:")
            print(self.df['Region'].unique())

            return self.df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def clean_data(self) -> pd.DataFrame:
        """
        Clean the data by handling missing values, outliers, and data type conversions
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        try:
            self.clean_df = self.df.copy()
            
            # Convert date column
            self.clean_df['Invoice Date'] = pd.to_datetime(self.clean_df['Invoice Date'])
            
            # Add derived time columns
            self.clean_df['Year'] = self.clean_df['Invoice Date'].dt.year
            self.clean_df['Quarter'] = self.clean_df['Invoice Date'].dt.quarter
            self.clean_df['Month'] = self.clean_df['Invoice Date'].dt.month
            
            # Check for missing values
            missing_values = self.clean_df.isnull().sum()
            if missing_values.sum() > 0:
                print("\nMissing Values:")
                print(missing_values[missing_values > 0])
                
                # Handle missing values
                numeric_columns = self.clean_df.select_dtypes(include=[np.number]).columns
                self.clean_df[numeric_columns] = self.clean_df[numeric_columns].fillna(
                    self.clean_df[numeric_columns].mean()
                )
                
                categorical_columns = self.clean_df.select_dtypes(include=['object']).columns
                self.clean_df[categorical_columns] = self.clean_df[categorical_columns].fillna('Unknown')
            
            def handle_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Identify outliers
                outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                print(f"Number of identified outliers in the column {column}: {outliers}")

                # Clip outliers
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
                return df

            numeric_columns = ['Total Sales (INR)', 'Operating Profit (INR)', 'Price per Unit (INR)']
            for column in numeric_columns:
                self.clean_df = handle_outliers(self.clean_df, column)
            
            return self.clean_df
            
        except Exception as e:
            print(f"Error cleaning data: {str(e)}")
            raise

    def create_visualizations(self):
        """
        Create visualizations in a compact, fitted layout
        """
        try:
            # Create figure with a reasonable size for most screens
            fig = plt.figure(figsize=(15, 18))  # Adjusted size for better fit
            
            # Create grid with better spacing
            gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.4)
            
            # Adjust font sizes for better readability in compact layout
            TITLE_SIZE = 10
            LABEL_SIZE = 8
            TICK_SIZE = 8
            
            # 1. Total Sales by Product Category
            ax1 = fig.add_subplot(gs[0, 0])
            product_sales = self.clean_df.groupby('Product')['Total Sales (INR)'].sum().sort_values(ascending=True)
            sns.barplot(x=product_sales.values, y=product_sales.index, ax=ax1)
            ax1.set_title('Total Sales by Product Category', fontsize=TITLE_SIZE)
            ax1.set_xlabel('Total Sales (INR)', fontsize=LABEL_SIZE)
            ax1.set_ylabel('Product Category', fontsize=LABEL_SIZE)
            ax1.tick_params(labelsize=TICK_SIZE)
            
            # 2. Sales Method Distribution
            ax2 = fig.add_subplot(gs[0, 1])
            sales_method = self.clean_df['Sales Method'].value_counts()
            colors = sns.color_palette("pastel")[0:len(sales_method)]
            wedges, texts, autotexts = ax2.pie(sales_method.values, labels=sales_method.index, 
                                             autopct='%1.1f%%', colors=colors)
            ax2.set_title('Sales Method Distribution', fontsize=TITLE_SIZE)
            plt.setp(autotexts, size=TICK_SIZE)
            plt.setp(texts, size=TICK_SIZE)
            
            # 3. Region-wise Total Sales
            ax3 = fig.add_subplot(gs[1, 0])
            region_sales = self.clean_df.groupby('Region')['Total Sales (INR)'].sum().sort_values(ascending=True)
            sns.barplot(x=region_sales.values, y=region_sales.index, palette="Set3", ax=ax3)
            ax3.set_title('Region-wise Total Sales', fontsize=TITLE_SIZE)
            ax3.set_xlabel('Total Sales (INR)', fontsize=LABEL_SIZE)
            ax3.set_ylabel('Region', fontsize=LABEL_SIZE)
            ax3.tick_params(labelsize=TICK_SIZE)
            
            # 4. Quarterly Sales Trend by Product
            ax4 = fig.add_subplot(gs[1, 1])
            quarterly_product_sales = self.clean_df.groupby(['Quarter', 'Product'])['Total Sales (INR)'].sum().unstack()
            for column in quarterly_product_sales.columns:
                ax4.plot(quarterly_product_sales.index, quarterly_product_sales[column], 
                        marker='o', label=column, linewidth=2)
            ax4.set_title('Quarterly Sales Trend by Product', fontsize=TITLE_SIZE)
            ax4.set_xlabel('Quarter', fontsize=LABEL_SIZE)
            ax4.set_ylabel('Total Sales (INR)', fontsize=LABEL_SIZE)
            ax4.legend(title='Product', bbox_to_anchor=(1.02, 1), loc='upper left', 
                      fontsize=TICK_SIZE-1, title_fontsize=TICK_SIZE)
            ax4.tick_params(labelsize=TICK_SIZE)
            ax4.grid(True, alpha=0.3)
            
            # 5. Quarterly Sales Distribution (Modified to show Units Sold)
            ax5 = fig.add_subplot(gs[2, 0])
            sns.violinplot(data=self.clean_df, x='Quarter', y='Units Sold', palette="Set2", ax=ax5)
            ax5.set_title('Quarterly Units Sold Distribution', fontsize=TITLE_SIZE)
            ax5.set_xlabel('Quarter', fontsize=LABEL_SIZE)
            ax5.set_ylabel('Units Sold', fontsize=LABEL_SIZE)
            ax5.tick_params(labelsize=TICK_SIZE)
            
            # 6. Quarterly Sales Trend by Region
            ax6 = fig.add_subplot(gs[2, 1])
            quarterly_region_sales = self.clean_df.groupby(['Quarter', 'Region'])['Total Sales (INR)'].sum().unstack()
            for column in quarterly_region_sales.columns:
                ax6.plot(quarterly_region_sales.index, quarterly_region_sales[column], 
                        marker='o', label=column, linewidth=2)
            ax6.set_title('Quarterly Sales Trend by Region', fontsize=TITLE_SIZE)
            ax6.set_xlabel('Quarter', fontsize=LABEL_SIZE)
            ax6.set_ylabel('Total Sales (INR)', fontsize=LABEL_SIZE)
            ax6.legend(title='Region', bbox_to_anchor=(1.02, 1), loc='upper left', 
                      fontsize=TICK_SIZE-1, title_fontsize=TICK_SIZE)
            ax6.tick_params(labelsize=TICK_SIZE)
            ax6.grid(True, alpha=0.3)
            
            # 7. Profit Margin Analysis
            ax7 = fig.add_subplot(gs[3, 0])
            self.clean_df['Profit Margin'] = (self.clean_df['Operating Profit (INR)'] / 
                                            self.clean_df['Total Sales (INR)'] * 100)
            profit_margins = self.clean_df.groupby('Product')['Profit Margin'].mean().sort_values(ascending=True)
            sns.barplot(x=profit_margins.values, y=profit_margins.index, palette="RdYlBu", ax=ax7)
            ax7.set_title('Average Profit Margin by Product', fontsize=TITLE_SIZE)
            ax7.set_xlabel('Profit Margin (%)', fontsize=LABEL_SIZE)
            ax7.set_ylabel('Product', fontsize=LABEL_SIZE)
            ax7.tick_params(labelsize=TICK_SIZE)
            
            # 8. State-wise Sales Analysis
            ax8 = fig.add_subplot(gs[3, 1])
            state_sales = self.clean_df.groupby('State')['Total Sales (INR)'].sum().sort_values(ascending=True)
            sns.barplot(x=state_sales.values, y=state_sales.index, palette="viridis", ax=ax8)
            ax8.set_title('State-wise Total Sales', fontsize=TITLE_SIZE)
            ax8.set_xlabel('Total Sales (INR)', fontsize=LABEL_SIZE)
            ax8.set_ylabel('State', fontsize=LABEL_SIZE)
            ax8.tick_params(labelsize=TICK_SIZE)
            
            # Add a main title
            fig.suptitle('Retail Sales Analysis Dashboard', 
                        fontsize=TITLE_SIZE+2, 
                        y=0.95)
            
            # Adjust layout to fit the screen
            plt.tight_layout()
            
            # Show the plot
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
            raise

    def generate_insights(self) -> Dict:
        """
        Generate key business insights from the data
        
        Returns:
            Dict: Dictionary containing key insights
        """
        try:
            insights = {
                'total_sales': self.clean_df['Total Sales (INR)'].sum(),
                'total_profit': self.clean_df['Operating Profit (INR)'].sum(),
                'avg_profit_margin': (self.clean_df['Operating Profit (INR)'].sum() / 
                                    self.clean_df['Total Sales (INR)'].sum() * 100),
                'top_product': self.clean_df.groupby('Product')['Total Sales (INR)'].sum().idxmax(),
                'top_region': self.clean_df.groupby('Region')['Total Sales (INR)'].sum().idxmax(),
                'top_state': self.clean_df.groupby('State')['Total Sales (INR)'].sum().idxmax(),
                'total_orders': len(self.clean_df),
                'avg_order_value': self.clean_df['Total Sales (INR)'].mean(),
                'best_performing_month': self.clean_df.groupby('Month')['Total Sales (INR)'].sum().idxmax(),
                'sales_method_split': self.clean_df['Sales Method'].value_counts().to_dict()
            }
            
            # Format currency values for better readability
            def format_currency(value):
                return f"₹{value:,.2f}"
            
            print("\n=== Key Business Insights ===")
            print(f"\nSales Metrics:")
            print(f"Total Sales: {format_currency(insights['total_sales'])}")
            print(f"Total Profit: {format_currency(insights['total_profit'])}")
            print(f"Average Profit Margin: {insights['avg_profit_margin']:.2f}%")
            print(f"Average Order Value: {format_currency(insights['avg_order_value'])}")
            
            print(f"\nTop Performers:")
            print(f"Top Product Category: {insights['top_product']}")
            print(f"Top Region: {insights['top_region']}")
            print(f"Top State: {insights['top_state']}")
            print(f"Best Performing Month: {insights['best_performing_month']}")
            
            print(f"\nOperational Metrics:")
            print(f"Total Orders Processed: {insights['total_orders']:,}")
            print("\nSales Method Distribution:")
            for method, count in insights['sales_method_split'].items():
                print(f"- {method}: {count:,} orders ({count/insights['total_orders']*100:.1f}%)")
            
            return insights
            
        except Exception as e:
            print(f"Error generating insights: {str(e)}")
            raise

def main():
    """
    Main function to execute the retail sales analysis
    """
    try:
        # Initialize the analysis
        analysis = RetailAnalysis('retail_sales.csv')
        
        # Load and explore data
        analysis.load_data()
        
        # Clean data
        analysis.clean_data()
        
        # Generate insights
        analysis.generate_insights()

        # Create visualizations
        analysis.create_visualizations()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
