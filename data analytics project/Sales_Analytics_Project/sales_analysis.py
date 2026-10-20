import pandas as pd

# Load the Excel file
file_path = "sales_customer_analytics_dataset.xlsx"
df = pd.read_excel(file_path, sheet_name="Sales_Data")

# 1. Number of rows and columns
print("Dataset Shape:")
print(df.shape)

# 2. Column names
print("\nColumn Names:")
print(df.columns.tolist())

# 3. Data types and non-null values
print("\nDataset Information:")
df.info()
# Check for duplicate rows
duplicates = df.duplicated().sum()

print("\nDuplicate Rows:")
print(duplicates)

# Check for missing values
missing_values = df.isnull().sum()

print("\nMissing Values:")
print(missing_values)

# Validate Profit calculation
calculated_profit = df["Sales"] - df["Cost"]

profit_errors = (df["Profit"] - calculated_profit).abs() > 0.01

print("\nProfit Calculation Errors:")
print(profit_errors.sum())


# Validate Profit Margin calculation
calculated_margin = (df["Profit"] / df["Sales"]) * 100

margin_errors = (df["Profit_Margin"] - calculated_margin).abs() > 0.01

print("\nProfit Margin Calculation Errors:")
print(margin_errors.sum())

# Check Sales for outliers using the IQR method

Q1 = df["Sales"].quantile(0.25)
Q3 = df["Sales"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - (1.5 * IQR)
upper_limit = Q3 + (1.5 * IQR)

sales_outliers = df[
    (df["Sales"] < lower_limit) |
    (df["Sales"] > upper_limit)
]

print("\nSales Outliers:")
print(len(sales_outliers))

print("\nSales Outlier Limits:")
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)


# Display the 10 highest sales transactions
top_sales = df.nlargest(10, "Sales")

print("\nTop 10 Sales:")
print(top_sales[["Order_ID", "Customer_ID", "Product", "Sales", "Cost", "Profit"]])

# Check Quantity range
print("\nQuantity Statistics:")
print(df["Quantity"].describe())

# Check Unit Price statistics
print("\nUnit Price Statistics:")
print(df["Unit_Price"].describe())

# Check Discount statistics
print("\nDiscount Statistics:")
print(df["Discount"].describe())

# Check Cost and Profit statistics
print("\nCost Statistics:")
print(df["Cost"].describe())

print("\nProfit Statistics:")
print(df["Profit"].describe())

# Count loss-making transactions
loss_making = df[df["Profit"] < 0]

print("\nLoss-Making Transactions:")
print(len(loss_making))

# Display loss-making transactions
loss_making = df[df["Profit"] < 0]

print("\nLoss-Making Transactions:")
print(
    loss_making[
        ["Order_ID", "Category", "Product", "Quantity",
         "Discount", "Sales", "Cost", "Profit", "Profit_Margin"]
    ].sort_values("Profit")
)

print("\nCost Statistics:")
print(df["Cost"].describe())

# Export cleaned dataset
output_file = "sales_customer_analytics_final.csv"
df.to_csv(output_file, index=False)

print("\nCleaned dataset exported successfully!")
print(output_file)
