import pandas as pd

# Step 1:Load the dataset  ───────────────────────────────────────────────────
df = pd.read_csv('online_retail.csv', encoding='latin1')
print("Step 1 ✅ Data Loaded")
print(f"  Shape: {df.shape}") 
 
# # ── Step 2: Inspect the data ───────────────────────────────────────────────────
print("\nStep 2 ✅ First 5 rows:")
print(df.head())
 
print("\nColumn types:")
print(df.dtypes)
 
print("\nMissing values:")
print(df.isnull().sum())

# ── Step 3: Handle missing values ─────────────────────────────────────────────
# CustomerID has 135,080 missing — drop those rows
df = df.dropna(subset=['CustomerID'])
 
# Description has 1,454 missing — fill with 'Unknown'
df['Description'] = df['Description'].fillna('Unknown')

print("\nStep 3 ✅ Missing values handled")
print(f"  Shape after dropping missing CustomerID: {df.shape}")

# ── Step 4:Remove duplicates based on InvoiceNo, StockCode, Quantity, and UnitPrice  ─────────────────────────────────────────────


df = df.drop_duplicates(subset=['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice'])
print("\nStep 4 ✅ Duplicates removed")
print("Removed:", {df.shape})

#  ── Step 5: Fix data types ────────────────────────────────────────────────────
# # Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
 
# # Convert CustomerID to integer (was float due to missing values)
df['CustomerID'] = df['CustomerID'].astype(int)
 
print("\nStep 5 ✅ Data types fixed")
print(df.dtypes)
 
# ── Step 6: Remove invalid rows ───────────────────────────────────────────────

# print("Cancelled invoices:", df['InvoiceNo'].astype(str).str.startswith('C').sum())

# Cancelled invoices start with 'C' in InvoiceNo — remove those rows
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]


# # Negative Quantity = returns/cancellations — remove them
df = df[df['Quantity'] > 0]

# Negative or zero UnitPrice = invalid — remove them
df = df[df['UnitPrice'] > 0]

print(f"\nStep 6 ✅ Invalid rows removed")
print(f"  Shape after removing negatives: {df.shape}")

# ── Step 7: Add useful columns ─────────────────────────────────────────────────
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['Year']       = df['InvoiceDate'].dt.year
df['Month']      = df['InvoiceDate'].dt.month
df['Day']        = df['InvoiceDate'].dt.day

print("\nStep 7 ✅ New columns added: TotalPrice, Year, Month, Day")

# ── Step 8: Clean text columns ────────────────────────────────────────────────
df['Description'] = df['Description'].str.strip().str.upper()
df['Country']     = df['Country'].str.strip()

print("\nStep 8 ✅ Text columns cleaned")

# ── Step 9: Final summary ─────────────────────────────────────────────────────
print("\nStep 9 ✅ Final Dataset Summary")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['InvoiceDate'].min()} → {df['InvoiceDate'].max()}")
print(f"  Unique customers: {df['CustomerID'].nunique()}")
print(f"  Unique products: {df['StockCode'].nunique()}")
print(f"  Total revenue: £{df['TotalPrice'].sum():,.2f}")
print("\nFirst 5 rows of cleaned data:")
print(df.head())

# ── Step 10: Save cleaned data ────────────────────────────────────────────────
df.to_csv('online_retail_cleaned.csv', index=False)
print("\nStep 10 ✅ Cleaned data saved as 'online_retail_cleaned.csv'")