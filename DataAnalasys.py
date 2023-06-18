import pandas
import matplotlib.pyplot as plt
import seaborn as snp
import numpy as np
from matplotlib import ticker

plt.rcParams['font.size'] = 20
pandas.set_option('display.float_format', '{:.2f}'.format)
df = pandas.read_csv("C:/Users/Bartek/Desktop/datasets/train.tsv", sep='\t')
df['brand_name'][df.brand_name.isnull()] = "missing"
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())
df["log_price"] = df.price.apply(lambda x:  np.log(x+1))
print(df.name.describe())
formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)


print(df['price'].describe().apply("{0:.5f}".format))  # Najważniejsze cechy kolumny Price
print("Item condition description:", df['item_condition_id'].describe())
item_counts = df['item_condition_id'].value_counts()
df['category_name'][df.category_name.isnull()] = "missing"
df['Tier_1'] = df.category_name.apply(lambda x:    x.split("/")[0] if len(x.split("/"))>=1 else "missing")
df['Tier_2'] = df.category_name.apply(lambda x:    x.split("/")[1] if len(x.split("/"))>1 else "missing")
df['Tier_3'] = df.category_name.apply(lambda x:    x.split("/")[2] if len(x.split("/"))>1 else "missing")

plt.figure(figsize=(20,15))
df_brand_name_sorted = df[["brand_name", "price"]].groupby(by="brand_name").max()
df_brand_name_sorted = df_brand_name_sorted.sort_values(by="price", ascending=False)
snp.barplot(x=df_brand_name_sorted.index[:10],y= df_brand_name_sorted.price[:10])
plt.title('Top 10 most expensive brand names')
plt.xticks(rotation=45)
plt.show()

#Print the most common categories in each tier with it's count
print(df.Tier_1.value_counts().index[:10])
print(df.Tier_1.value_counts().values[:10])
print(df.Tier_2.value_counts().index[:10])
print(df.Tier_2.value_counts().values[:10])
print(df.Tier_3.value_counts().index[:10])
print(df.Tier_3.value_counts().values[:10])

#Print top 10 most common brand_names
print(df.brand_name.value_counts().index[:10])
print(df.brand_name.value_counts().values[:10])

#Print top 10 most common item_descriptions
print(df.item_description.value_counts().index[:10])
print(df.item_description.value_counts().values[:10]),

#Print shipping distribution
print(df.shipping.value_counts().index)
print(df.shipping.value_counts().values)
#Print item_condition distribution
print(df.item_condition_id.value_counts().index)
print(df.item_condition_id.value_counts().values)

#Plot the item_condition distribution
plt.figure(figsize=(20,15))
snp.barplot(x=df.item_condition_id.value_counts().index, y=df.item_condition_id.value_counts().values)
plt.xlabel("Item_Condition_ID")
plt.title("item_condition distribution")
plt.show()
#Plot the shipping distrubution
plt.figure(figsize=(20,15))
snp.barplot(x=df.shipping.value_counts().index, y=df.shipping.value_counts().values)
plt.xlabel("Shipping")
plt.title("Shipping distribution")
plt.show()

#Number of chars in name
plt.figure(figsize=(20, 15))
plt.hist(df.name.apply(len), color="green")
plt.xlabel("Length of name")
plt.title("Number of chars in name")
plt.show()
#Number of words in name
plt.figure(figsize=(20, 15))
plt.hist(df.name.apply(lambda x: len(x.split())), color = 'blue')
plt.xlabel("Number of words in name")
plt.title("Number of words in item_description")
plt.show()

#Number of words in item_description
plt.figure(figsize=(20,15))
plt.hist(df.item_description.apply(lambda x: len(str(x))), color="blue")
plt.xlabel("Length of description")
plt.title("Number of words in item_description")
plt.show()

#Log price distribution
plt.figure(figsize=(20, 15))
plt.hist(df.log_price,bins=30,color="teal")
plt.title("Log(Price+1) Distribution")
plt.xlabel("log(Price+1)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(20, 15))
plt.hist(df.price, bins=30, label='Price')
plt.title("Price distribution")
plt.xlabel('Price')
plt.ylabel('Amount')
plt.ticklabel_format(style='plain')
plt.show()

#Analyzing mean and median value for Tier1 categories
grouped_df = df.groupby('Tier_1')['price']
plt.figure(figsize=(20,15))
snp.barplot(y=grouped_df['price'].values, x=grouped_df["Tier_1"])
plt.title("Tier1 vs Price")
plt.xlabel("Tier1 category")
plt.xticks(rotation=45)
plt.show()

grouped_df = df.groupby('Tier_1')['price'].mean().reset_index()
plt.figure(figsize=(20,15))
snp.barplot(y=grouped_df['price'].values, x=grouped_df["Tier_1"])
plt.title("Tier1 vs Price")
plt.xlabel("Tier1 category")
plt.xticks(rotation=45)
plt.show()

# Price distribution looked at via shipping
plt.figure(figsize=(15, 10))
bins = 50
plt.hist(df[df['shipping']==1]['price'],bins, density=True, range=[0,250], alpha=0.6, label='Price when the seller paid for the shipping')
plt.hist(df[df['shipping']==0]['price'], bins, density=True, range=[0,250], alpha=0.6, label='Price when the buyer paid for the shipping')
plt.title("Shipping vs Price")
plt.xlabel('Price')
plt.ticklabel_format(style='plain')
plt.show()

#Item condition vs median price
plt.figure(figsize=(20,15))
snp.barplot(x=df.item_condition_id, y=df.price)
plt.title("Item_Condition vs Mean Price")
plt.show()

#Analiza najdroższych brandów w danych trenningowych
plt.figure(figsize=(20,15))
df_brand_name_sorted = df[["brand_name", "price"]].groupby(by="brand_name").max()
df_brand_name_sorted = df_brand_name_sorted.sort_values(by="price", ascending=False)
snp.barplot(x=df_brand_name_sorted.index[:10],y= df_brand_name_sorted.price[:10])
plt.title('Top 10 most expensive brand names')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(20, 15))
df_brand_name_sorted = df[["brand_name", "price"]].groupby(by="brand_name").mean()
df_brand_name_sorted = df_brand_name_sorted.sort_values(by="price", ascending=False)
snp.barplot(x=df_brand_name_sorted.index[:10], y=df_brand_name_sorted.price[:10])
plt.title('Brand_name vs Price')
plt.xticks(rotation=45)
plt.show()

#Shipping vs price boxplot
plt.figure(figsize=(20,15))
snp.boxplot(x='shipping', y='log_price', data=df)
plt.title("Shipping vs log_price")
plt.show()

# Item_condition vs price
my_plot = []
for i in df['item_condition_id'].unique():
    my_plot.append(df[df['item_condition_id']==i]['log_price'])
fig, axes = plt.subplots(figsize=(10, 10))
bp = axes.boxplot(my_plot,vert=True,patch_artist=True,labels=range(1,6))
colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes.yaxis.grid(True)
plt.title('BoxPlot log_price X item_condition_id')
plt.xlabel('item_condition_id')
plt.ylabel('log_price')
plt.show()

#Analyzing mean and median value for Tier1 categories
plt.figure(figsize=(20,15))
snp.barplot(x='Category', y='log_price', data=df)
plt.title("Tier 1 log_price distribution")
plt.show()
plt.figure(figsize=(20,15))
snp.barplot(x='Category', y='Mean price', data=df[["price","Tier_1"]].groupby(by='Tier_1').mean().values)
plt.title("Mean Tier_1 log_price distribution")
plt.show()


# Boxplot Tier1 vs log_price
plt.figure(figsize=(20, 15))
snp.boxplot(x='Tier_1', y='log_price', data=df)
plt.ylabel('Log Price')
plt.xlabel('Tier_1')
plt.title('BoxPlot Log Price by Tier_1')
plt.xticks(rotation=45)
plt.show()

