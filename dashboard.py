import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

sns.set(style='dark')

# Helper function yang dibutuhkan untuk menyiapkan berbagai dataFrame

def create_daily_orders_df(df):
    daily_orders_df = df.resample(rule='D', on='order_purchase_timestamp').agg({
        "order_id":"nunique",
        "payment_value":"sum"
    })
    daily_orders_df = daily_orders_df.reset_index()
    daily_orders_df.rename(columns={
        "order_id":"order_count",
        "payment_value":"revenue"
    },inplace=True)

    return daily_orders_df

def create_sum_order_item_df(df):
    sum_order_item_df = order_item_product_payment_df.groupby("product_name").order_id.count().reset_index()
    return sum_order_item_df 

def create_sum_order_revenue_df(df):
    sum_order_revenue_df = order_item_product_payment_df.groupby("product_name").payment_value.sum().reset_index()
    return sum_order_revenue_df 

def create_customer_city_df(df):
    customer_city_df = customer_df.groupby("customer_city").customer_id.count().reset_index()
    return customer_city_df

def create_customer_state_df(df):
    customer_state_df = customer_df.groupby("customer_state").customer_id.count().reset_index()
    return customer_state_df

def create_seller_city_df(df):
    seller_city_df = seller_df.groupby("seller_city").seller_id.count().reset_index()
    return seller_city_df

def create_seller_state_df(df):
    seller_state_df = seller_df.groupby("seller_state").seller_id.count().reset_index()
    return seller_state_df

def create_customer_review_df(df):
    customer_review_df = all_df.groupby("review_score").customer_id.count().reset_index()
    customer_review_df.rename(columns={
    "customer_id": "customer_count"
    }, inplace=True)
    return customer_review_df 

def create_review_product_df(df):
    review_product_df = all_df.groupby("review_score").product_id.count().reset_index()
    review_product_df.rename(columns={
    "product_id": "product_count"
    }, inplace=True)
    return review_product_df

def create_seller_order_count_df(df):
    seller_order_count_df = all_df.groupby("seller_id").order_id.count().reset_index()
    seller_order_count_df.rename(columns={
    "order_id": "order_count"
    }, inplace=True)
    return seller_order_count_df

def create_seller_revenue_df(df):
    seller_revenue_df = all_df.groupby("seller_id").payment_value.sum().reset_index()
    seller_revenue_df.rename(columns={
    "payment_value": "revenue"
    }, inplace=True)
    return seller_revenue_df

def create_product_review_df(df):
    product_review_df = all_df.groupby("product_name").review_score.mean().reset_index()
    product_review_df.rename(columns={
    "review_score": "review_mean"
    }, inplace=True)
    return product_review_df

def create_rfm_df(df):
    rfm_df = all_df.groupby(by="customer_id", as_index=False).agg({
    "order_purchase_timestamp": "max",
    "order_id": "nunique",
    "payment_value": "sum"
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    rfm_df["max_order_timestamp"] = pd.to_datetime(rfm_df["max_order_timestamp"])
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date

    recent_date = order_df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df = rfm_df.drop(columns=["max_order_timestamp"])
    return rfm_df

# Load data
customer_df = pd.read_csv("customer.csv")
seller_df = pd.read_csv("seller.csv")
order_df = pd.read_csv("order.csv")
item_df = pd.read_csv("item.csv")
product_df = pd.read_csv("product.csv")
payment_df = pd.read_csv("payment.csv")
review_df = pd.read_csv("review.csv")

# load gabungan data
item_product_df = pd.merge(
    left=item_df,
    right=product_df,
    how="left",
    left_on="product_id",
    right_on="product_id"
)
order_item_product_df = pd.merge(
    left=order_df,
    right=item_product_df,
    how="left",
    left_on="order_id",
    right_on="order_id"
)
order_item_product_payment_df = pd.merge(
    left=order_item_product_df,
    right=payment_df,
    how="left",
    left_on="order_id",
    right_on="order_id"
)
all_df = pd.merge(
    left=order_item_product_payment_df,
    right=review_df,
    how="left",
    left_on="order_id",
    right_on="order_id"
)


# Mengubah kolom objek menjadi datetime
datetime_columns = ["order_purchase_timestamp"]
all_df.sort_values(by="order_purchase_timestamp",inplace=True)
all_df.reset_index(inplace=True)

for column in datetime_columns:
    all_df["order_purchase_timestamp"]=pd.to_datetime(all_df["order_purchase_timestamp"])
for column in datetime_columns:
    order_df["order_purchase_timestamp"]=pd.to_datetime(order_df["order_purchase_timestamp"])

#Filter data
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

with st.sidebar:
    #menambahkan logo
    #mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label = 'Rentang Waktu', min_value=min_date,
        max_value=max_date,
        value=[min_date,max_date]
    )
main_df = all_df[(all_df["order_purchase_timestamp"]>=str(start_date))&
                 (all_df["order_purchase_timestamp"]<= str(end_date))]

# st.dataframe(main_df)
# # menyiapkan berbagai dataframe
daily_orders_df = create_daily_orders_df(main_df)
sum_order_item_df = create_sum_order_item_df(main_df)
sum_order_revenue_df = create_sum_order_revenue_df(main_df)
customer_city_df = create_customer_city_df(main_df)
customer_state_df = create_customer_state_df(main_df)
seller_city_df = create_seller_city_df(main_df)
seller_state_df = create_seller_state_df(main_df)
customer_review_df = create_customer_review_df(main_df)
review_product_df = create_review_product_df(main_df)
seller_order_count_df = create_seller_order_count_df(main_df)
seller_revenue_df = create_seller_revenue_df(main_df)
product_review_df = create_product_review_df(main_df)
rfm_df = create_rfm_df(main_df)

#plot number of daily orders (2016-2018)
st.header('E-Commerce')
st.subheader('Daily Orders')

col1,col2 = st.columns(2)

with col1:
    total_orders = daily_orders_df.order_count.sum()
    st.metric("Total Orders",value=total_orders)

with col2:
    total_revenue = format_currency(daily_orders_df.revenue.sum(),"BRL")
    st.metric("Total Revenue",value=total_revenue)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    daily_orders_df["order_purchase_timestamp"],
    daily_orders_df["order_count"], 
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)

st.pyplot(fig)

# Product performance
st.subheader ("Produk dengan Penjualan Terbaik dan Terburuk")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))
 
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
 
sns.barplot(x="order_id", y="product_name", data=sum_order_item_df.sort_values(by="order_id",ascending=False).head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Banyak Order",fontsize=30)
ax[0].set_title("Produk dengan Penjualan Terbaik", loc="center", fontsize=50)
ax[0].tick_params(axis ='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)
 
sns.barplot(x="order_id", y="product_name", data=sum_order_item_df.sort_values(by="order_id", ascending=True).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Banyak Order", fontsize=30)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Produk dengan Penjualan Terburuk", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)

st.pyplot(fig)

# Produk dengan Best & Worst Revenue
st.subheader("Produk dengan Revenue Terbaik dan Terburuk")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))
 
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
 
sns.barplot(x="payment_value", y="product_name", data=sum_order_revenue_df.sort_values(by="payment_value",ascending=False).head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Revenue (BRL)", fontsize=30)
ax[0].set_title("Produk dengan Penjualan Terbaik", loc="center", fontsize=50)
ax[0].tick_params(axis ='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)
 
sns.barplot(x="payment_value", y="product_name", data=sum_order_revenue_df.sort_values(by="payment_value", ascending=True).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Revenue (RBL)",fontsize=30)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Produk dengan Penjualan Terburuk", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)

st.pyplot(fig)

# Demografi Customer dan Seller
st.subheader("Top 5 Customer City & State")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))
 
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(x="customer_id", y="customer_city", data=customer_city_df.sort_values(by="customer_id", ascending=False).head(5), palette=colors, ax=ax[0])

ax[0].set_ylabel(None)
ax[0].set_xlabel("Jumlah Customer",fontsize=30)
ax[0].set_title("Top 5 Customer City", loc="center", fontsize=50)
ax[0].tick_params(axis ='y', labelsize=35)
ax[0].tick_params(axis='x',labelsize=30)
 
sns.barplot(x="customer_id", y="customer_state", data=customer_state_df.sort_values(by="customer_id", ascending=False).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Jumlah Customer",fontsize=30)
ax[1].set_title("Top 5 Customer State", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)
 
st.pyplot(fig)

# Top 5 Seller City & state
st.subheader("Top 5 Seller City & State")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))
 
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
sns.barplot(x="seller_id", y="seller_city", data=seller_city_df.sort_values(by="seller_id", ascending=False).head(5), palette=colors, ax=ax[0])

ax[0].set_ylabel(None)
ax[0].set_xlabel("Jumlah Seller",fontsize=30)
ax[0].set_title("Top 5 Seller City", loc="center", fontsize=50)
ax[0].tick_params(axis ='y', labelsize=35)
ax[0].tick_params(axis='x',labelsize=30)
 
sns.barplot(x="seller_id", y="seller_state", data=seller_state_df.sort_values(by="seller_id", ascending=False).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Jumlah Seller",fontsize=30)
ax[1].set_title("Top 5 Seller State", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)

st.pyplot(fig)

# Tingkat kepuasan Customer
st.subheader("Tingkat Kepuasan Customer")
fig, ax = plt.subplots(figsize=(12, 6))

colors = ["#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3","#72BCD4"]

sns.barplot(x="review_score", y="customer_count", data=customer_review_df.sort_values(by="customer_count", ascending=False), palette=colors, ax=ax)
ax.set_ylabel(None)
ax.set_xlabel("Review Score", fontsize=12)
ax.set_title("Tingkat Kepuasan Customer", loc="center", fontsize=15)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)
st.pyplot(fig)

#Review Score Product
st.subheader("Review Score Produk")
fig, ax = plt.subplots(figsize=(12, 6))

colors = ["#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3","#72BCD4"]

sns.barplot(x="review_score", y="product_count", data=review_product_df.sort_values(by="product_count", ascending=False), palette=colors, ax=ax)
ax.set_ylabel(None)
ax.set_xlabel("Review Score", fontsize=12)
ax.set_title("Review Score Product", loc="center", fontsize=15)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', labelsize=12)
st.pyplot(fig)

# Seller dengan penjualan terbaik dan terburuk
st.subheader("Seller dengan Penjualan Terbaik & Terburuk")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24,6))
 
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
 
sns.barplot(x="order_count", y="seller_id", data=seller_order_count_df.sort_values(by="order_count",ascending=False).head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Jumlah Penjualan",fontsize=12)
ax[0].set_title("Seller dengan Penjualan Terbaik", loc="center", fontsize=15)
ax[0].tick_params(axis ='y', labelsize=12)
ax[0].tick_params(axis='x', labelsize=12)
 
sns.barplot(x="order_count", y="seller_id", data=seller_order_count_df.sort_values(by="order_count", ascending=True).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Jumlah Penjualan",fontsize=12)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Seller dengan Penjualan Terburuk", loc="center", fontsize=15)
ax[1].tick_params(axis='y', labelsize=12)
ax[1].tick_params(axis='x', labelsize=12)
st.pyplot(fig)

#Seller revenue best & worst
st.subheader("Seller dengan Revenue Terbaik dan Terburuk")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))
 
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
 
sns.barplot(x="revenue", y="seller_id", data=seller_revenue_df.sort_values(by="revenue",ascending=False).head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Revenue",fontsize=12)
ax[0].set_title("Seller dengan Revenue Terbaik", loc="center", fontsize=15)
ax[0].tick_params(axis ='y', labelsize=12)
ax[0].tick_params(axis='x', labelsize=12)
 
sns.barplot(x="revenue", y="seller_id", data=seller_revenue_df.sort_values(by="revenue", ascending=True).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Revenue", fontsize=12)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Seller dengan Revenue Terburuk", loc="center", fontsize=15)
ax[1].tick_params(axis='y', labelsize=12)
ax[1].tick_params(axis='x', labelsize=12)
st.pyplot(fig)

# produk dengan rata-rata review terbaik dan terburuk
st.subheader("Produk dengan Review Terbaik dan Terburuk")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))
 
colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
 
sns.barplot(x="review_mean", y="product_name", data=product_review_df.sort_values(by="review_mean",ascending=False).head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Rata-Rata Review Score", fontsize=30)
ax[0].set_title("Produk dengan Review Score Terbaik", loc="center", fontsize=50)
ax[0].tick_params(axis ='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)
 
sns.barplot(x="review_mean", y="product_name", data=product_review_df.sort_values(by="review_mean", ascending=True).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Rata-Rata Review Score",fontsize=30)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Produk dengan Review Score Terburuk", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)
st.pyplot(fig)

# RFM ANALYSIS
st.subheader("RFM Analysis")
# Rank Customer
rfm_df['r_rank'] = rfm_df['recency'].rank(ascending=False)
rfm_df['f_rank'] = rfm_df['frequency'].rank(ascending=True)
rfm_df['m_rank'] = rfm_df['monetary'].rank(ascending=True)

# normalisasi ranking
rfm_df['r_rank_norm'] = (rfm_df['r_rank']/rfm_df['r_rank'].max())*100
rfm_df['f_rank_norm'] = (rfm_df['f_rank']/rfm_df['f_rank'].max())*100
rfm_df['m_rank_norm'] = (rfm_df['m_rank']/rfm_df['m_rank'].max())*100
 
rfm_df.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)

rfm_df['RFM_score'] = 0.15*rfm_df['r_rank_norm']+0.28 * \
    rfm_df['f_rank_norm']+0.57*rfm_df['m_rank_norm']
rfm_df['RFM_score'] *= 0.05
rfm_df = rfm_df.round(2)
rfm_df[['customer_id', 'RFM_score']]

# Segmentasi Customer
rfm_df["customer_segment"] = np.where(
    rfm_df['RFM_score'] > 4.5, "Top customers", (np.where(
        rfm_df['RFM_score'] > 4, "High value customer",(np.where(
            rfm_df['RFM_score'] > 3, "Medium value customer", np.where(
                rfm_df['RFM_score'] > 1.6, 'Low value customers', 'lost customers'))))))

customer_segment_df = rfm_df.groupby(by="customer_segment", as_index=False).customer_id.nunique()
customer_segment_df['customer_segment'] = pd.Categorical(customer_segment_df['customer_segment'], [
    "lost customers", "Low value customers", "Medium value customer",
    "High value customer", "Top customers"
])


fig, ax = plt.subplots(figsize=(16, 8))

colors = ["#D3D3D3", "#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(x="customer_id", y="customer_segment", data=customer_segment_df.sort_values(by="customer_segment", ascending=False), palette=colors, ax=ax)
ax.set_ylabel(None)
ax.set_xlabel("Jumlah Customer", fontsize=16)
ax.set_title("Segmentasi Customer", loc="center", fontsize=30)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='x', labelsize=16)
st.pyplot(fig)

