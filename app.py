import pandas as pd
import plotly.express as px
import streamlit as st

# Page configuration
st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")

# ---- READ EXCEL ----
@st.cache_data
def get_data_from_excel(file):
    df = pd.read_excel(
        io=file,
        engine="openpyxl",
        sheet_name="Sales",
        skiprows=3,
        usecols="B:R",
        nrows=1000,
    )
    # Add 'hour' column to dataframe
    df["hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
    return df

# ---- SIDEBAR ----
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to:",
    options=["Upload File", "Dataset", "Forecasted"]
)

# ---- FILE UPLOAD SECTION ----
if page == "Upload File":
    st.title(":file_folder: Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])

    if uploaded_file is not None:
        df = get_data_from_excel(uploaded_file)
        st.success("File uploaded and data loaded successfully!")

        # Show a sample of the uploaded dataset
        st.write("### Dataset Preview")
        st.write(df.head())

        st.sidebar.success("File uploaded successfully. Go to 'Dataset' to explore the data.")
        st.stop()

# ---- FILTERING SECTION (For Dataset Page) ----
if page == "Dataset":
    if 'df' not in locals():
        st.warning("Please upload a file first.")
        st.stop()

    st.sidebar.header("Please Filter Here:")
    city = st.sidebar.multiselect(
        "Select the City:",
        options=df["City"].unique(),
        default=df["City"].unique()
    )

    customer_type = st.sidebar.multiselect(
        "Select the Customer Type:",
        options=df["Customer_type"].unique(),
        default=df["Customer_type"].unique(),
    )

    gender = st.sidebar.multiselect(
        "Select the Gender:",
        options=df["Gender"].unique(),
        default=df["Gender"].unique()
    )

    df_selection = df.query(
        "City == @city & Customer_type ==@customer_type & Gender == @gender"
    )

    # Check if the dataframe is empty:
    if df_selection.empty:
        st.warning("No data available based on the current filter settings!")
        st.stop()  # This will halt the app from further execution.

# ---- MAINPAGE FOR DATASET ----
if page == "Dataset":
    st.title(":bar_chart: Sales Dashboard")
    st.markdown("##")

    # TOP KPI's
    total_sales = int(df_selection["Total"].sum())
    average_rating = round(df_selection["Rating"].mean(), 1)
    star_rating = ":star:" * int(round(average_rating, 0))
    average_sale_by_transaction = round(df_selection["Total"].mean(), 2)

    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Total Sales:")
        st.subheader(f"US $ {total_sales:,}")
    with middle_column:
        st.subheader("Average Rating:")
        st.subheader(f"{average_rating} {star_rating}")
    with right_column:
        st.subheader("Average Sales Per Transaction:")
        st.subheader(f"US $ {average_sale_by_transaction}")

    st.markdown("""---""")

    # SALES BY PRODUCT LINE [BAR CHART]
    sales_by_product_line = df_selection.groupby(by=["Product line"])[["Total"]].sum().sort_values(by="Total")
    fig_product_sales = px.bar(
        sales_by_product_line,
        x="Total",
        y=sales_by_product_line.index,
        orientation="h",
        title="<b>Sales by Product Line</b>",
        color_discrete_sequence=["#0083B8"] * len(sales_by_product_line),
        template="plotly_white",
    )
    fig_product_sales.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # SALES BY HOUR [BAR CHART]
    sales_by_hour = df_selection.groupby(by=["hour"])[["Total"]].sum()
    fig_hourly_sales = px.bar(
        sales_by_hour,
        x=sales_by_hour.index,
        y="Total",
        title="<b>Sales by hour</b>",
        color_discrete_sequence=["#0083B8"] * len(sales_by_hour),
        template="plotly_white",
    )
    fig_hourly_sales.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),
    )

    left_column, right_column = st.columns(2)
    left_column.plotly_chart(fig_hourly_sales, use_container_width=True)
    right_column.plotly_chart(fig_product_sales, use_container_width=True)

# ---- MAINPAGE FOR FORECASTED ----
if page == "Forecasted":
    st.title(":chart_with_upwards_trend: Forecasted Sales")
    st.markdown("This page is reserved for forecasting data.")
    # Placeholder for future forecasting visualization or analysis
    st.write("Here you can add sales forecasting visualizations and analytics.")
    
# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
