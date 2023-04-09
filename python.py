import streamlit as st 
import pandas as pd 
import numpy as np 
from PIL import Image
import seaborn as sns 
import matplotlib.pyplot as plt 
st.set_option('deprecation.showPyplotGlobalUse', False)
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from prophet.plot import plot_plotly , plot_components_plotly
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric



df = pd.read_csv("C:/Users/Dell/OneDrive/Desktop/python_ds/oil/DCOILWTICO.csv")
df.rename(columns={"DATE":"date","DCOILWTICO":"price"},inplace=True)
df["Forward_Fill"] = df["price"].ffill()
df.drop("price",axis=1,inplace=True)

st.set_page_config(layout='wide',initial_sidebar_state='collapsed')
st.header("FRED_OIL_PRICE_PREDICTIONS APP")
st.info("Click on the arrow on the top left to view the - About_Me ,  Home , Dataset , Model_performance , Data_Exploration" , icon="ℹ️")

menu = [ 'Home🏡','Dataset📂', 'Data_Exploration📊', 'Decomposing Time Series', 'Testing for Stationarity📈','Model Performance🔍','About Me👨‍💻']

# Create a function to display the content for each menu item
def display_content(choice):
    if choice == 'About Me👨‍💻':
        st.write("About me and connection details")
    elif choice == 'Home':
        st.write("Business Objective and deatils of the dataset")
    elif choice == 'Dataset':
        st.write('Here is the dataset and functions')
    elif choice == 'Model Performance':
        st.write('Here is the model performance')
    elif choice == 'Data_Exploration':
        st.write('Here are the data visualizations')
    elif choice == 'Testing for Stationarity':
        st.write('Stationarity Check')

# Create the sidebar menu
st.sidebar.title('Navigation')
choice = st.sidebar.radio('Select an option', menu)

with st.expander(choice, expanded=True):
    display_content(choice)

if choice == "About Me👨‍💻":
    col1, col2 = st.columns([1, 1])
    with col1:
      st.header("Project Designed by Jagadish_Vemula ")
      st.markdown("""
              ## Data Science Enthusiast
            - 🔭 I’m currently working as an Intern at [AIVariant](https://aivariant.com/)
            - 👀 I’m interested in Data Science, Machine Learning,Artificial Intelligence, Business Intelligence
            - 📫 You can reach me on my email: jagadeesh12337@gmail.com""")
      st.write('Contact Information')
      st.write('📭jagadeesh12337@gmail.com')
      st.write('Connect with me on:')
      st.write('Click On the Website: ')
      st.write( 'Linkdin_id :' 'https://www.linkedin.com/in/jagadish-vemula-599415206/')
      st.write('My WorkSpace')
      st.write( 'Github_id  :'  'https://github.com/vemula2594')
    with col2:
      image = Image.open("C:/Users/Dell/OneDrive/Desktop/python_ds/oil/WhatsApp Image 2023-04-08 at 1.48.57 AM (1).jpeg")
      st.image(image, caption='Jagadish Vemula', use_column_width=True)

if choice == "Home🏡":
   image = Image.open("C:/Users/Dell/OneDrive/Desktop/python_ds/oil/edit1.jpg")
   st.image(image)
   st.write("Objective : Oil is a product that goes completely in a different direction for a single market event as the oil prices are rarely based on real-time data, instead, it is driven by externalities making our attempt to forecast it even more challenging As the economy will be highly affected by oil prices our model will help to understand the pattern in prices to help the customers and businesses to make smart decisions.")
   st.write("Data_set Reference : FRED -Federal Reserve Economic Data")
   st.write('Data Set contains of 9406 rows and 2 columns')
   st.write("Website : https://fred.stlouisfed.org/series/DCOILWTICO#")

if choice == "Dataset📂":
    options = ['Data','Top_10 Records','End_10 Records','Sample' , 'Null_values' , 'Data_types', 'describe' , 'Shape' , 'Columns']
    selected_options = st.selectbox('Select the functions to Display' , options)
    if selected_options == 'Data':
        st.markdown('<span style="color: rgb(255, 165, 0);">Total Data</span>', unsafe_allow_html=True)
        st.table(df)

    elif selected_options == 'Top_10 Records':
        st.markdown("<span style='color: red;'>Displaying 1st 10 records of the dataset</span>", unsafe_allow_html=True)
        st.write(df.head(10))

    elif selected_options == 'End_10 Records':
        st.markdown('<span style="color: #00FF00;">Displaying last 10 records of the dataset</span>', unsafe_allow_html=True)
        st.write(df.tail(10))

    elif selected_options == 'Sample':
        st.markdown('<span style="color: blue;">Sample of the dataset </span>', unsafe_allow_html=True)
        st.write(df.sample(10))

    elif selected_options == 'Null_values': 
        st.markdown('<span style="color: BlueViolet;">Null_Values  of the dataset </span>', unsafe_allow_html=True)
        st.write(df.isnull().sum())

    elif selected_options == 'Data_types':
        st.markdown('<span style="color: Crimson;"> Data_Types of the dataset </span>', unsafe_allow_html=True)
        st.write(df.dtypes)

    elif selected_options == 'describe':
        st.markdown('<span style="color: DarkSlateGrey;">Summary of the dataset </span>', unsafe_allow_html=True)
        st.write(df.describe())

    elif selected_options =='Shape':
        st.markdown('<span style="color: MediumSpringGreen;"> Shape of the dataset </span>', unsafe_allow_html=True)
        st.write(df.shape)

    elif selected_options == 'Columns':
        st.markdown('<span style="color: MediumTurquoise;"> Displaying Columns of the dataset </span>', unsafe_allow_html=True)
        st.write(df.columns)

if choice == 'Data_Exploration📊':
    data1=df.copy()
    data1["date"] = pd.to_datetime(data1.date,format="%m/%d/%Y")
    data1["month"] =data1.date.dt.strftime("%b") # month extraction
    data1["year"] = data1.date.dt.strftime("%Y") # year extraction
    options = ['HeatMap' , 'Barplot' , 'Boxplot' , 'Timeseries plotting', 'lineplot']
    selected_options = st.selectbox('Select the Visualization' , options)


    if selected_options == 'HeatMap':
         # Create a pivot table of mean Forward_Fill values by year and month
        heatmap_y_month = pd.pivot_table(data=data1,values="Forward_Fill",index="year",columns="month",aggfunc="mean",fill_value=0)
        # Create a heatmap using seaborn
        fig, ax = plt.subplots(figsize=(14,15))
        sns.heatmap(heatmap_y_month, annot=True, fmt="g", ax=ax)
        # Display the plot using Streamlit's pyplot function
        st.pyplot(fig)

    if selected_options == 'Barplot':
        plt.figure(figsize=(20,10))
        sns.barplot(x="year",y="Forward_Fill",data=data1)
        plt.title("Forward Fill By Year")
        plt.xlabel("Year")
        plt.ylabel("Forward Fill")
        st.pyplot()

    if selected_options == 'Boxplot':
        # Create sidebar slicer for the year
        selected_year = st.sidebar.selectbox("Select a year", data1["year"].unique())
        # Filter data based on selected year
        data_filtered = data1[data1["year"] == selected_year]
        # Create boxplot
        plt.figure(figsize=(30,10))
        sns.boxplot(x="month",y="Forward_Fill" , data = data_filtered)
        plt.title(f"Boxplot of Forward Fill Prices for {selected_year}")
        st.pyplot()

    if selected_options == 'Timeseries plotting':
        data1['Forward_Fill'].plot(figsize=(22, 10),color='blue')
        plt.title('Time Series Ploting')
        st.pyplot()

    if selected_options == 'lineplot':
        plt.figure(figsize=(25,10))
        sns.lineplot(x="month" , y="Forward_Fill" , data = data1)
        plt.title("lineplot")
        st.pyplot()

if choice == 'Decomposing Time Series':
    data1 = df.copy()
    series = data1.Forward_Fill[:7089]
    result = seasonal_decompose(series, model='multiplicative', period=365)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 8))
    result.observed.plot(ax=ax1, legend=False)
    ax1.set_ylabel('Observed')
    result.trend.plot(ax=ax2, legend=False)
    ax2.set_ylabel('Trend')
    result.seasonal.plot(ax=ax3, legend=False)
    ax3.set_ylabel('Seasonal')
    result.resid.plot(ax=ax4, legend=False)
    ax4.set_ylabel('Residual')
    st.pyplot(fig)

    
    st.subheader(':red[Decomposition of time series is a statistical technique that breaks down a time series into its constituent components, namely trend, seasonality, and noise.]')
    st.write(':green[Observed: usually refers to the actual values of a time series that have been recorded or observed over a period of time. These values are the raw data and are used to analyze and model the behavior of the time series]')
    st.write(':green[Trend: It refers to the underlying long-term pattern or direction of a time series. It represents the overall direction of the series, which could be upward, downward or stable over]')
    st.write(':green[Seasonality: It refers to the periodic fluctuations in a time series that occur at regular intervals. For example, the sales of air conditioners may increase every summer season and decline during winter.]')
    st.write(':green[Noise: It represents the random or unpredictable variations in a time series that cannot be explained by the trend or seasonality]')

if choice == 'Testing for Stationarity📈':
    options = ['ACf &PICF plots','Rolling Mean & Standard Deviation','adfuller_test']
    selected_option = st.radio('Select to view ' , options)
    data1 = df.copy()
    if selected_option == 'ACf &PICF plots':
        plot_acf(data1.Forward_Fill,lags=130)
        st.pyplot()
        plot_pacf(data1.Forward_Fill,lags=130)
        st.pyplot()

    elif selected_option == 'Rolling Mean & Standard Deviation':
        # Determing rolling statistics
        rolmean = data1.Forward_Fill.rolling(window=12).mean()
        rolstd = data1.Forward_Fill.rolling(window=12).std()
        # Plot rolling statistics:
        orig = plt.plot(data1.Forward_Fill, label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        st.pyplot()

    elif selected_option == 'adfuller_test':
        test_result = adfuller(data1['Forward_Fill'])
        #Ho: It is non stationary
        #H1: It is stationary
        def adfuller_test(Forward_Fill):
            result=adfuller(Forward_Fill)
            labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
            for value,label in zip(result,labels):
                st.write(label+' : '+str(value) )
            if result[1] <= 0.05:
                st.write("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
            else:
                st.write("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary - Convert the data into stationary ")
    
        adfuller_test(data1['Forward_Fill'])

if choice == 'Model Performance🔍':
    options = ['Making Data Stationary','Fb_Prophet_Model','Future Predictions of Oil Price🛢️']
    selected_option = st.radio('Select to view ' , options)

    if selected_option == 'Making Data Stationary':
       st.write('Making Data Stationary')
       data1 = df.copy()
       data1['Forward_Fill First Difference'] = data1['Forward_Fill'] - data1['Forward_Fill'].shift(1)
       data1['Seasonal First Difference']=data1['Forward_Fill']-data1['Forward_Fill'].shift(12)
       adf_result = adfuller(data1['Seasonal First Difference'].dropna())
       st.write('ADF Test Results:')
       st.write(f'ADF Test Statistic: {adf_result[0]}')
       st.write(f'p-value: {adf_result[1]}')
       st.write(f'#Lags Used: {adf_result[2]}')
       st.write(f'Number of Observations Used: {adf_result[3]}')
       if adf_result[1] <= 0.05:
           st.write("Strong evidence against the null hypothesis (Ho), reject the null hypothesis. Data has no unit root and is stationary.")
       else:
           st.write("Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary.") 
       
       st.markdown('As p-value: 2.2676454752570256e-19 - Hence the data is concluded as stationary')
       st.markdown(':red[Data Visualization after making the data stationary]')
       data1['Seasonal First Difference'].plot()
       st.pyplot()

    if selected_option == 'Fb_Prophet_Model':
        df.columns = ['ds','y']
        st.write('Showing the sameple record after updating the data columns')
        st.table(df.sample())
        st.write('Fb_Prophet Model')
        model=Prophet()
        model.fit(df)
        model.component_modes
        future_dates=model.make_future_dataframe(periods=7300)
        st.write('Showing the sample future dates of the predictions')
        st.table(future_dates.sample(10))
        st.write('Sample Predictions of the data')
        prediction = model.predict(future_dates)
        st.table(prediction.sample(10))
        st.write('Future predictions graph')
        prediction = model.predict(future_dates)
        model.plot(prediction)
        st.pyplot()
        st.write(':red[Visualize Each Components[Trends,yearly]]')
        plot_plotly(model,prediction)
        model.plot_components(prediction)
        st.pyplot()
        st.write('Visualization of evalution matix using rmse')
        df_cv = cross_validation(model, initial='730 days', period='180 days', horizon = '365 days')
        fig = plot_cross_validation_metric(df_cv, metric='rmse')
        st.pyplot()
        st.write('Visualization of evalution matix using mse')
        fig = plot_cross_validation_metric(df_cv , metric='mse')
        st.pyplot()
        st.write('Visualization of evalution matix using coverage')
        fig = plot_cross_validation_metric(df_cv , metric='coverage')
        st.pyplot()


    if selected_option == 'Future Predictions of Oil Price🛢️':
        with open(r'C:/Users/Dell/OneDrive/Desktop/python_ds/oil/pexels-tima-miroshnichenko-7578613-4096x2160-25fps.mp4' , 'rb') as video_file:
                  video_bytes = video_file.read()
                  st.video(video_bytes)
        df.columns = ['ds', 'y']
        model = Prophet()
        model.fit(df)
        future_dates = model.make_future_dataframe(periods=7300)
        prediction = model.predict(future_dates)
        year_options = prediction['ds'].dt.year.unique()
        selected_year = st.selectbox('Select the year to view the oil price predictions for the next 20 years', year_options)
        selected_year_df = prediction[prediction['ds'].dt.year == selected_year]
        st.write(selected_year_df)
        st.write('You selected the year:', selected_year)
        year_prediction = prediction[prediction['ds'].dt.year == selected_year]
        st.line_chart(year_prediction.set_index('ds')['yhat'])





    














 

          


        
    




