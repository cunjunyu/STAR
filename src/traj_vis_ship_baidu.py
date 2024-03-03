import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
import os
import tqdm
from matplotlib import rcParams

def convert_timestamp(tm):
    d=datetime.datetime.fromtimestamp(tm)
    return d

def get_day(tm):
    d=tm.date()
    # return d.strftime("%Y-%m-%d %H:%M:%S")
    return d


# Set the default font to one that supports Chinese, e.g., 'Microsoft YaHei'
# rcParams['font.family'] = 'Microsoft YaHei'
# rcParams['font.size'] = 10  # You can also set the size of the font if needed
# Define a function to add noise to simulate prediction
def simulate_prediction(real_values):
    noise = np.random.normal(0, np.std(real_values) * 0.05, size=real_values.shape)
    return real_values + noise


# To simulate predicted positions with a smaller deviation, we'll use a smaller noise factor.
def simulate_small_prediction(real_values):
    noise = np.random.normal(0, np.std(real_values) * 0.01, size=real_values.shape)
    return real_values + noise


# Since the goal is to create a smoother transition for the heading similar to a hand-drawn plot,
# we can apply a smoothing function to the latitude and longitude data.
# This function will help in creating a more gradual change in the heading.
def smooth_path(latitudes, longitudes, smoothing_factor=10):
    # Apply a simple moving average to smooth the path
    smooth_lat = latitudes.rolling(window=smoothing_factor, center=True).mean()
    smooth_lon = longitudes.rolling(window=smoothing_factor, center=True).mean()

    return smooth_lat.fillna(method='bfill').fillna(method='ffill'), \
           smooth_lon.fillna(method='bfill').fillna(method='ffill')


# Function to calculate bearing between two points
def calculate_bearing(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Compute the difference in longitudes
    dLon = lon2 - lon1

    # Calculate the bearing
    x = np.sin(dLon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dLon))
    initial_bearing = np.arctan2(x, y)

    # Convert bearing from radians to degrees
    initial_bearing = np.degrees(initial_bearing)

    # Normalize the bearing
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

#  Plot for Heading
def plot_heading(data):
    # Create a line plot for Heading and Speed over time
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Real Heading
    ax1.plot(data['Timestamp'], data['Heading'], label='Real Heading', color='tab:blue')
    # Predicted Heading
    ax1.plot(data['Timestamp'], data['Predicted Heading'], label='Predicted Heading', color='tab:red', linestyle='--')

    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Heading')
    ax1.legend(loc='best')
    #ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    fig1.autofmt_xdate()

    plt.title('Heading: Real vs Predicted')
    plt.tight_layout()
    # 创建路径
    dir_path = os.path.join(base_dir, 'ship' + str(int(data['Ship ID'][0])))
    # Create the directory if it does not exist
    os.makedirs(dir_path, exist_ok=True)
    # Construct the full file path
    heading_plot_path = os.path.join(dir_path, 'heading_real_vs_predicted.png')
    plt.savefig(heading_plot_path)
    plt.close(fig1)  # Close the figure to free memory

# Plot for Speed
def plot_speed(data):
    fig2,ax2 = plt.subplots(figsize=(12, 6))
    # Real Speed
    ax2.plot(data['Timestamp'], data['Speed'], label='Real Speed', color='tab:blue')
    # Predicted Speed
    ax2.plot(data['Timestamp'], data['Predicted Speed'], label='Predicted Speed', color='tab:red', linestyle='--')

    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Speed')
    ax2.legend(loc='best')
    #ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    #ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    fig2.autofmt_xdate()

    plt.title('Speed: Real vs Predicted')
    plt.tight_layout()
    dir_path = os.path.join(base_dir, 'ship' + str(int(data['Ship ID'][0])))
    # Create the directory if it does not exist
    os.makedirs(dir_path, exist_ok=True)
    # Construct the full file path
    speed_plot_path = os.path.join(dir_path, 'speed_real_vs_predicted.png')
    plt.savefig(speed_plot_path)
    plt.close(fig2)  # Close the figure to free memory

def plot_position(data):
    fig3, ax3 = plt.subplots(figsize=(8, 6))

    # Real positions
    ax3.scatter(data['Longitude'], data['Latitude'], label='Real Position', color='tab:blue', s=10)
    # Predicted positions with smaller deviation
    ax3.scatter(data['Predicted Longitude Small'], data['Predicted Latitude Small'], label='Predicted Position', color='tab:red', s=10, alpha=0.6)

    # Set labels and title
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title('Ship Positions : Real vs Predicted')
    ax3.legend()

    # Simulate a map background by just showing grid lines
    ax3.grid(True)

    # Tight layout for saving without clipping
    plt.tight_layout()

    # Save the plot as an image
    dir_path = os.path.join(base_dir, 'ship' + str(int(data['Ship ID'][0])))
    # Create the directory if it does not exist
    os.makedirs(dir_path, exist_ok=True)
    # Construct the full file path
    positions_small_plot_path = os.path.join(dir_path, 'positions_real_vs_small_predicted.png')
    plt.savefig(positions_small_plot_path)
    plt.close(fig3)  # Close the figure to free memory

def plot_cal_heading(data):
    # plot for position to heading
    bearings = []
    for i in range(1, len(data)):
        lat1 = data.iloc[i-1]['Smoothed Latitude']
        lon1 = data.iloc[i-1]['Smoothed Longitude']
        lat2 = data.iloc[i]['Smoothed Latitude']
        lon2 = data.iloc[i]['Smoothed Longitude']
        bearing = calculate_bearing(lat1, lon1, lat2, lon2)
        bearings.append(bearing)
    # Since there's no previous point for the first data point, we'll just use the first calculated bearing for it.
    bearings.insert(0, bearings[0])
    # Add the calculated bearings to the dataframe
    data['Calculated Heading'] = bearings
    # Now let's plot the calculated heading over time
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(data['Timestamp'], data['Calculated Heading'], label='Calculated Heading', color='tab:blue')

    # Set the labels and title
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Calculated Heading')
    plt.title('Calculated Heading Over Time')
    plt.legend()
    plt.tight_layout()

    # Save the plot to a file
    dir_path = os.path.join(base_dir, 'ship' + str(int(data['Ship ID'][0])))
    # Create the directory if it does not exist
    os.makedirs(dir_path, exist_ok=True)
    # Construct the full file path
    calculated_heading_plot_path = os.path.join(dir_path, 'calculated_heading.png')
    plt.savefig(calculated_heading_plot_path)
    plt.close(fig4)  # Close the figure to free memory

if __name__ == '__main__':
    # Load the data from the Excel file
    # change to your file position
    file_path = '/workspace/trajectron/STAR/data/ship_baidu/train.csv'
    base_dir = os.path.dirname(file_path)
    df_train = pd.read_csv(file_path)
    tm=df_train['timestamp']
    df_train['ts']=tm.apply(lambda x :convert_timestamp(x))   #将timestamp时间转换为长时间格式
    df_train['day']=df_train['ts'].apply(lambda x :get_day(x))#提取以日期用以区分训练集
    #通过数据分析发现不同船只编号mmsi有不同时间的数据，为防止日期重复，需要进行分别差分
    dup_mmsi = list(df_train['mmsi'].drop_duplicates())  #mmsi索引列表
    df_combin=[]                                                            #缓存处理数据集，用于事后合并
    for mmsi_index in tqdm.tqdm(dup_mmsi):                                  #按照mmsi索引不同船只
        dft= df_train[df_train['mmsi']==mmsi_index]
        dup_day = list(dft['day'].drop_duplicates())                        #区分不同日期划分子集
        if len(dup_day)>0:
            for days in dup_day:
                dd = dft[dft['day']==days]
                dd.set_index("ts", inplace = True)
                new_df_train=dd.asfreq("1s").interpolate(method='slinear')     #以1秒为单位进行线性差值
                new_df_train['ts']=new_df_train.index
                new_df_train['day']=days
                df_combin.append(new_df_train)
                # new_df_train.to_excel(r'./train_split/'+str(mmsi_index)+"_"+str(days)+".xlsx")#将数据保存到本地
    df_last=pd.concat(df_combin) #将处理后的数据进行合并
    # new_df_train
    # 将数据的列名称进行替换
    df_last.columns = ['Ship ID', 'Latitude', 'Longitude','Speed','Heading','Timestamp','day','ts']
    # 依据不同的ship ID 汇聚多个单个数据 进行分割 而后每个都相应的
    # 两个数据集的范围不同 lat[30.22,30.27] lon[122.245,122.252] => lat[30.35,30.55] lon[121.65,121.85]
    df_last['Latitude'] = df_last['Latitude'] - 0.18
    df_last['Longitude'] = df_last['Longitude'] - 0.18
    # Now you can iterate over each group and perform your analysis
    # Convert UNIX timestamp to datetime
    data = df_last
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
    # Extracting just the time part of the datetime
    data['Time'] = data['Timestamp'].dt.time
    # Simulate predictions for Heading and Speed
    data['Predicted Heading'] = simulate_prediction(data['Heading'])
    data['Predicted Speed'] = simulate_prediction(data['Speed'])
    # Smoothing the latitude and longitude
    data['Smoothed Latitude'], data['Smoothed Longitude'] = smooth_path(data['Latitude'], data['Longitude'])
    # Since we cannot access real map data, we will create a simple representation
    # Create a scatter plot for the ship's positions
    # Simulate smaller predictions for Longitude and Latitude
    data['Predicted Longitude Small'] = simulate_small_prediction(data['Longitude'])
    data['Predicted Latitude Small'] = simulate_small_prediction(data['Latitude'])
    grouped_data = data.groupby('Ship ID')
    for ship_id, group in tqdm.tqdm(grouped_data, desc='Processing ships'):
        grouped_data_by = group.groupby('day')
        # for day,grouped_data_by_single in grouped_data_by:
        #  Get the first group only
        day, grouped_data_by_single = next(iter(grouped_data_by))
        plot_heading(grouped_data_by_single)
        plot_speed(grouped_data_by_single)
        plot_position(grouped_data_by_single)









