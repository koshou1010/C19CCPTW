import json
import requests
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Bidirectional
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt


class WebCrawler:
    def __init__(self):
        self.code_map = {
        "id": "ID",
        "a01": "iso_code",
        "a02": "洲名",
        "a03": "國家",
        "a04": "日期",
        "a05": "總確診數",
        "a06": "新增確診數",
        "a07": "七天移動平均新增確診數",
        "a08": "總死亡數",
        "a09": "新增死亡數",
        "a10": "七天移動平均新增死亡數",
        "a11": "每百萬人確診數",
        "a12": "每百萬人死亡數",
        "a13": "傳染率",
        "a14": "新增檢驗件數",
        "a15": "總檢驗件數",
        "a16": "每千人檢驗件數",
        "a17": "七天移動平均新增檢驗件數",
        "a18": "陽性率",
        "a19": "每確診案例相對檢驗數量",
        "a20": "疫苗總接種總劑數",
        "a21": "疫苗總接種人數",
        "a22": "疫苗新增接種劑數",
        "a23": "七天移動平均疫苗新增接種劑數",
        "a24": "每百人接種疫苗劑數",
        "a25": "每百人接種疫苗人數",
        "a26": "疫情控管指數",
        "a27": "總人口數",
        "a28": "中位數年紀",
        "a29": "70歲以上人口比例",
        "a30": "平均壽命",
        "a31": "解除隔離數",
        "a32": "解封指數"
        }
    
    def craw_data(self)->(requests.models.Response):
        return requests.get("https://covid-19.nchc.org.tw/api/covid19?CK=covid-19@nchc.org.tw&querydata=4001&limited=TWN")

    def grab_dataframe(self)->(pd.DataFrame):
        for index, date in enumerate(self.df['a04']):
            if date == '2022-04-01':
                start = index
            end = index
        return self.df[start : end+1]
            
    def check_newest_one(self)->(bool):
        return 0 in self.df[0:1]['a06']
        
    def save_data(self, response:requests.models.Response)->(pd.DataFrame):
        self.df = pd.json_normalize(response.json())
        if self.check_newest_one():
            self.df = self.df.drop([0])
        # df.rename(columns=self.code_map, inplace=True)
        self.df = self.df.sort_values(by = 'a04')
        self.df.reset_index(inplace=True)
        self.df.to_csv('data.csv', encoding='utf-8-sig')
        return self.grab_dataframe()

class DLPredict:
    def __init__(self, dataframe):
        self.dataset = dataframe.drop(['index'], axis = 1)
        self.n_timestamp = 7
        self.train_days = 1500  # number of days to train from
        self.testing_days = 500 # number of days to be predicted
        self.n_epochs = 25
        self.filter_on = 1
        self.model_type = 2# Select model type {1: Single cell 2: Stacked 3: Bidirectional}
        
    def data_split(self,sequence, n_timestamp):
        x = []
        y = []
        for i in range(len(sequence)):
            end_ix = i + n_timestamp
            if end_ix > len(sequence)-1:
                break
            # i to end_ix as input
            # end_ix as target output
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            x.append(seq_x)
            y.append(seq_y)
        return np(x), np(y)
    
    def model_build(self):
        if self.model_type == 1:
            # Single cell LSTM
            self.model = Sequential()
            self.model.add(LSTM(units = 50, activation='relu',input_shape = (self.x_train.shape[1], 1)))
            self.model.add(Dense(units = 1))
        if self.model_type == 2:
            # Stacked LSTM
            self.model = Sequential()
            self.model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
            self.model.add(LSTM(50, activation='relu'))
            self.model.add(Dense(1))
        if self.model_type == 3:
            # Bidirectional LSTM
            self.model = Sequential()
            self.model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(self.x_train.shape[1], 1)))
            self.model.add(Dense(1))
    
    def model_training(self):
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        history = self.model.fit(self.x_train, self.y_train, epochs = self.n_epochs, batch_size = 32)
        loss = history.history['loss']
        epochs = range(len(loss))
    
    def main(self):
        '''
        filtering
        '''

        # if self.filter_on == 1:
        #     self.dataset['a06'] = medfilt(self.dataset['a06'], 3)
        #     print(self.dataset['a06'].tolist())
        #     self.dataset['a06'] = gaussian_filter1d(self.dataset['a06'].toliost(), 1.2)   
        # print(self.dataset['a06'])
    
        train_set = self.dataset.reset_index(drop=True)
        test_set = self.dataset.reset_index(drop=True)
        training_set = train_set.iloc[:, 6:7].values
        testing_set = test_set.iloc[:, 6:7].values
        
        # print(train_set)
        # print(test_set)
        # print(training_set)
        # print(testing_set)
        
        '''
        Normalize data first
        '''
        sc = MinMaxScaler(feature_range = (0, 1))
        training_set_scaled = sc.fit_transform(training_set)
        testing_set_scaled = sc.fit_transform(testing_set)
        # print(training_set_scaled)
        # print(testing_set_scaled)
        
        
        self.x_train, self.y_train = self.data_split(training_set_scaled, self.n_timestamp)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        self.x_test, self.y_test = self.data_split(testing_set_scaled, self.n_timestamp)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)
        self.model_build()
        self.model_training()
        y_predicted = self.model.predict(self.x_test)
        y_predicted_descaled = sc.inverse_transform(y_predicted)
        y_train_descaled = sc.inverse_transform(self.y_train)
        y_test_descaled = sc.inverse_transform(self.y_test)
        y_pred = y_predicted.ravel()
        y_pred = [round(yx, 2) for yx in y_pred]
        y_tested = self.y_test.ravel()
        
        '''
        show result
        '''
        plt.figure(figsize=(8,7))

        plt.subplot(3, 1, 1)
        plt.plot(dataset['Temperature'], color = 'black', linewidth=1, label = 'True value')
        plt.ylabel("Temperature")
        plt.xlabel("Day")
        plt.title("All data")


        plt.subplot(3, 2, 3)
        plt.plot(y_test_descaled, color = 'black', linewidth=1, label = 'True value')
        plt.plot(y_predicted_descaled, color = 'red',  linewidth=1, label = 'Predicted')
        plt.legend(frameon=False)
        plt.ylabel("Temperature")
        plt.xlabel("Day")
        plt.title("Predicted data (n days)")

        plt.subplot(3, 2, 4)
        plt.plot(y_test_descaled[0:75], color = 'black', linewidth=1, label = 'True value')
        plt.plot(y_predicted_descaled[0:75], color = 'red', label = 'Predicted')
        plt.legend(frameon=False)
        plt.ylabel("Temperature")
        plt.xlabel("Day")
        plt.title("Predicted data (first 75 days)")

        plt.subplot(3, 3, 7)
        plt.plot(epochs, loss, color='black')
        plt.ylabel("Loss (MSE)")
        plt.xlabel("Epoch")
        plt.title("Training curve")

        plt.subplot(3, 3, 8)
        plt.plot(y_test_descaled-y_predicted_descaled, color='black')
        plt.ylabel("Residual")
        plt.xlabel("Day")
        plt.title("Residual plot")

        plt.subplot(3, 3, 9)
        plt.scatter(y_predicted_descaled, y_test_descaled, s=2, color='black')
        plt.ylabel("Y true")
        plt.xlabel("Y predicted")
        plt.title("Scatter plot")

        plt.subplots_adjust(hspace = 0.5, wspace=0.3)
        plt.show()



        mse = mean_squared_error(y_test_descaled, y_predicted_descaled)
        r2 = r2_score(y_test_descaled, y_predicted_descaled)
        print("mse=" + str(round(mse,2)))
        print("r2=" + str(round(r2,2)))