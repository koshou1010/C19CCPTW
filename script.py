import utils
'''

work flow: 
1.request grab data and organise that
2.put in LSTM to predict
https://www.rs-online.com/designspark/lstm-cn
'''


if __name__ == '__main__':
  grab = utils.WebCrawler()
  dataframe = grab.save_data(grab.craw_data())
  a = utils.DLPredict(dataframe)
  a.main()