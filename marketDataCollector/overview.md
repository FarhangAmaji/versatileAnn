1. data download
    1. ??? source questions
        1. what may provide data for 1. forex 2. crypto
        2. at what intervals they update their data
        3. in what manner we may access their data(is it zip files, api, need scraping)
    2. update with some interval
       1. possible bugs
           1. (related to `creating timeFrames`) assume in the last update for 15 min data is only had data for it's 13th min of that candle so we want to recorrect that 15min candle 
2. data cleaning
    1. what cleaning are needed, what are possible data corruptions
    2. how to deal missing data(probably for some actions we need user prefrences)
3. creating timeFrames
    - fast: should do it with numpy or cython or even first with numpy then with cython
    - how the data should be saved, at the lowest timeframe, or should get options from the users to enable them fetching the data faster by saving their desired timeframe
4. (in mid term) adapt data with VannTsDataset
    - rn I guess it won't be challenging