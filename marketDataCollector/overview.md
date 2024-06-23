# base features 
1. data download
    1. ??? source questions
        1. do we want use other packages or we want to implement from scratch
        1.1. if we want to make from scratch
           - what may provide data for 1. forex 2. crypto
           - for each source:
               - in what manner we may access their data(is it zip files, api, need scraping)
        1.2. if we want to use other packages
           - are there any packages which have done this?(more probably yes, so which packages what done this)
        2. at what intervals they update their data
    2. update with some interval
       1. possible bugs
           1. (related to `creating timeFrames`) assume in the last update for 15 min data is only had data for it's 13th min of that candle so we want to recorrect that 15min candle
    3. should make interfaces to download from each source
2. data cleaning
    1. what cleaning are needed, what are possible data corruptions
    2. how to deal missing data(probably for some actions we need user prefrences)
3. creating timeFrames
    - fast: should do it with numpy or cython or even first with numpy then with cython
    - how the data should be saved, at the lowest timeframe, or should get options from the users to enable them fetching the data faster by saving their desired timeframe
4. (in mid term) adapt data with VannTsDataset
    - rn I guess it won't be challenging
5. backTesting with visualzation
    - desired feature:
        - don't have bugs of providing future data to algoTrader algorithm
        - (for later) should be able to combine multiple strategies
            - collisions
                - if strat1 says to sell and strat2 says to buy what should have be done
                    - prioritize one over
                    - do after deduction of volume sizes
                    - warn if there is some contradictory action is going on, for i.e. start1 2 candles ago has decided to buy and now start2 decides to sell
        - visualization should support
            - strategy profit, drawdown, ...
            - should visualize the times each strategy made a decision and was it profitable or not
    - questions:
        1. do we want use other packages or we want to implement from scratch
            1.1. if we want to make from scratch
              - why? (to showcase?!!? I guess that's not a good showcase)
              - do enable us later customize some parts easier
                  - so what options they don't provide
            1.1. why we want to make from scratch
# questions
# possible bugs