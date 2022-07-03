# This allows multiple outputs from a single jupyter notebook cell:
from types import NoneType
import numpy as np
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"

import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

class KeyFinancials:
    def __init__(self, ticker_symbol):
        self.ticker = ticker_symbol
        self.df = yf.download(self.ticker, period="10y", interval="1d") # d, wk, m, y
        self.QQQ_df = yf.download("QQQ", period="10y", interval="1d")
        self.SCHD_df = yf.download("SCHD", period="10y", interval="1d")
        self.size_10y = len(list(self.QQQ_df["Adj Close"]))
        self.size_5y = int(self.size_10y/2)
        self.size_2y = int(self.size_10y/5)
        self.size_6m = int(self.size_10y/20)
        self.size_1m = int(self.size_10y/120)
        # here you can find what info you can get
        # 1+3+4+3=11 items
        # https://stackoverflow.com/questions/61522604/get-information-using-yfinance-for-tickers-with
        self.history_info = yf.Ticker(self.ticker)
        self.history_info.history(period="5y")
        self.info_list = ['currentPrice', 
                          'marketCap', 'totalDebt', 'totalCash', 
                          'totalRevenue', 'operatingCashflow', 'freeCashflow', 'netIncomeToCommon', 
                          'revenueGrowth', 'quickRatio', 'trailingPE']
        self.competitors = []
        # P/S, P/OCF, P/FCF, P/E
        self.industry_average_ratios = []
        self.customized_info_df = []
        self.long_term_debt = []
        self.revenue = []
        self.ocf = []
        self.fcf = []
        self.fcf_to_revenue = [] # I will use this for valuation
        self.ocf_to_revenue = [] # use this for valuation

    # Full period here means 10 years
    # The stock to be investigated may not have 10 years' history
    def get_full_period_data(self, original_df):
        # Stock to be investigated
        price_original = list(original_df["Adj Close"])
        size_original = len(price_original)

        # Add zeros in the front of the list 
        # In case the stock has less than 10 years' history
        if self.size_10y - size_original > 0:
            price_10y = [0]* (self.size_10y - size_original) + price_original
        else:
            price_10y = price_original

        return size_original, price_original, price_10y
    

    # Get the price percentage change in a given period
    def get_percentage_change_period(self, full_period_price, i_first_nonzero_full_period_price, period_size):
        # Get all the price for this period
        price_period = full_period_price[-period_size:]
        # percentage change
        if price_period[0] == 0:
            # When we try to get data for the last 10 years, 
            # but some stock doesn't have 10 years of data while QQQ does have 10 years of data
            # so we add zeros in the front for stocks that doesn't have 10 years of data
            number_of_zeros = period_size - (self.size_10y - i_first_nonzero_full_period_price)
            stock_price_percentage_change = [0] * number_of_zeros
            for i in range(number_of_zeros, period_size):
                stock_price_percentage_change.append(price_period[i]/full_period_price[i_first_nonzero_full_period_price] - 1)
        else:
            stock_price_percentage_change = [x/price_period[0] - 1 for x in price_period]

        return stock_price_percentage_change


    def get_daily_return(self):
        self.df["Daily Return"]      = self.df["Adj Close"] / self.df["Adj Close"].shift(1) - 1
        self.QQQ_df["Daily Return"]  = self.QQQ_df["Adj Close"] / self.QQQ_df["Adj Close"].shift(1) - 1
        self.SCHD_df["Daily Return"] = self.SCHD_df["Adj Close"] / self.SCHD_df["Adj Close"].shift(1) - 1
    

    def plot_return_correlation(self):
        self.get_daily_return()
        all_stocks_return = pd.concat([self.df["Daily Return"],
                                       self.QQQ_df["Daily Return"],
                                       self.SCHD_df["Daily Return"]], axis=1).dropna()
        # add column names
        names = [self.ticker, 'QQQ', 'SCHD']
        all_stocks_return.columns = names
    
        # Get the coorelation matrix
        mtx = all_stocks_return.corr()

        # Plot the heatmap of coorelation matrix
        plt.imshow(mtx, cmap='hot', interpolation='nearest')
        plt.show()


    def plot_benchmark_comparisons(self):
        # Get the data
        original_size, _, price_10y= self.get_full_period_data(self.df)
        _, _, QQQ_10y = self.get_full_period_data(self.QQQ_df)
        _, _, SCHD_10y = self.get_full_period_data(self.SCHD_df)

        # Plot the data
        plt.rcParams['lines.linewidth'] = 1.5
        fig = plt.figure(figsize=(12, 8))
        
        # full period - 10 years
        stock_change_10y = self.get_percentage_change_period(price_10y, self.size_10y - original_size, self.size_10y)
        QQQ_change_10y = self.get_percentage_change_period(QQQ_10y, 0, self.size_10y)
        SCHD_change_10y = self.get_percentage_change_period(SCHD_10y, 0, self.size_10y)

        # two rows, two columns, first and second plots
        sub1 = fig.add_subplot(2,2,(1,2))
        sub1.plot(stock_change_10y, label=self.ticker, color = "m", linestyle="solid")
        sub1.plot(QQQ_change_10y, label='QQQ', color = "#708090", linestyle="solid")
        sub1.plot(SCHD_change_10y, label='SCHD', color = "#708090", linestyle="solid")
        plt.gca().yaxis.grid(True)
        sub1.set_title("Comparison - 10 years (Growth Percentage)")
        sub1.legend(bbox_to_anchor =(1.14, 1))
        plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])

        # since IPO
        # two rows, two columns, third plot
        stock_change_since_IPO = self.get_percentage_change_period(price_10y, self.size_10y - original_size, original_size)
        QQQ_change_period = self.get_percentage_change_period(QQQ_10y, 0, original_size)
        SCHD_change_period = self.get_percentage_change_period(SCHD_10y, 0, original_size) 

        sub2 = fig.add_subplot(2,2,3)
        sub2.plot(stock_change_since_IPO, label=self.ticker, color = "m", linestyle="solid")
        sub2.plot(QQQ_change_period, label='QQQ', color = "#708090", linestyle="solid")
        sub2.plot(SCHD_change_period, label='SCHD', color = "#708090", linestyle="solid")
        plt.gca().yaxis.grid(True)
        plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()]) 
        sub2.set_title("Comparison - since IPO of my stock (Growth Percentage)")

        # past 6 months
        # two rows, two columns, fourth plot
        stock_change_last_6months = self.get_percentage_change_period(price_10y, self.size_10y - original_size, self.size_6m)
        QQQ_change_period = self.get_percentage_change_period(QQQ_10y, 0, self.size_6m)
        SCHD_change_period = self.get_percentage_change_period(SCHD_10y, 0, self.size_6m)  

        sub3 = fig.add_subplot(2,2,4)
        sub3.plot(stock_change_last_6months, label=self.ticker, color = "m", linestyle="solid")
        sub3.plot(QQQ_change_period, label='QQQ', color = "#708090", linestyle="solid")
        sub3.plot(SCHD_change_period, label='SCHD', color = "#708090", linestyle="solid")
        sub3.set_title("Comparison - last 6 months (Growth Percentage)")
        plt.gca().yaxis.grid(True)
        plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])
        # show the image
        plt.show()

    def get_customized_info_single_stock(self, ticker):
        print("Loading history info of", ticker, "......")
        all_info = yf.Ticker(ticker)
        # get historical market data
        all_info.history(period="10y")
        # Create empty DataFrame
        my_info_df = pd.DataFrame()
        # Note that the first 7 items are in dollars
        # it has 11+4+3=18 items
        rows = ['Current Price', 
                'Market Cap(m)', 'Total Debt(m)', 'Total Cash(m)',
                'Total Revenue(m)', 'Operating Cashflow(m)', 'Free Cashflow(m)', 'Net Income(m)', 
                'Revenue Growth', 'Quick Ratio', 'Debt/FCF',
                'P/S', 'P/OCF', 'P/FCF', 'P/E', 
                'Net Income/S', 'FCF/OCF', 'OCF/S']
        # set names for all rows
        my_info_df.index = rows
        # Note that the original info list has 2+2+4+3=11 items which is defined in the __init__ function
        # info_list = ['currentPrice', 
        #              'marketCap', 'totalDebt', 'totalCash', 
        #              'totalRevenue', 'operatingCashflow', 'freeCashflow', 'netIncomeToCommon',
        #              'revenueGrowth', 'quickRatio', 'trailingPE']
        values = []
        size = len(self.info_list)

        # add the first item, get current stock price, add it to the list - values
        x = self.info_list[0]
        values.append(self.get_two_decimals(all_info.info[x]))

        # Convert the 2nd to 8th items into millions, and add them to the list - values
        for i in range(1, 8):
            x = self.info_list[i]
            temp = self.fix_keyError_column(all_info.info, x)
            temp = self.get_two_decimals(temp/1000000)
            # add the value
            values.append(temp)
        
        # add the values of the last 3 items to the list - values
        for i in range(8, size):
            x = self.info_list[i]
            temp = self.fix_keyError_column(all_info.info, x)
            temp = self.get_two_decimals(temp)
            # add the value
            values.append(temp)
        
        # Get values
        marketCap = values[1]
        debt = values[2]
        sales = values[4] # total revenue
        ocf = values[5] # operating cashflow
        fcf = values[6] # free cashflow
        net_income = values[7]
        pe_ratio = values[10]
        values[10] = self.fix_zero_denominator(debt, fcf) # change the 11th item to debt/FCF
        values.append(self.fix_zero_denominator(marketCap, sales)) # add P/S, the 12th item
        values.append(self.fix_zero_denominator(marketCap, ocf)) # add P/OCF, the 13th item
        values.append(self.fix_zero_denominator(marketCap, fcf)) # add P/FCF, the 14th item
        values.append(pe_ratio) # add P/E back, the 15th item
        values.append(self.fix_zero_denominator(net_income, sales)) # add netIncome/S, the 16th item
        values.append(self.fix_zero_denominator(fcf, ocf)) # add fcf/ocf, the 17th item
        values.append(self.fix_zero_denominator(ocf, sales)) # add ocf/sales, the 18th item

        # add the entire column
        my_info_df[self.ticker] = values

        # return
        return all_info, my_info_df, values

    def get_customized_info_all_stocks(self):
        if len(self.competitors) == 0:
            raise Exception("Sorry, you need to add ticker symbols to the list - instance_name.competitors! ")
        # df with only one column
        _, my_info_df, _ =  self.get_customized_info_single_stock(self.ticker)
        # all the columns for competitors' stock
        for x in self.competitors:
            _, _, values = self.get_customized_info_single_stock(x)
            my_info_df[x] = values
        # Save it
        self.customized_info_df = my_info_df
        return my_info_df

    def plot_shares(self):
        num_shares_df = pd.DataFrame()
        rows = list(self.history_info.shares.index)
        rows_m = []
        # make the rows look like 2019(m)
        for x in rows:
            x = str(x) + '(m)'
            rows_m.append(x)
        num_shares_df.index = rows_m

        # makes shares experessed in millions
        cols = list(self.history_info.shares['BasicShares'])
        shares_two_decimals = self.get_millions(cols)
        
        # Get shares in the last 5 years
        num_shares_df[self.ticker] = shares_two_decimals

        # Start plotting
        fig = plt.figure(figsize=(11, 3.5))

        # bar properties
        barWidth = 0.2
        num_bars = len(rows_m)
        br1 = np.arange(num_bars)
        # first plot
        sub1 = fig.add_subplot(1,2,1)
        print("\nOutstanding Shares in millions = ", shares_two_decimals)
        plt.bar(br1, shares_two_decimals, color ='c', width = barWidth, label ='Outstanding Shares')
        plt.xticks([r + barWidth for r in range(num_bars)], rows_m)
        sub1.set_title("Outstanding shares(in millions)")
        plt.gca().yaxis.grid(True)

        # second plot
        sub2 = fig.add_subplot(1,2,2)
        shares_change_percentage = []
        i = 1
        for i in range(num_bars):
            temp = self.fix_zero_denominator(shares_two_decimals[i]-shares_two_decimals[i-1], shares_two_decimals[i-1])
            shares_change_percentage.append(temp)
        print("Shares change in percentage = ", shares_change_percentage)
        num_shares_df["Percentage_change"] = shares_change_percentage
        plt.bar(br1, num_shares_df["Percentage_change"], color ='m', width = barWidth, label ='Outstanding Shares Change in Percentage')
        plt.xticks([r + barWidth for r in range(num_bars)], rows_m)
        sub2.set_title("Outstanding Shares Change in Percentage")
        plt.gca().yaxis.grid(True)
        plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])

        # Show the image
        plt.show()

    def plot_income_statement(self):
        income_statement_df = self.history_info.financials
        income_statement_df.columns = list(income_statement_df.columns.year)
        years = list(income_statement_df.columns)
        years.reverse()
        num_bars = len(years) # number of bars(years) for one type of data, e.g. revenue
        # all the info I'm interested
        info_list = ['Total Revenue', 
                     'Cost Of Revenue', 
                     'Selling General Administrative', 
                     'Operating Income']

        # Get all data
        all_data = []
        for x in info_list:
            temp = self.fix_keyError_row(income_statement_df, x, num_bars)
            temp.reverse()
            temp = self.get_millions(temp) # convert the numbers into a unit of millions
            all_data.append(temp)
            print(x, "=", temp)

        # Plot the data
        fig = plt.figure(figsize=(12, 3.5))

        # bar properties
        barWidth = 0.1
        br1 = np.arange(num_bars)
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]

        # first plot
        sub1 = fig.add_subplot(1,2,1)
        self.revenue = all_data[0]
        plt.bar(br1, self.revenue, color ='c', width = barWidth, label ='Sales')
        plt.bar(br2, all_data[3], color ='#FF66B2', width = barWidth, label ='Operating Income')
        sub1.legend(bbox_to_anchor =(-0.1, 1))
        sub1.set_title("Income statement (in millions)")
        plt.xticks([r + barWidth for r in range(num_bars)], years)
        plt.gca().yaxis.grid(True)

        # second plot
        sub2 = fig.add_subplot(1,2,2)
        # Revenue growth
        revenue_growth_percentage = []
        i = 1
        for i in range(num_bars):
            temp = self.fix_zero_denominator(self.revenue[i]-self.revenue[i-1], self.revenue[i-1])
            revenue_growth_percentage.append(temp)
        print("\nRevenue growth = ",revenue_growth_percentage)

        # Revenue related raios
        cost_of_revenue_to_revenue = []
        sga_to_revenue = []
        operating_income_to_revenue = []
        i=0
        for i in range(num_bars):
            cost_of_revenue_to_revenue.append(self.fix_zero_denominator(all_data[1][i], self.revenue[i]))
            sga_to_revenue.append(self.fix_zero_denominator(all_data[2][i], self.revenue[i]))
            operating_income_to_revenue.append(self.fix_zero_denominator(all_data[3][i], self.revenue[i]))
        print("Cost of Revenue (Sales) / Sales =", cost_of_revenue_to_revenue)
        print("SG&A / Sales =", sga_to_revenue)
        print("Operating Income / Sales =", operating_income_to_revenue)

        plt.bar(br1, revenue_growth_percentage, color ='c', width = barWidth, label ='Revenue Growth')
        plt.bar(br2, cost_of_revenue_to_revenue, color ='#FF66B2', width = barWidth, label ='Cost Of Sales/Sales')
        plt.bar(br3, sga_to_revenue, color ='#1E90FF', width = barWidth, label ='SG&A/Sales')
        plt.bar(br4, operating_income_to_revenue, color ='#FFFF00', width = barWidth, label ='Operating Income/Sales')
        
        sub2.legend(bbox_to_anchor =(1.1, 1))
        sub2.set_title("Revenue (Sales) related margins")
        plt.xticks([r + barWidth for r in range(num_bars)], years)
        plt.gca().yaxis.grid(True)
        plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])

        # Show the image
        plt.show()

    def plot_cashflow_statement(self):
        cashflow_statement_df = self.history_info.cashflow
        cashflow_statement_df.columns = list(cashflow_statement_df.columns.year)
        years = list(cashflow_statement_df.columns)
        years.reverse()
        num_bars = len(years) # number of bars for one info

        # all the info I'm interested
        info_list = ['Total Cash From Operating Activities', 
                     'Total Cashflows From Investing Activities', 
                     'Total Cash From Financing Activities', 
                     'Capital Expenditures',
                     'Net Income']
        
        # Get all data
        all_data = []
        for x in info_list:
            temp = self.fix_keyError_row(cashflow_statement_df, x, num_bars)
            temp.reverse()
            temp = self.get_millions(temp) # convert the numbers into a unit of millions
            all_data.append(temp)
            print(x, "=", temp)

        self.ocf = all_data[0]
        cfi = all_data[1]
        cff = all_data[2]
        capex = all_data[3]
        net_income = all_data[4]

        # Calculate Free Cashflow 
        i = 0
        for i in range(num_bars):
            # CapEx is already negative, that's why it's OCF+CapEx, not OCF-CapEx
            self.fcf.append(self.get_two_decimals(self.ocf[i]+capex[i]))
        print("Free Cashflow =", self.fcf)

        # Make the plot
        fig = plt.figure(figsize=(12, 3.5))

        # bar properties
        barWidth = 0.1
        br1 = np.arange(num_bars)
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]
        br5 = [x + barWidth for x in br4]

        # first plot
        sub1 = fig.add_subplot(1,2,1)

        plt.bar(br1, self.ocf, color ='c', width = barWidth, label ='CFO')
        plt.bar(br2, cfi, color ='#FF66B2', width = barWidth, label ='CFI')
        plt.bar(br3, cff, color ='#1E90FF', width = barWidth, label ='CFF')
        plt.bar(br4, capex, color ='#FFFF00', width = barWidth, label ='CapEx')
        plt.bar(br5, self.fcf, color ='#8A2BE2', width = barWidth, label ='Free Cashflow')
        sub1.legend(bbox_to_anchor=(-0.1, 1))
        sub1.set_title("Cashflow statement (in millions)")
        plt.xticks([r + barWidth for r in range(num_bars)], years)
        plt.gca().yaxis.grid(True)

        # Second plot
        sub2 = fig.add_subplot(1,2,2)
        
        # OCF to revenue
        i = 0
        for i in range(num_bars):
            self.ocf_to_revenue.append(self.fix_zero_denominator(self.ocf[i], self.revenue[i]))
        print("\nOCF to Revenue = ",self.ocf_to_revenue)
        
        # FCF to revenue
        i = 0
        for i in range(num_bars):
            self.fcf_to_revenue.append(self.fix_zero_denominator(self.fcf[i], self.revenue[i]))
        print("FCF to Revenue = ", self.fcf_to_revenue)

        # Net Income to revenue
        net_income_to_revenue = []
        i = 0
        for i in range(num_bars):
            net_income_to_revenue.append(self.fix_zero_denominator(net_income[i], self.revenue[i]))
        print("Net Income to Revenue = ", net_income_to_revenue)

        # plot bar chart
        plt.bar(br1, self.ocf_to_revenue, color ='c', width = barWidth, label ='OCF to Sales')
        plt.bar(br2, self.fcf_to_revenue, color ='#FF66B2', width = barWidth, label ='FCF to Sales')
        plt.bar(br3, net_income_to_revenue, color ='#1E90FF', width = barWidth, label ='Net Income to Sales')

        sub2.legend(bbox_to_anchor=(1.1, 1))
        sub2.set_title("Revenue (Sales) related margins")
        plt.xticks([r + barWidth for r in range(num_bars)], years)
        plt.gca().yaxis.grid(True)
        plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])

        # Show the image
        plt.show()

    def plot_balance_sheet(self):
        balance_sheet_df = self.history_info.balance_sheet
        balance_sheet_df.columns = list(balance_sheet_df.columns.year)
        years = list(balance_sheet_df.columns)
        years.reverse()
        num_bars = len(years)

        # all the info I'm interested
        info_list = ['Long Term Debt', 
                     'Total Liab', 
                     'Cash', 
                     'Net Tangible Assets',
                     'Intangible Assets']
        
        # Get all data
        all_data = []
        for x in info_list:
            temp = self.fix_keyError_row(balance_sheet_df, x, num_bars)
            temp.reverse()
            temp = self.get_millions(temp) # convert the numbers into a unit of millions
            all_data.append(temp)
            print(x, "=", temp)

        # Make the plot
        fig = plt.figure(figsize=(12, 3.5))

        # bar properties
        barWidth = 0.1
        br1 = np.arange(num_bars)
        br2 = [x + barWidth for x in br1]
        br3 = [x + barWidth for x in br2]
        br4 = [x + barWidth for x in br3]
        br5 = [x + barWidth for x in br4]

        # first plot
        sub1 = fig.add_subplot(1,2,1)
        self.long_term_debt = all_data[0]
        total_cash = all_data[2]
        plt.bar(br1, self.long_term_debt, color ='c', width = barWidth, label ='Long-term Debt')
        plt.bar(br2, all_data[1], color ='#FF66B2', width = barWidth, label ='Total Liability')
        plt.bar(br3, total_cash, color ='#1E90FF', width = barWidth, label ='Total Cash')
        plt.bar(br4, all_data[3], color ='#FFFF00', width = barWidth, label ='Tangible Assets')
        plt.bar(br5, all_data[4], color ='#8A2BE2', width = barWidth, label ='Intangible Assets')
        plt.xticks([r + barWidth for r in range(num_bars)], years)
        sub1.set_title("Balance Sheet (in millions)")
        sub1.legend(bbox_to_anchor =(-0.1, 1))
        plt.gca().yaxis.grid(True)

        # second plot
        sub2 = fig.add_subplot(1,2,2)
        fcf_to_debt = []
        cash_to_debt = []
        i = 0
        for i in range(num_bars):
            fcf_to_debt.append(self.fix_zero_denominator(self.fcf[i], self.long_term_debt[i]))
        i = 0
        for i in range(num_bars):
            cash_to_debt.append(self.fix_zero_denominator(total_cash[i], self.long_term_debt[i]))
        print("\nFCF to Long-term Debt = ",fcf_to_debt)
        print("Total Cash to Long-term Debt = ",fcf_to_debt)
        plt.bar(br1, fcf_to_debt, color ='c', width = barWidth, label ='FCF to Long-term Debt')
        plt.bar(br2, cash_to_debt, color ='#FF66B2', width = barWidth, label ='Total Cash to Long-term Debt')
        plt.xticks([r + barWidth for r in range(num_bars)], years)
        sub2.legend(bbox_to_anchor=(1.1, 1))
        plt.gca().yaxis.grid(True)
        plt.gca().set_yticklabels([f'{x:.0%}' for x in plt.gca().get_yticks()])

        # Show the image
        plt.show()

    # Calcualte fair stock price
    def calculate3y_value_relative(self, revenue_growth_prediction_list):
        print("Note that the predicted price (3-year) assumes that the number of shares outstanding stays the same as the current value! ")
        # ['Current Price', 
        #  'Market Cap(m)', 'Total Debt(m)', 'Total Cash(m)',
        #  'Total Revenue(m)', 'Operating Cashflow(m)', 'Free Cashflow(m)', 'Net Income(m)', 
        #  'Revenue Growth', 'Quick Ratio', 'Debt/FCF',
        #  'P/S', 'P/OCF', 'P/FCF', 'P/E', 
        #  'netIncome/S', 'FCF/OCF', 'OCF/S']
        valuation_df = pd.DataFrame()
        rows = ['Current Price of ' + self.ticker, 
                'Valuation using P/S',
                'Valuation using P/OCF', 
                'Valuation using P/FCF', 
                'Valuation using P/E', 
                'Valuation in 3-years(P/S)',
                'Valuation in 3-years(P/OCF)',
                'Valuation in 3-years(P/FCF)']
        valuation_df.index = rows

        # Get the current price of self.ticker
        current_price = self.customized_info_df[self.ticker].values[0]
        PS_ratio = self.customized_info_df[self.ticker].values[11]
        POCF_ratio = self.customized_info_df[self.ticker].values[12]
        PFCF_ratio = self.customized_info_df[self.ticker].values[13]
        PE_ratio = self.customized_info_df[self.ticker].values[14]

        # Calculate the predicted revenue (3 years later)
        current_market_cap = self.customized_info_df[self.ticker].values[1]
        current_revenue = self.customized_info_df[self.ticker].values[4]
        predicted_revenue_after_3y = current_revenue
        if len(revenue_growth_prediction_list) != 3:
            raise Exception("Sorry, we need 3 years of revenue growth assumption! 8 percentage is 0.08!")
        print("Current Market Cap = ", current_market_cap)
        print("Current Revenue = ", current_revenue)
        i=0
        for x in revenue_growth_prediction_list:
            i += 1
            predicted_revenue_after_3y = predicted_revenue_after_3y * (1 + x)
            print("Predicted revenue after ", i, " year = ", predicted_revenue_after_3y)     
        
        # Calculate the predicted OCF (3 years later)
        print("\n")
        if len(self.ocf_to_revenue) == 0:
            raise Exception("Sorry, you need to process cash flow statement first to get OCF to Sales ratio.")
        sum_ocf_to_revenue = 0
        for x in self.ocf_to_revenue:
            sum_ocf_to_revenue += x
        average_ocf_to_revenue = self.fix_zero_denominator(sum_ocf_to_revenue, len(self.ocf_to_revenue))
        print("Average OCF/S ratio of last 4 years = ", average_ocf_to_revenue)
        predicted_ocf_after_3y = average_ocf_to_revenue * predicted_revenue_after_3y
        print("Predicted OCF after 3 years = ", predicted_ocf_after_3y)

        # Calculate the predicted FCF (3 years later)
        print("\n")
        if len(self.fcf_to_revenue) == 0:
            raise Exception("Sorry, you need to process cash flow statement first to get FCF to Sales ratio.")
        sum_fcf_to_revenue = 0
        for x in self.fcf_to_revenue:
            sum_fcf_to_revenue += x
        average_fcf_to_revenue = self.fix_zero_denominator(sum_fcf_to_revenue, len(self.fcf_to_revenue))
        print("Average FCF/S ratio of last 4 years = ", average_fcf_to_revenue)
        predicted_fcf_after_3y = average_fcf_to_revenue * predicted_revenue_after_3y
        print("Predicted FCF after 3 years = ", predicted_fcf_after_3y)

        # Valuation
        i=0
        for x in self.competitors:
            # Add the current stock price
            cols = [self.get_two_decimals(current_price)]
            i += 1
            # Get the ratios of current competetor
            PS_ratio_x = self.customized_info_df[x].values[11]
            POCF_ratio_x = self.customized_info_df[x].values[12]
            PFCF_ratio_x = self.customized_info_df[x].values[13]
            PE_ratio_x = self.customized_info_df[x].values[14]

            # ratios: the stock under investigation versus the i-th stock in the competitor list
            PS_ticker_vs_x = PS_ratio/PS_ratio_x
            cols.append(self.get_two_decimals(current_price/PS_ticker_vs_x))

            POCF_ticker_vs_x = POCF_ratio/POCF_ratio_x
            cols.append(self.get_two_decimals(current_price/POCF_ticker_vs_x))

            PFCF_ticker_vs_x = PFCF_ratio/PFCF_ratio_x
            cols.append(self.get_two_decimals(current_price/PFCF_ticker_vs_x))

            PE_ticker_vs_x = PE_ratio/PE_ratio_x
            cols.append(self.get_two_decimals(current_price/PE_ticker_vs_x))

            # Predict the market cap based on the P/S ratio of the i-th competitor
            predicted_market_cap_based_on_x_psratio = PS_ratio_x * predicted_revenue_after_3y
            market_cap_predicted_vs_current = predicted_market_cap_based_on_x_psratio/current_market_cap
            predicted_price_using_PS = market_cap_predicted_vs_current * current_price
            cols.append(self.get_two_decimals(predicted_price_using_PS))

            # Predict the market cap based on the P/OCF ratio of the i-th competitor
            predicted_market_cap_based_on_x_pocfratio = POCF_ratio_x * predicted_ocf_after_3y
            market_cap_predicted_vs_current = predicted_market_cap_based_on_x_pocfratio/current_market_cap
            predicted_price_using_POCF = market_cap_predicted_vs_current * current_price
            cols.append(self.get_two_decimals(predicted_price_using_POCF))

            # Predict the market cap based on the P/FCF ratio of the i-th competitor
            predicted_market_cap_based_on_x_pfcfratio = PFCF_ratio_x * predicted_fcf_after_3y
            market_cap_predicted_vs_current = predicted_market_cap_based_on_x_pfcfratio/current_market_cap
            predicted_price_using_PFCF = market_cap_predicted_vs_current * current_price
            cols.append(self.get_two_decimals(predicted_price_using_PFCF))

            # Add the entire column to the dataframe
            x = 'Price based on ' + x
            valuation_df[x] = cols
        
        # add valuation based on stock market average P/E, P/FCF, P/S ratios
        if len(self.industry_average_ratios) == 0:
            raise Exception("Sorry, you need to provide the industry average ratios: P/S, P/OCF, P/FCF, P/E!")
        # Current stock price
        cols = [self.get_two_decimals(current_price)]
        # Calculate price
        industry_average_ps = self.industry_average_ratios[0]
        ps_ticker_vs_market = PS_ratio/industry_average_ps
        cols.append(self.get_two_decimals(current_price/ps_ticker_vs_market))

        industry_average_pocf = self.industry_average_ratios[1]
        pocf_ticker_vs_market = POCF_ratio/industry_average_pocf
        cols.append(self.get_two_decimals(current_price/pocf_ticker_vs_market))

        industry_average_pfcf = self.industry_average_ratios[2]
        pfcf_ticker_vs_market = PFCF_ratio/industry_average_pfcf
        cols.append(self.get_two_decimals(current_price/pfcf_ticker_vs_market))

        industry_average_pe = self.industry_average_ratios[3]
        pe_ticker_vs_market = PE_ratio/industry_average_pe
        cols.append(self.get_two_decimals(current_price/pe_ticker_vs_market))

        predicted_market_cap_based_on_industry_psratio = industry_average_ps * predicted_revenue_after_3y
        market_cap_predicted_vs_current = predicted_market_cap_based_on_industry_psratio/current_market_cap
        predicted_price_using_PS = current_price * market_cap_predicted_vs_current
        cols.append(self.get_two_decimals(predicted_price_using_PS))

        predicted_market_cap_based_on_industry_pocfratio = industry_average_pocf * predicted_ocf_after_3y
        market_cap_predicted_vs_current = predicted_market_cap_based_on_industry_pocfratio/current_market_cap
        predicted_price_using_POCF = current_price * market_cap_predicted_vs_current
        cols.append(self.get_two_decimals(predicted_price_using_POCF))

        predicted_market_cap_based_on_industry_pfcfratio = industry_average_pfcf * predicted_fcf_after_3y
        market_cap_predicted_vs_current = predicted_market_cap_based_on_industry_pfcfratio/current_market_cap
        predicted_price_using_PFCF = current_price * market_cap_predicted_vs_current
        cols.append(self.get_two_decimals(predicted_price_using_PFCF))

        # Add the entire column to the dataframe
        x = 'Price based on industry average'
        valuation_df[x] = cols
        return valuation_df
    
    # convert the numbers in a list to numbers expressed in millions
    def get_millions(self, a_list):
        b_list=[]
        for x in a_list:
            temp = self.get_two_decimals(x/1000000)
            b_list.append(temp)
        
        return b_list
    
    @staticmethod
    def fix_keyError_column(df, keyString):
        temp = 0
        try:
            temp = df[keyString]
            if temp is None:
                print(keyString, "is None in the Dataframe. Set it to 0")
                temp = 0
        except KeyError as e:
            print(keyString, "can't be found in the Dataframe. Set it to 0")
            temp = 0
        return temp
    
    @staticmethod
    def fix_keyError_row(df, keyString, size):
        temp = []
        try:
            temp = list(df.loc[keyString, :])
            if temp is None:
                print(keyString, "is None in the Dataframe. Set it to 0")
                temp = [0]*size
        except KeyError as e:
            print(keyString, "can't be found in the Dataframe. Set it to 0")
            temp = [0]*size
        return temp
    
    def fix_zero_denominator(self, numerator, denominator):
        if denominator != 0:
            temp = self.get_two_decimals(numerator/denominator)
        else:
            temp = 0
        return temp
    
    @staticmethod
    def get_two_decimals(float_number):
        # make the number only has two decimals
        format_float = float("{:.2f}".format(float_number))
        return format_float

#=======================================================================
#   Unit test
#=======================================================================
if __name__ == "__main__":
    ticker_symbol = "BROS"
    BROS_analysis = KeyFinancials(ticker_symbol)
    BROS_analysis.competitors = ['SBUX', 'MCD']
    print(BROS_analysis.history_info.cashflow)
    print(BROS_analysis.history_info.cashflow.shape)
    # BROS_analysis.plot_income_statement()
    BROS_analysis.plot_cashflow_statement()
