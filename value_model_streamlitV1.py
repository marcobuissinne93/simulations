import streamlit as st
import pandas as pd 
import numpy as np
import altair as alt
from itertools import cycle
import sqlite3
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import pandas as pd
from random import random
from typing import Dict, List


st.set_page_config(
    "LP Valuation",
    "📊",
    initial_sidebar_state="expanded",
    layout="wide",
)

# VALUATION_PERIOD: int = 5
# INVESTMENT_SIZE: float = 3_500_000
# INVESTMENT_EXPENSES_PER_DEAL: float = 500_000
# # DEAL_COUNT: int = 30 
# DEAL_FREQ: int = 5
# MANAGEMENT_FEE_PERC: float = 0.03
# DISCOUNT_RATE: float = 0.35
# MIN_FEE: int = 300_000
# MAX_FEE: int = 15_000_000





def reset_vars(): 
    con = sqlite3.connect("data.db")
    cur = con.cursor()
    cur.execute("DROP TABLE if exists variable_dist;")
    data = {'status': ['fail', 'bad', 'average', 'good', 'great'],
        'cagr': [-0.1, 0.1, 0.2, 0.4, 0.6],
        'management_fees': [300_000, 300_000, 600_000, 800_000, 1_000_000],
        'roe': [0, 2, 5, 10, 20],
        'distribution': [0.05, 0.15, 0.35, 0.35, 0.1]}
    df = pd.DataFrame(data)
    df.to_sql('variable_dist', con=con)
    con.close()
    grid_response2['data'] = df
    print(grid_response2)
    # global df_main 
    # df_main = df
    # print("updated")
    # print(df_main)


def save(df):
    print("Saving.........")
    con = sqlite3.connect("data.db")
    cur = con.cursor()
    cur.execute("DROP TABLE if exists variable_dist;")
    df.to_sql('variable_dist', con=con)
    con.close()


def get_dist_vars():
    cols = ['status', 'cagr', 'management_fees', 'roe', 'distribution']
    con = sqlite3.connect("data.db")
    cur = con.cursor()
    df = pd.DataFrame(cur.execute("SELECT * FROM variable_dist"))
    df = df.iloc[:, 1:]
    df.columns = cols
    cur.close()
    con.close()
    return df 


# @st.cache(allow_output_mutation=True)
# def fetch_data(samples):
#     deltas = cycle([
#             pd.Timedelta(weeks=-2),
#             pd.Timedelta(days=-1),
#             pd.Timedelta(hours=-1),
#             pd.Timedelta(0),
#             pd.Timedelta(minutes=5),
#             pd.Timedelta(seconds=10),
#             pd.Timedelta(microseconds=50),
#             pd.Timedelta(microseconds=10)
#             ])
#     dummy_data = {
#         "date_time_naive":pd.date_range('2021-01-01', periods=samples),
#         "apple":np.random.randint(0,100,samples) / 3.0,
#   keys      "banana":np.random.randint(0,100,samples) / 5.0,
#         "chocolate":np.random.randint(0,100,samples),
#         "group": np.random.choice(['A','B'], size=samples),
#         "date_only":pd.date_range('2020-01-01', periods=samples).date,
#         "timedelta":[next(deltas) for i in range(samples)],
#         "date_tz_aware":pd.date_range('2022-01-01', periods=samples, tz="Asia/Katmandu")
#     }
#     return pd.DataFrame(dummy_data)

#Example controlers
st.sidebar.subheader("Global Variables")

VALUATION_PERIOD: int = st.sidebar.number_input("Valuation Period", min_value=3, max_value=7, value=5)
INVESTMENT_SIZE: float = st.sidebar.number_input("Investment Size", min_value=1_000_000, max_value=7_500_000, value=3_500_000)
INVESTMENT_EXPENSES_PER_DEAL: float = st.sidebar.number_input("Expense per Deal", min_value=100_000, max_value=1_000_000, value=500_000)
# DEAL_COUNT: int = 30 
DEAL_FREQ: int = st.sidebar.number_input("Deal Frequency p.a.", min_value=2, max_value=10, value=5)
DISCOUNT_RATE: float = st.sidebar.number_input("Discount Rate", min_value=0.15, max_value=0.45, value=0.35)
MIN_FEE: int = st.sidebar.number_input("Min Management Fee", min_value=200_000, max_value=300_000, value=300_000)
MAX_FEE: int = st.sidebar.number_input("Max Management Fee", min_value=10_000_000, max_value=20_000_000, value=15_000_000)


simulation_count = st.sidebar.number_input("Simulations", min_value=5, max_value=100_000, value=5)



df_main = get_dist_vars()
gb_main = GridOptionsBuilder.from_dataframe(df_main)
gb_main.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
gb_main.configure_side_bar()
gb_main.configure_grid_options(domLayout='normal')
gridOptions_main = gb_main.build()

df_main[['status', 'distribution']].to_dict('records')



def get_random_key(df) -> str:
    rand: float = random()
    cum_dist: List[float] = []
    cum_value: float = 0.0
    keys = []
    dist: Dict = dict() 
    for x in df[['status', 'distribution']].to_dict('records'):
        dist.update({x['status']: x['distribution']})
        keys.append(x['status'])
    for _, val in dist.items():
        cum_value += val
        cum_dist.append(cum_value)
    for x in cum_dist:
        if rand <= x:
            # print(list(CAGR_VALUES.keys())[cum_dist.index(x)])
            # print(rand)
            return keys[cum_dist.index(x)]

class Simulate:
    def __init__(self, period, inv_size, expense, deal_freq, disc_rate, min_fee, max_fee):
        self.period = period 
        self.inv_size = inv_size
        self.expense = expense
        self.deal_freq = deal_freq 
        self.disc_rate = disc_rate 
        self.min_fee = min_fee
        self.max_fee = max_fee 


    def run(self, n_sims, df):
        final = pd.DataFrame()
        for k in range(n_sims):
            self.results = []
            successes: int = 0
            failures: int = 0
            for i in range(1, VALUATION_PERIOD+1):
                for j in range(DEAL_FREQ):
                    deal = Investment(df, i, self.period, self.disc_rate, self.inv_size, self.max_fee)
                    if deal.cagr_rand == 'fail':
                        failures += 1
                    else: 
                        successes += 1
                    terminal_management_fee = deal.terminal_fee
                    result = {'year': i, 'deal_no': j+1, 
                            'investment_size': self.inv_size, 'expenses': self.expense,
                            'initial_fee': deal.initial_fee, 'cagr': deal.cagr, 
                            'terminal_fee': terminal_management_fee, 'equity_valuation': deal.investment_value}
                    self.results.append(result)
            data = pd.DataFrame(self.results)
            tmp = pd.DataFrame([{'sim_no': k, 'successes': successes, 'failures': failures,
                        'total_invested': data['investment_size'].sum(),
                        'total_expenses': data['expenses'].sum(),
                        'total_terminal_fee': data['terminal_fee'].sum(),
                        'total_equity_valuation': data['equity_valuation'].sum(),
                        'man_fee_per_investment':  data['terminal_fee'].sum()/successes,
                        'premium_inc_per_investment':  data['terminal_fee'].sum()/0.03/successes}])
            if k == 0:      
                final = tmp
            else: 
                final = pd.concat([final, tmp])
            my_bar.progress((k+1)/n_sims)
        my_bar.empty()
        return final

    def summary(self):
        ...




class Investment:
    def __init__(self, distribution_table: pd.DataFrame, investment_year: int, 
                period: int, discount_rate: float, inv_size: float, max_fee: float):
        self.year = investment_year
        self.cagr_rand, self.fee_rand = get_random_key(distribution_table), get_random_key(distribution_table)
        self.investment_rand = self.cagr_rand # ROI is assumed to be driven by the CAGR
        self.cagr = distribution_table.query("status == @self.cagr_rand")['cagr'].values[0] # CAGR_VALUES[self.cagr_rand]
        self.initial_fee = distribution_table.query("status == @self.fee_rand")["management_fees"].values[0]
        self.roi = distribution_table.query("status == @self.investment_rand")["roe"].values[0]
        self._terminal_fee: float = 0.0
        self._investment_value: float = 0.0
        self.period = period 
        self.discount_rate = discount_rate
        self.inv_size = inv_size
        self.max_fee = max_fee 

    def calc_terminal_fee(self):
        if self.cagr_rand == 'fail':
            self._terminal_fee = 0
        else:
            self._terminal_fee = min(self.initial_fee * pow((1+self.cagr), self.period-self.year+1), self.max_fee)


    def calc_investment_value(self):
        self._investment_value = self.inv_size * self.roi * pow(1+self.discount_rate, -(self.year-1))

    @property 
    def terminal_fee(self):
        if self._terminal_fee == 0:
            self.calc_terminal_fee()
        return self._terminal_fee

    @property
    def investment_value(self):
        self.calc_investment_value()
        return self._investment_value




# grid_height = st.sidebar.number_input("Grid height", min_value=200, max_value=800, value=300)

# return_mode = st.sidebar.selectbox("Return Mode", list(DataReturnMode.__members__), index=1)
# return_mode_value = DataReturnMode.__members__[return_mode]

# update_mode = st.sidebar.selectbox("Update Mode", list(GridUpdateMode.__members__), index=6)
# update_mode_value = GridUpdateMode.__members__[update_mode]

#enterprise modules
# enable_enterprise_modules = st.sidebar.checkbox("Enable Enterprise Modules")
# if enable_enterprise_modules:
#     enable_sidebar =st.sidebar.checkbox("Enable grid sidebar", value=False)
# else:
#     enable_sidebar = False

#features
# fit_columns_on_grid_load = st.sidebar.checkbox("Fit Grid Columns on Load")

# enable_selection=st.sidebar.checkbox("Enable row selection", value=True)
# if enable_selection:
#     st.sidebar.subheader("Selection options")
#     selection_mode = st.sidebar.radio("Selection Mode", ['single','multiple'], index=1)
    
#     use_checkbox = st.sidebar.checkbox("Use check box for selection", value=True)
#     if use_checkbox:
#         groupSelectsChildren = st.sidebar.checkbox("Group checkbox select children", value=True)
#         groupSelectsFiltered = st.sidebar.checkbox("Group checkbox includes filtered", value=True)

#     if ((selection_mode == 'multiple') & (not use_checkbox)):
#         rowMultiSelectWithClick = st.sidebar.checkbox("Multiselect with click (instead of holding CTRL)", value=False)
#         if not rowMultiSelectWithClick:
#             suppressRowDeselection = st.sidebar.checkbox("Suppress deselection (while holding CTRL)", value=False)
#         else:
#             suppressRowDeselection=False
#     st.sidebar.text("___")

# enable_pagination = st.sidebar.checkbox("Enable pagination", value=False)
# if enable_pagination:
#     st.sidebar.subheader("Pagination options")
#     paginationAutoSize = st.sidebar.checkbox("Auto pagination size", value=True)
#     if not paginationAutoSize:
#         paginationPageSize = st.sidebar.number_input("Page size", value=5, min_value=0, max_value=sample_size)
#     st.sidebar.text("___")


# gb_main.configure_column("date_tz_aware", type=["dateColumnFilter","customDateTimeFormat"], custom_format_string='yyyy-MM-dd HH:mm zzz', pivot=True)
# gb_main.configure_column("apple", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2, aggFunc='sum')
# gb_main.configure_column("banana", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=1, aggFunc='avg')
# gb_main.configure_column("chocolate", type=["numericColumn", "numberColumnFilter", "customCurrencyFormat"], custom_currency_symbol="R$", aggFunc='max')


# df = fetch_data(sample_size)

# #Infer basic colDefs from dataframe types
# gb = GridOptionsBuilder.from_dataframe(df)

# #customize gridOptions
# gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)

# gb.configure_column("date_tz_aware", type=["dateColumnFilter","customDateTimeFormat"], custom_format_string='yyyy-MM-dd HH:mm zzz', pivot=True)

# gb.configure_column("apple", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=2, aggFunc='sum')
# gb.configure_column("banana", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=1, aggFunc='avg')
# gb.configure_column("chocolate", type=["numericColumn", "numberColumnFilter", "customCurrencyFormat"], custom_currency_symbol="R$", aggFunc='max')

# #configures last row to use custom styles based on cell's value, injecting JsCode on components front end
# cellsytle_jscode = JsCode("""
# function(params) {
#     if (params.value == 'A') {
#         return {
#             'color': 'white',
#             'backgroundColor': 'darkred'
#         }
#     } else {
#         return {
#             'color': 'black',
#             'backgroundColor': 'white'
#         }
#     }
# };
# """)
# gb.configure_column("group", cellStyle=cellsytle_jscode)

# if enable_sidebar:
#     gb.configure_side_bar()

# if enable_selection:
#     gb.configure_selection(selection_mode)
#     if use_checkbox:
#         gb.configure_selection(selection_mode, use_checkbox=True, groupSelectsChildren=groupSelectsChildren, groupSelectsFiltered=groupSelectsFiltered)
#     if ((selection_mode == 'multiple') & (not use_checkbox)):
#         gb.configure_selection(selection_mode, use_checkbox=False, rowMultiSelectWithClick=rowMultiSelectWithClick, suppressRowDeselection=suppressRowDeselection)

# if enable_pagination:
#     if paginationAutoSize:
#         gb.configure_pagination(paginationAutoPageSize=True)
#     else:
#         gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=paginationPageSize)

# gb.configure_grid_options(domLayout='normal')
# gridOptions = gb.build()

#Display the grid
st.header("Launchpad Valuation Simulation")
st.subheader("Simulation Parameters")
st.caption("""The values in the table below can be altered by the user. Please ensure that the 'Distribution' column sums to 1.
            For scenario analysis change the 'CAGR', 'Management_Fees' or 'Distribution' columns. Increasing the 'Fail' distribution
            value to 0.4 would be a good way to conduct a stress test.""")


# grid_response = AgGrid(
#     df, 
#     gridOptions=gridOptions,
#     height=grid_height, 
#     width='100%',
#     data_return_mode=return_mode_value, 
#     update_mode=update_mode_value,
#     fit_columns_on_grid_load=fit_columns_on_grid_load,
#     allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
#     enable_enterprise_modules=enable_enterprise_modules,
#     )

grid_response2 = AgGrid(
    df_main, 
    gridOptions=gridOptions_main,
    height='180px', 
    # data_return_mode=return_mode_value, 
    # update_mode=update_mode_value,
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
    enable_enterprise_modules=True,
    theme='dark'
    )

if 0.99 <= grid_response2['data']['distribution'].sum() <= 1:
    print("yes")
    st.success("Success: The sum of the 'distribution' column sums to 1")
else:
    print("no")
    st.error(f"Error: The 'distribution' column sums to {round(grid_response2['data']['distribution'].sum(),2)}, but must be equal to 1.")

# AgGrid
col1, col2 = st.columns(2)
with col1:
    if st.button("Persist Updates"):
        save(grid_response2['data'])
with col2: 
    if st.button("Reset"):
        reset_vars()



my_bar = st.progress(0)
x = Simulate(VALUATION_PERIOD, INVESTMENT_SIZE, INVESTMENT_EXPENSES_PER_DEAL, DEAL_FREQ, DISCOUNT_RATE, MIN_FEE, MAX_FEE).run(simulation_count, df_main)

def get_js(field):
    js_formatter = '''
            function(params){
                return params.data.field_name.toLocaleString(undefined, { 
            minimumFractionDigits: 0, 
            maximumFractionDigits: 0
            });
            }
            '''
    js_formatter = js_formatter.replace("field_name", field)
    return JsCode(js_formatter)


gb_x = GridOptionsBuilder.from_dataframe(x)
gb_x.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
gb_x.configure_side_bar()
gb_x.configure_grid_options(domLayout='normal', )
gb_x.configure_column("sim_no", header_name="Sim No", valueFormatter=get_js("sim_no"),  maxWidth = 100)
gb_x.configure_column("successes", header_name="Successes",  maxWidth = 130)
gb_x.configure_column("failures", header_name="Failures",  maxWidth = 130)
gb_x.configure_column("total_invested", header_name="Total Invested", valueFormatter=get_js("total_invested"))
gb_x.configure_column("total_expenses", header_name="Total Expenses", valueFormatter=get_js("total_expenses"))
gb_x.configure_column("total_terminal_fee", header_name="Management Fee Income", valueFormatter=get_js("total_terminal_fee"))
gb_x.configure_column("total_equity_valuation", header_name="Equity Valuation", valueFormatter=get_js("total_equity_valuation"))
gb_x.configure_column("man_fee_per_investment", header_name="Avg Management Fee", valueFormatter=get_js("man_fee_per_investment"))
gb_x.configure_column("premium_inc_per_investment", header_name="Average Premium", valueFormatter=get_js("premium_inc_per_investment"))
# gb_x = {
#     "columnDefs": [
#         {
#             "headerName": "Sim Number",
#             "field": "sim_no",
#             "editable": False
#         },
#         {
#             "headerName": "Total Invested",
#             "field": "total_invested",
#             "editable": False
#         },
#         {
#             "headerName": "Total Expenses",
#             "field": "total_expenses",
#             "editable": False,
#             "valueFormatter": JsCode("params => params.data.total_expenses/10000")
#         },
#         {
#             "headerName": "Total Terminal Fees",
#             "field": "total terminal fee",
#             "editable": False,
#             "valueFormatter": JsCode("params => params.data.number.toFixed(2)")
#         },
#         {
#             "headerName": "Total Equity Valuation",
#             "field": "total Equity Valuation",
#             "editable": False,
#             "valueFormatter": JsCode("params => params.data.number.toFixed(2)")
#         }
#     ]
# }
# gb_x.configure_column("sim_no", header_name="Sim No", editable=False, valueFormatter = params => params.data.number.toFixed(2),)
gridOptions_x = gb_x.build()

# print(gridOptions_x)

st.subheader("Simulation Output")

with st.expander("Simulation Output Table"):
    sim_result = AgGrid(
        x, 
        gridOptions=gridOptions_x,
        height='300px', 
        width='400px',
        # data_return_mode=return_mode_value, 
        # update_mode=update_mode_value,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
        enable_enterprise_modules=True,
        theme='dark'
        )

x['sim_no'] = x['sim_no'].astype(str)

# c = alt.Chart(x).mark_circle().encode(
#      x='sim_no', y='total_terminal_fee', tooltip=['sim_no', 'total_terminal_fee'])



# c = px.histogram(x, x="total_terminal_fee", histnorm='probability density', nbins=30)
# c = alt.Chart(x).mark_bar().encode(
#     alt.X("sim_no:Q", bin=True),
#     y='count()',
# )

c = x['total_terminal_fee'].hist(bins=30, backend='plotly')
c2 = x['total_equity_valuation'].hist(bins=30, backend='plotly')
dist_col1, dist_col2 = st.columns(2)

with st.expander("Management Fee Distribution"):
    st.plotly_chart(c, use_container_width=True)

with st.expander("Equity Value Distribution"):
    st.plotly_chart(c2, use_container_width=True)



# num formatting function
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'B', 'T', 'P'][magnitude])


st.subheader("Valuation Summary")

col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)

with col_summary1:
    st.metric("Total investment per Sim", human_format(x['total_invested'].mean()))
with col_summary2:
    st.metric("Total expense per Sim", human_format(x['total_expenses'].mean()))
with col_summary3:
    st.metric("Average Management Fee Income", human_format(x['total_terminal_fee'].mean()))
with col_summary4:
    st.metric("Average Equity Valuation", human_format(x['total_equity_valuation'].mean()))

col_summary5, col_summary6, col_summary7, col_summary8 = st.columns(4)

with col_summary5:
    st.metric("Average Required Management fee per cell", human_format(x['man_fee_per_investment'].mean()))
with col_summary6:
    st.metric("Average Required Premium income per cell", human_format(x['premium_inc_per_investment'].mean()))
with col_summary7:
    st.metric("Avg. Successes", f"{x['successes'].mean()}")
with col_summary8:
    st.metric("Avg. Failures", f"{x['failures'].mean()}")

            # tmp = pd.DataFrame([{'sim_no': k, 'successes': successes, 'failures': failures,
            #             'total_invested': data['investment_size'].sum(),
            #             'total_expenses': data['expenses'].sum(),
            #             'total_terminal_fee': data['terminal_fee'].sum(),
            #             'total_equity_valuation': data['equity_valuation'].sum(),
            #             'man_fee_per_investment':  data['terminal_fee'].sum()/successes,
            #             'premium_inc_per_investment':  data['terminal_fee'].sum()/0.03/successes}])


