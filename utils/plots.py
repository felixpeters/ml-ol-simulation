import matplotlib.pyplot as plt

# Set default figure size and resolution
plt.rcParams["figure.figsize"] = (15,9)
plt.rcParams["figure.dpi"] = 100

def plot_time_series(ts, 
                     dep_var='code_kl', 
                     query_vars=['num_ml', 'p_ml', 'p_1', 'p_2'],
                     x_var='step',
                     plot_var='p_3',
                     yticks=np.arange(0, 1.05, step=0.05),
                     fname='test',
                     save_path='data/'):
    """
    Creates time series plots for one variable and stores them in HTML file.

    Args:
        - ts: Reindexed pandas DataFrame with time series
        - dep_var: Dependent variable, i.e., y-axis
        - query_vars: Variables that determine number of created plots
        - x_var: Independent variable, i.e., x-axis
        - plot_var: Variable that determines different lines in the plot
        - yticks: Scale for y-axis
        - fname: Name of created HTML file (dependent variable name will be
          added)
        - save_path: Path to directory where output file will be stored
    """
    # get all possible parameter combinations
    queries = []
    for var in query_vars:
        queries.append(pd.unique(ts[var]))
        
    html = f"""
    <h1>Time series analysis for {dep_var}</h1>
    """
    # iterate of all parameter combinations
    for query_vals in product(*queries):
        # Get relevant configurations
        query = ' & '.join([f'({q[0]} == {q[1]})' for q in zip(query_vars, query_vals)])
        configs = np.unique(ts.query(query)['config'].values)
        df = ts.loc[ts['config'].isin(configs)]
        # Calculate average knowledge level for each level of plot_var and step
        kls = df.groupby([plot_var, x_var]).mean()[dep_var]
        # Plot different levels of plot_var on new figure
        plt.figure()
        for p in kls.index.unique(level=0):
            plt.plot(kls.loc[p].index, kls.loc[p], label=f"{plot_var}={p}")
        plt.title(', '.join([f'{q[0]}={q[1]}' for q in zip(query_vars, query_vals)]))
        plt.legend()
        plt.xlabel(f'{x_var}')
        plt.ylabel(f'avg({dep_var})')
        plt.yticks(yticks)
        plt.hlines(yticks, ts[x_var].min(), ts[x_var].max(), colors='lightgray')
        # Save created plot to temporary file
        tmpfile = BytesIO()
        plt.savefig(tmpfile, format='png')
        plt.close()
        # Add plot to HTML file
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html += f"<h2>Configuration: {', '.join([f'{q[0]}={q[1]}' for q in zip(query_vars, query_vals)])}</h2>"
        html += f"<img src=\'data:image/png;base64,{encoded}\'>"
    # Save HTML file to specified directory
    with open(f'{save_path}{fname}_{dep_var}.html','w') as f:
        f.write(html)
    return

def plot_summary(agg, 
                 dep_var='code_kl_last',
                 query_vars=['num_ml', 'p_ml'],
                 x_var='p_3',
                 plot_vars=['p_1', 'p_2'],
                 yticks=np.arange(0, 1.05, 0.05),
                 fname='agg_test',
                 save_path='data/'):
    """
    Creates summary plots for one variable and stores them in HTML file.

    Args:
        - agg: Reindexed pandas DataFrame with aggregated simulation data
        - dep_var: Dependent variable, i.e., y-axis
        - query_vars: Variables that determine number of created plots
        - x_var: Independent variable, i.e., x-axis
        - plot_var: Variable that determines different lines in the plot
        - yticks: Scale for y-axis
        - fname: Name of created HTML file (dependent variable name will be
          added)
        - save_path: Path to directory where output file will be stored
    """
    queries = []   
    for var in query_vars:
        queries.append(pd.unique(agg[var]))
        
    html = f"""
    <h1>Summary analysis for {dep_var}</h1>
    """
    for query_vals in product(*queries):
        query = ' & '.join([f'({q[0]} == {q[1]})' for q in zip(query_vars, query_vals)])
        configs = np.unique(agg.query(query)['config'].values)
        df = agg.loc[agg['config'].isin(configs)]
        kls = df.groupby(plot_vars + [x_var]).mean()[dep_var]
        plot_queries = []
        for i in range(len(plot_vars)):
            plot_queries.append(kls.index.unique(level=i))
        plt.figure()
        for p in product(*plot_queries):
            plt.plot(kls.loc[p].index, kls.loc[p], label=f"{', '.join([f'{q[0]}={q[1]}' for q in zip(plot_vars, p)])}")
        plt.title(', '.join([f'{q[0]}={q[1]}' for q in zip(query_vars, query_vals)]))
        plt.legend()
        plt.yticks(yticks)
        plt.xticks(pd.unique(agg[x_var]))   
        plt.hlines(yticks, agg[x_var].min(), agg[x_var].max(), colors='lightgray')             
        plt.xlabel(f'{x_var}')
        plt.ylabel(f'avg({dep_var})')
        tmpfile = BytesIO()
        plt.savefig(tmpfile, format='png')
        plt.close()
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html += f"<h2>Configuration: {', '.join([f'{q[0]}={q[1]}' for q in zip(query_vars, query_vals)])}</h2>"
        html += f"<img src=\'data:image/png;base64,{encoded}\'>"
    with open(f'{save_path}{fname}_{dep_var}.html','w') as f:
        f.write(html)
    return
