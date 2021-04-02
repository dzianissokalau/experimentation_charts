import numpy as np
from scipy import stats
import matplotlib.pyplot as plt




def plot_cvr_distributions(control, experimental, alpha=0.05, tails=2):
    """Parameters:
        control, experimental: an array of 1 and 0, where 1 indicates success.
        alpha: significance level (between 0 and 1), 0.05 by default.
        tails: one-tailed comparison (1), two-tailed comparison (2). 2 is default value.
        
    Returns:
        None: plot distribution of experimental and control with annotations.
        
    Example:
        control = np.array([1,0,0,1,0,1,0,0,0,1,0])
        experimental = np.array([0,1,1,1,0,0,1,0,0,1,1])
        plot_cvr_distributions(control, experimental, alpha=0.05, tails=1)
    """
    # calculate trials
    control_trials = len(control)
    experimental_trials = len(experimental)
    
    # calculate conversion rates
    control_cvr = control.mean()
    experimental_cvr = experimental.mean()
    
    # calculate standard deviation
    population_std = np.sqrt((1 / control_trials + 1 / experimental_trials) * control_cvr * (1 - control_cvr)) # np.sqrt(1 / control_trials * control_cvr * (1 - control_cvr))
    experimental_std = np.sqrt(1 / experimental_trials * experimental_cvr * (1 - experimental_cvr))
    
    # calculate values for x axis and y axis
    x_control = np.linspace(control_cvr - 4 * population_std, control_cvr + 4 * population_std, 1000)
    x_experimental = np.linspace(experimental_cvr - 4 * experimental_std, experimental_cvr + 4 * experimental_std, 1000)

    y_control = stats.norm(control_cvr, population_std).pdf(x_control)
    y_experimental = stats.norm(experimental_cvr, experimental_std).pdf(x_experimental)
    
    # create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # control
    ax.plot(x_control, y_control, label='control', color='grey')
    
    # experimental
    ax.plot(x_experimental, y_experimental, label='experimental', color='green')
    
    if tails==2:
        # right tail
        ax.fill_between(
            x_control, 0, 
            stats.norm(control_cvr, population_std).pdf(x_control), 
            color='green', alpha=0.25, 
            where=(x_control >= stats.norm(control_cvr, population_std).ppf(1-alpha/2)))

        ax.vlines(
            stats.norm(control_cvr, population_std).ppf(1-alpha/2), 
            ymin=0, 
            ymax=stats.norm(control_cvr, population_std).pdf(stats.norm(control_cvr, population_std).ppf(1-alpha/2)), 
            color='grey', linewidth=1, linestyle='--')

        # left tail
        ax.fill_between(
            x_control, 0, 
            stats.norm(control_cvr, population_std).pdf(x_control), 
            color='green', alpha=0.25, 
            where=(x_control <= stats.norm(control_cvr, population_std).ppf(alpha/2)))

        ax.vlines(
            stats.norm(control_cvr, population_std).ppf(alpha/2), 
            ymin=0, 
            ymax=stats.norm(control_cvr, population_std).pdf(stats.norm(control_cvr, population_std).ppf(alpha/2)), 
            color='grey', linewidth=1, linestyle='--')
    
    else:
        # right tail
        ax.fill_between(
            x_control, 0, 
            stats.norm(control_cvr, population_std).pdf(x_control), 
            color='green', alpha=0.25, 
            where=(x_control >= stats.norm(control_cvr, population_std).ppf(1-alpha)))

        ax.vlines(
            stats.norm(control_cvr, population_std).ppf(1-alpha), 
            ymin=0, 
            ymax=stats.norm(control_cvr, population_std).pdf(stats.norm(control_cvr, population_std).ppf(1-alpha)), 
            color='grey', linewidth=1, linestyle='--')        
        
        
    # set max and min y
    ymax = y_experimental.max() if y_experimental.max() > y_control.max() else y_control.max()
    ax.set(ylim=(ymax*-0.1, ymax*1.1))
    
    # create hline for y==0
    ax.hlines(
        0, 
        xmin=np.min([x_control.min(), x_experimental.min()]), 
        xmax=np.max([x_control.max(), x_experimental.max()]), 
        color='black')
    
    # annotate experimental cvr
    ax.vlines(
        experimental_cvr, 
        ymin=0, 
        ymax=y_experimental.max(), 
        color='grey', linewidth=1, linestyle='--')
    
    plt.plot(experimental_cvr, 0, marker='o', color='black')
    
    plt.annotate(
        str(round(experimental_cvr, 4)), 
        (experimental_cvr, ymax*-0.07),
        ha='center',
        fontsize=12)
    
    # annotate control cvr
    ax.vlines(
        control_cvr, 
        ymin=0, 
        ymax=y_control.max(), 
        color='grey', linewidth=1, linestyle='--')
    
    plt.plot(control_cvr, 0, marker='o', color='black')

    plt.annotate(
        str(round(control_cvr, 4)), 
        (control_cvr, ymax*-0.07),
        ha='center',
        fontsize=12)
        
    ax.legend(fontsize=12)
    ax.set_title('CVR: experimental vs control', fontdict={'size':22})
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    plt.show()  