import os
import numpy as np
import matplotlib.pyplot as plt

def plot_1D_regression(context_x, context_y, target_x, target_y, *args, **kargs):
    '''
        Plots the predictive distribution (mean, var) and context points

        Args:
            All arguments are "NP arrays"
            target_x : [batch_size, the number of total_points, x_size(dimension)] 
            target_y : [batch_size, the number of total_points, y_size(dimension)] 
            context_x : [batch_size, the number of context_points, x_size(dimension)] 
            context_y : [batch_size, the number of context_points, y_size(dimension)] 
            pred_y : [batch_size, the number of total_point, y_size(dimension)] same as target_y
            var  : [batch_size, the number of total_point, y_size(dimension)] same as var
    '''
    # Path
    result_path = kargs.get("result_path", "./")
    file_name = kargs.get("file_name", "regression_result.png")
    FILE_NAME = os.path.join(result_path, file_name)

    # Pred
    pred = kargs.get("pred_mu", None)
    std = kargs.get("pred_sigma", None)
    
    # Shape
    task_size, n_points, _ = context_x.shape

    # Idx
    idx = np.random.choice(task_size, size=1)
    
    # scatter plot
    plt.plot(np.squeeze(context_x[idx],axis=0), np.squeeze(context_y[idx], axis=0), 'ko', markersize=5)

    # Line plot
    plt.plot(np.squeeze(target_x[idx], axis=0), np.squeeze(target_y[idx], axis=0), 'k:', linewidth=1)
    if pred is not None and std is not None:
        # line plot
        plt.plot(np.squeeze(target_x[idx], axis=0), np.squeeze(pred[idx], axis=0), 'b', linewidth=2)

        # var
        plt.fill_between(np.squeeze(target_x[idx]), \
            np.squeeze(pred[idx]) - np.squeeze(std[idx]),\
            np.squeeze(pred[idx]) + np.squeeze(std[idx]),
            alpha = 0.2,
            facecolor='#65c9f7',
            interpolate=True
        )
        

    # Make a plot pretty
    TITLE = kargs.get("title", "1D regression plot")
    ITERATION = kargs.get("iteration", None)

    # Title
    plt.title(TITLE, fontweight='bold', loc='center')
    
    # Label
    plt.xlabel("{} iterations".format(ITERATION)) if not ITERATION is None else _
    
    # Axis
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-4, 0, 4], fontsize=16)
    plt.ylim([-1, 1])
    plt.grid(False)
    ax = plt.gca()
    ax.set_facecolor('white')

    # Save
    plt.savefig(FILE_NAME, dpi=300)

    # Close plt
    plt.clf()
