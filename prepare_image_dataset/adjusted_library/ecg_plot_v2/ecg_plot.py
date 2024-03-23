#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os
from math import ceil 
import random

def _ax_plot(ax, x, y, secs=10, lwidth=0.5, amplitude_ecg = 1.8, time_ticks =0.2):
    # ax.set_xticks(np.arange(0,11,time_ticks))    
    # ax.set_yticks(np.arange(-ceil(amplitude_ecg),ceil(amplitude_ecg),1.0))

    #ax.set_yticklabels([])
    #ax.set_xticklabels([])

    ax.minorticks_on()
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    ax.set_ylim(-amplitude_ecg, amplitude_ecg)
    ax.set_xlim(0, secs)

    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))

    ax.plot(x,y, linewidth=lwidth)


lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
def plot_12(
        ecg, 
        sample_rate = 500, 
        title       = 'ECG 12', 
        lead_index  = lead_index, 
        lead_order  = None,
        columns     = 2
        ):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on 
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order 
        columns    : display columns, defaults to 2
    """
    if not lead_order:
        lead_order = list(range(0,len(ecg)))

    leads = len(lead_order)
    seconds = len(ecg[0])/sample_rate

    plt.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots(
        ceil(len(lead_order)/columns),columns,
        sharex=True, 
        sharey=True,
        figsize=(0.7*seconds*columns, 1.1*leads/columns)
        )
    fig.subplots_adjust(
        hspace = 0, 
        wspace = 0.04,
        left   = 0.04,  # the left side of the subplots of the figure
        right  = 0.98,  # the right side of the subplots of the figure
        bottom = 0.06,  # the bottom of the subplots of the figure
        top    = 0.95
        )
    fig.suptitle(title)

    step = 1.0/sample_rate

    for i in range(0, len(lead_order)):
        if(columns == 1):
            t_ax = ax[i]
        else:
            t_ax = ax[i//columns,i%columns]
        t_lead = lead_order[i]
        t_ax.set_ylabel(lead_index[t_lead])
        t_ax.tick_params(axis='x',rotation=90)
       
        _ax_plot(t_ax, np.arange(0, len(ecg[t_lead])*step, step), ecg[t_lead], seconds)

def plot(
        ecg, 
        full_ecg_name,        # changed to let us have (or don't have) full ecg in the printed format
        full_ecg,             # changed
        
        sample_rate    = 500, 
        title          = 'ECG 12', 
        lead_index     = lead_index, 
        lead_order     = None,
        style          = None,
        columns        = 2,
        row_height     = 6,
        show_lead_name = True,
        show_grid      = True,
        show_separate_line  = True,
        ):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        full_ecg_name: the name of the full ecg. if `None`, then no lead wil be printed completely.
        full_ecg   : signal of the lead that you want to be printed completely. It works if full_ecg_name is assigned
        
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        lead_index : Lead name array in the same order of ecg, will be shown on 
            left of signal plot, defaults to ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        lead_order : Lead display order 
        columns    : display columns, defaults to 2
        style      : display style, defaults to None, can be 'bw' which means black white
        row_height :   how many grid should a lead signal have,
        show_lead_name : show lead name
        show_grid      : show grid
        show_separate_line  : show separate line
    """

    if not lead_order:
        lead_order = list(range(0,len(ecg)))
    secs  = len(ecg[0])/sample_rate
    leads = len(lead_order)
    rows  = int(ceil(leads/columns)) 
    display_factor = 1
    # display_factor = display_factor ** 0.5
    mm = 1/25.4  # milimeters in inches    # added to specify values in milimeter
    line_width = 4*mm/25.4*72       # added to determine the line width in milimeters. lind width in milimeters is devided by 25.4*72 to be converted to unit of points.
    
    d_column = 40/45*mm
    # determine figure size 
    n_empty_cell_at_left = 3.25   # number of free cells at the least left
    n_empty_cell_at_right = 7 
    n_empty_cell_at_up = 4 
    n_empty_cell_at_down = 4
    fig_width = (n_empty_cell_at_left)*5*mm + 25*secs*columns*mm + (columns-1)*d_column + (n_empty_cell_at_right)*5*mm   # added
    if full_ecg_name:
        fig_height = (n_empty_cell_at_up + n_empty_cell_at_down)*5*mm + (rows)*row_height*5*mm + 0.05   # added
    else:
        fig_height = (n_empty_cell_at_up + n_empty_cell_at_down)*5*mm + (rows-1)*row_height*5*mm    # added

    figsize = (fig_width, fig_height)  # added
    fig, ax = plt.subplots(figsize=figsize, dpi= 700)

    # fig.subplots_adjust(
    #     hspace = 0, 
    #     wspace = 0,
    #     left   = 0,  # the left side of the subplots of the figure
    #     right  = 1,  # the right side of the subplots of the figure
    #     bottom = 0,  # the bottom of the subplots of the figure
    #     top    = 1
    #     )

    fig.suptitle(title)

    
    x_min = - 0.2 * n_empty_cell_at_left                                            # changed to put free space (3 cell which is equal to 0.6 seconds or 15 milimeters) in the left part of the least left lead   چپ ترین
    x_max = columns*secs + (columns-1)*40/45*0.2 + (n_empty_cell_at_right * 0.2)         # changed  to put free space in the right part of the most right lead   راست ترین
    if full_ecg_name:
        y_min = - 0.5*(n_empty_cell_at_down + 2) - (rows)*row_height*0.5         # changed to put free space in the down of the picture 
    else:
        y_min = -0.5*(n_empty_cell_at_down) - (rows-1)*row_height*0.5
    y_max = 0.5 * (n_empty_cell_at_down)                                 # changed to put free space in the above of the picture

    if (style == 'bw'):
        color_major = (0.4,0.4,0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line  = (0,0,0)
    else:
        color_major = (0.75, 0.5, 0.5) # red            # changed to have different grid line
        color_minor = (1, 0.93, 0.93)                   # changed
        color_line  = (0,0,0) # black                   # changed
    
    if(show_grid):
        ax.set_xticks(np.arange(x_min,x_max,0.2))    
        ax.set_yticks(np.arange(y_min,y_max,0.5))
        # disable the xtickslabels
        ax.set_xticklabels([])                    # changed
        ax.set_yticklabels([])                    # changed

        # disable the frame around the plot
        ax.spines['top'].set_color('none')        # changed
        ax.spines['bottom'].set_color('none')     # changed
        ax.spines['left'].set_color('none')       # changed
        ax.spines['right'].set_color('none')      # changed

        # disable the minor and major ticks
        ax.tick_params(which='both', width=2, color='none')   # changed

        ax.minorticks_on()  
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    # ax.set_aspect('equal')      # changed to have same scaling from data to plot units for x and y
    ax.set_ylim(y_min,y_max)
    ax.set_xlim(x_min,x_max)

    output_log = {
        "figsize": figsize,
        "y_min": y_min,
        "y_max": y_max,
        "x_min": x_min,
        "x_max": x_max,
        "leads": []
    }

    for c in range(0, columns):
        for i in range(0, rows):
            if (c * rows + i < leads):
                y_offset = -(row_height/2) * ceil(i%rows)
                # if (y_offset < -5):
                #     y_offset = y_offset + 0.25

                x_offset = 0
                if(c > 0):
                    x_offset = secs * c + (c)*d_column       # changed to have 1/5 space between two columns
                    if(show_separate_line):

                        # below was changed to have the vertical seprator line in a broken format
                        ax.plot([x_offset-0.5*d_column, x_offset-0.5*d_column], [ecg[t_lead][0] + y_offset - 0.5, ecg[t_lead][0] + y_offset - 0.2], linewidth=line_width * display_factor, color=color_line)  # changed
                        ax.plot([x_offset-0.5*d_column, x_offset-0.5*d_column], [ecg[t_lead][0] + y_offset + 0.2, ecg[t_lead][0] + y_offset + 0.5], linewidth=line_width * display_factor, color=color_line)  # changed

         
                t_lead = lead_order[c * rows + i]
         
                step = 1.0/sample_rate
                if(show_lead_name):
                    x_text = x_offset + 0.01
                    y_text = y_offset + 0.8
                    content_text = lead_index[t_lead]
                    fontsize = 9 * display_factor
                    ax.text(x_text, y_text, content_text, fontsize=fontsize)
                
                x_plot = np.arange(0, len(ecg[t_lead])*step, step) + x_offset
                y_plot = ecg[t_lead] + y_offset
                ax.plot(
                    x_plot, 
                    y_plot,
                    linewidth=line_width * display_factor, 
                    color=color_line
                    )

                output_log['leads'].append({
                    "min_x_plot": min(x_plot),
                    "max_x_plot": max(x_plot),
                    "min_y_plot": min(y_plot),
                    "max_y_plot": max(y_plot),
                    "x_text": x_text,
                    "y_text": y_text,
                    "fontsize": fontsize,
                    "text": content_text,
                    "ecg": list(ecg[t_lead]),
                })                
    ## below was added to have full ecg in the last row. changed
    if full_ecg_name: 
        y_offset_full_ecg = - ((row_height/2) * ceil((i+1)%(rows+1)))
        if(show_lead_name):
            x_text = 0 + 0.01
            y_text = y_offset_full_ecg + 0.8
            fontsize = 9 * display_factor
            content_text = full_ecg_name
            ax.text(x_text, y_text, full_ecg_name, fontsize=9 * display_factor)
        x_plot = np.arange(0, len(full_ecg)*step, step) + 0
        y_plot = full_ecg + y_offset_full_ecg
        ax.plot( 
                x_plot,
                y_plot,
                linewidth=line_width * display_factor, 
                color=color_line
                )
        output_log['leads'].append({
            "min_x_plot": min(x_plot),
            "max_x_plot": max(x_plot),
            "min_y_plot": min(y_plot),
            "max_y_plot": max(y_plot),
            "x_text": x_text,
            "y_text": y_text,
            "fontsize": fontsize,
            "text": content_text,
            "ecg": list(full_ecg),
        })


    return output_log
        

def plot_1(ecg, sample_rate=500, title = 'ECG', fig_width = 15, fig_height = 2, line_w = 0.5, ecg_amp = 1.8, timetick = 0.2):
    """Plot multi lead ECG chart.
    # Arguments
        ecg        : m x n ECG signal data, which m is number of leads and n is length of signal.
        sample_rate: Sample rate of the signal.
        title      : Title which will be shown on top off chart
        fig_width  : The width of the plot
        fig_height : The height of the plot
    """
    plt.figure(figsize=(fig_width,fig_height))
    plt.suptitle(title)
    plt.subplots_adjust(
        hspace = 0, 
        wspace = 0.04,
        left   = 0.04,  # the left side of the subplots of the figure
        right  = 0.98,  # the right side of the subplots of the figure
        bottom = 0.2,   # the bottom of the subplots of the figure
        top    = 0.88
        )
    seconds = len(ecg)/sample_rate

    ax = plt.subplot(1, 1, 1)
    #plt.rcParams['lines.linewidth'] = 5
    # Changed
    step = 1.0/sample_rate
    _ax_plot(ax,np.arange(0,len(ecg)*step,step),ecg, seconds, line_w, ecg_amp, timetick)
    
DEFAULT_PATH = './'
show_counter = 1
def show_svg(tmp_path = DEFAULT_PATH):
    """Plot multi lead ECG chart.
    # Arguments
        tmp_path: path for temporary saving the result svg file
    """ 
    global show_counter
    file_name = tmp_path + "show_tmp_file_{}.svg".format(show_counter)
    plt.savefig(file_name)
    os.system("open {}".format(file_name))
    show_counter += 1
    plt.close()

def show():
    plt.show()


def save_as_png(file_name, path = DEFAULT_PATH, dpi = 100, layout='tight'):
    """Plot multi lead ECG chart.
    # Arguments
        file_name: file_name
        path     : path to save image, defaults to current folder
        dpi      : set dots per inch (dpi) for the saved image
        layout   : Set equal to "tight" to include ax labels on saved image
    """
    plt.ioff()
    plt.savefig(path + file_name + '.png', dpi = dpi, bbox_inches=layout)
    plt.close()

def save_as_svg(file_name, path = DEFAULT_PATH):
    """Plot multi lead ECG chart.
    # Arguments
        file_name: file_name
        path     : path to save image, defaults to current folder
    """
    plt.ioff()
    plt.savefig(path + file_name + '.svg')
    plt.close()

def save_as_jpg(file_name, path = DEFAULT_PATH, dpi=200):
    """Plot multi lead ECG chart.
    # Arguments
        file_name: file_name
        path     : path to save image, defaults to current folder
    """
    plt.ioff()
    plt.savefig(path + file_name + '.jpg',  dpi = dpi, bbox_inches='tight')
    plt.close()
