#!/usr/bin/env python3
# validate_sample_bdt.py

import sys, os

__doc__ = """Validate the sample BDT using pybdt.validate."""

try:
    import matplotlib
except ImportError:
    print("Matplotlib is missing, so validate_sample_bdt.py cannot run.")
    sys.exit()

matplotlib.use ('Agg')

from pybdt.histlite import Plotter
from pybdt.util import load, mkdir
from pybdt.validate import Validator

I3_BUILD = os.getenv ('I3_BUILD')
data_dir = '{0}/pybdt/resources/examples/data'.format(I3_BUILD)
output_dir = '{0}/pybdt/resources/examples/output'.format(I3_BUILD)

v = load ('{0}/sample.validator'.format(output_dir))

from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt



# find out when the data rate is close to the sig rate
h_data = v.get_Hist ('test_data', 'scores', range=(-1,1), bins=10000)
h_sig = v.get_Hist ('test_sig', 'scores', range=(-1,1), bins=10000)
rate_data = h_data.cumulative_left
rate_sig = h_sig.cumulative_left
rate_ratio = rate_data / rate_sig
bdt_cut = rate_ratio.bins[rate_ratio.values < 1.1][0]
print ('Selecting BDT cut {0}...'.format (bdt_cut))


# these are the things we will plot on a bunch of figures
lines = ['test_data', 'test_sig', 'bg', 'total_mc']


def add_cut_lines (*axeses):
    # add a line indicating our chosen "signal-level" cut
    for axes in axeses:
        axes.axvline (bdt_cut, color='g', ls='--')


def do_summary_plots ():
    print ('BDT score distribution...')
    # make a dual linear/log figure showing the BDT score distribution
    objs = v.create_plot ('scores', 'dist',
            lines,
            bins=100,
            range=(-1,.5),
            dual=True,
            data_mc=True,
            xlabel='BDT score',
            left_ylabel='Hz per bin',
            title='BDT score distributions',
            linear_kwargs=dict (
                legend=dict (loc='best'),
                ),
            )
    print (sorted (objs))
    objs['second_main_ax'].set_ylim (ymin=1e-6)
    add_cut_lines (objs['first_main_ax'], objs['second_main_ax'])
    objs['fig'].savefig ('{0}/dist_vs_bdt.png'.format(output_dir))

    print ('Rate vs BDT cut...')
    # make a dual linear/log figure showing the rate vs cut
    objs = v.create_plot ('scores', 'rate',
            lines,
            bins=100,
            range=(-.2,.5),
            data_mc=True,
            dual=True,
            xlabel='BDT score',
            left_ylabel='Hz per bin',
            log_kwargs = dict (
                legend=dict (loc='lower left'),
                ),
            title='Rate vs BDT cut',
            )
    objs['first_main_ax'].set_ylim (0, 1e-2)
    add_cut_lines (objs['first_main_ax'], objs['second_main_ax'])
    objs['fig'].savefig ('{0}/rate_vs_bdt.png'.format(output_dir))

    print ('Efficiency vs BDT cut...')
    # make a dual linear/log figure showing the efficiency vs cut
    objs = v.create_plot ('scores', 'eff',
            lines,
            bins=100,
            range=(-1,.5),
            dual=True,
            dual_legend='left',
            data_mc=True,
            xlabel='BDT score',
            left_ylabel='efficiency',
            linear_kwargs=dict (
                legend=dict (loc='lower left'),
                ),
            title='Efficiency vs BDT cut',
            )
    add_cut_lines (objs['first_main_ax'], objs['second_main_ax'])
    objs['fig'].savefig ('{0}/eff_vs_bdt.png'.format(output_dir))

def do_overtrain_plot ():
    propsmall = FontProperties (size='small')

    print ('Overtrain check...')
    objs = v.create_overtrain_check_plot (
            'train_sig', 'test_sig',
            'train_data', 'test_data',
            left_ylabel='relative abundance (background)',
            right_ylabel='relative abundance (signal)',
            legend=dict (loc='upper left'),
            legend_side='left',
            )
    objs['fig'].savefig ('{0}/overtrain_check.png'.format(output_dir))

def do_variable_plots ():

    def do_variable_plot (name, cut_level, outdir):
        filename = '{0}/{1}.png'.format (outdir, name)
        # note that both the thing to plot (name, in this case) and the cut can
        # be specified in a very flexible way; see Validator.eval()
        # documentation for more info.
        objs = v.create_plot (name, 'dist',
            lines,
            bins=50,
            dual=True,
            data_mc=True,
            left_ylabel='Hz per bin',
            xlabel=name,
            log_kwargs=dict (
                legend=dict (loc='best')
                ),
            title='{0} | bdt score > {1:.3f}'.format (name, cut_level),
            cut='scores > {0}'.format (cut_level)
            )
        objs['fig'].savefig (filename)

    input_vars_dir = '{0}/vars_at_input'.format(output_dir)
    print ('Plotting input vars in {0} ...'.format (input_vars_dir))
    mkdir (input_vars_dir)
    do_variable_plot ('a', -1, input_vars_dir)
    do_variable_plot ('b', -1, input_vars_dir)
    do_variable_plot ('c', -1, input_vars_dir)

    postcut_vars_dir = '{0}/vars_after_cut'.format(output_dir)
    print ('Plotting vars after score > {0} cut in {1} ...'.format (
        bdt_cut, postcut_vars_dir))
    mkdir (postcut_vars_dir)
    do_variable_plot ('a', bdt_cut, postcut_vars_dir)
    do_variable_plot ('b', bdt_cut, postcut_vars_dir)
    do_variable_plot ('c', bdt_cut, postcut_vars_dir)

    fig = v.create_correlation_matrix_plot ('train_sig')
    fig.savefig ('{0}/correlation_matrix-signal.png'.format(output_dir))

    fig = v.create_correlation_matrix_plot ('train_data')
    fig.savefig ('{0}/correlation_matrix-background.png'.format(output_dir))


def do_eff_plots ():

    eff_plot_dir = '{0}/eff_vs_vars'.format(output_dir)

    mkdir (eff_plot_dir)

    print ('Plotting efficiency vs training variables...')

    def do_eff_plot (name, cut_level):

        # Validator doesn't know how to make efficiency plots, but it can
        # provide the histogram objects with which to do so.

        bins = 25
        vals = v.data['test_sig'].eval (name)
        range = (vals.min (), vals.max ())
        h_input = v.get_Hist ('test_sig', name,
                bins=bins, range=range,
                )
        h_cut = v.get_Hist ('test_sig', name,
                bins=bins, range=range,
                cut='scores > {0}'.format (cut_level)
                )
        eff = h_cut / h_input
        fig = plt.figure ()
        ax = fig.add_subplot (111)
        plotter = Plotter (ax)
        plotter.add (eff)
        plotter.finish ()
        ax.set_xlabel (name)
        ax.set_xlim (range)
        ax.set_ylabel ('efficiency')
        ax.set_ylim (0,1.05)
        ax.set_title ('efficiency vs {0}'.format (name))
        fig.savefig ('{0}/eff_vs_{1}.png'.format (eff_plot_dir, name))
        plt.close (fig)

    do_eff_plot ('a', bdt_cut)
    do_eff_plot ('b', bdt_cut)
    do_eff_plot ('c', bdt_cut)

do_summary_plots ()
do_overtrain_plot ()
do_variable_plots ()
do_eff_plots ()
