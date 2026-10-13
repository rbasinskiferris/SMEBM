Folder with code needed to reproduce the main results in the preprint titled "Physical Processes Driving Year-to-Year Surface Temperature Variability Changes in a Warming Climate"

The main figures are made in the following:

Figure 1: Plot made in variability_LE_forpaper.ipynb. Underlying data needed to plot is in the ENSO filtering file (de_enso_withsst_fixmlp.ipynb). Pre-processing the CESM-LE data is done in scripts within the subfolder regridding_cesm_data.

Figure 2: Plot made in plot_feedbackparameters_and_heatcap.ipynb. Variability feedback parameter was found in variability_regression_ensoremoved.ipynb. The data needed for the regression for the variability feedback is made in the ENSO filtering files de_enso_withsst_fixmlp.ipynb and de_enso_toa_withsst.ipynb)

Figure 3: Plot made in make_plots.ipynb with output from run_main_smebm_experiment.py which uses functions in utils_withmoist.py

Most supplementary figures (S1-S10) for v1 of the paper are also plotted in the above files:
variability_LE_forpaper.ipynb: S1, S2
plot_feedbackparameters_and_heatcap.ipynb: S3, S5, S7, S9 
make_plots.ipynb: S4, S6

But, note that the underlying .py for creating output to plot S4, S6, S9 still need to be added to the repo. Should also add code for making S8.