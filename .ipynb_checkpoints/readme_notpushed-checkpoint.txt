(Note to self: This folder contains minor changes and reorganization from the folder cleaned_model_sebm)


Files that we played around with but don't need to go on github are in the comparison_with_old and working_on subfolders. comparison_with_old: this includes double checking that the cleaned up code did the same processing for the enso filtering etc. Additionally in there we did other checks. For example, won't push these to github because not used but there's a new check in these that the variability feedback has the same trend if we had used the mlp filtered toa flux and sst rather than the linear regression filtered.



The .py file that makes the main results is run_main_smebm_experiment.py. This is basically exactly the same as run_decomp_notikhonov_8pt74_updatedlam_diff65_updateland_paperparams_quarterdt.py in cleaned_model_sebm/SEBM/tune_timestep/params_kimnorth1992_sstlambdafix/. We read in the different variability feedback for consistency, but we have verified that those are exactly the same.

NOTE OCT 11 THE FIRST PART (other than transport decomp) was rerun to make sure all our results are the same. these all have _testoct right before .nc. We checked these on OCT 11 and they looked the same. This test is make_plots_test_diff.ipynb and teh version of run_main_smebm_experiment.py for the comparison is in the comparison_with_hold fodler along with these plots.


!!!!!!!!!Things we may want to update in future but we aren't worrying about for initial submission (PAY ATTENTION HERE AND TAKE ANOTHER LOOK):!!!!!!!!!
In plot_feedbackparameters_and_heatcap in comparison_with_old we see with our Greenland selection, we get a tiny amount (like a few grid cells) of the Canadian Arctic, which we could fix if we wanted to, but seems pretty minor.

Removed run with noies one process in space. Claude wrote me a function for noise red in space that's in the utils_withmoist.py in workingon called noise_mat_red_space. an initial check with that also in the working on folder called run_decomp_notikhonov_8pt74_updatedlam_diff65_updateland_paperparams_quarterdt_redspace shows that you get very minor changes when you plot it (see make_plots in workingon). We didn't show this as a figure in the SM for now but just comment that simple tests adding spatial correlation don't significantly change the results. Could add this as a figure, but want to double check the Claude function first.

Not sure we did exactly the right thing for regressing for C. Couldn't use integration of dT/dt as December 31 - Jan 1 because so much noise in that we got nonsensincal result (maybe regression dilution?) so did the finite difference that you see. But that includes stuff from the boundary years. Could maybe do mean of december minus mean of january instead.

