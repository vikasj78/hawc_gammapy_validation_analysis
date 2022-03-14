#this script is to get a hawc maps selected for an ROI and get the detector response in hd5 format.
#hawc_hal installation is needed to run this script
#pathlib is not installed by default so please install it 'pip install pathlib' with hawc_hal installation

from hawc_hal import HAL, HealpixConeROI
import matplotlib.pyplot as plt
from threeML import *
import argparse as ap
import sys,os
sys.path.insert(0,'/lfs/l2/hawc/users/vikasj78/software-hawc/threeml-analysis-scripts/fitModel/analysis_modules')
import choose_bins
from pathlib import Path
import numpy as np
from pandas import HDFStore

parser = ap.ArgumentParser(
    description="Fit a Point spectrum with both energy variables.",
    formatter_class=ap.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--estimator", default="GP_2D", required=False, help="Which energy estimator are you using: choices are nhit, gp, nn"
)

parser.add_argument(
     "--use-bins", default=None
     , nargs="*", help="Bins to  use for the analysis"
     , dest="use_bins"
)
parser.add_argument(
     "--exclude-bins", default=None
     , nargs="*", help="Exclude Bins"
     , dest="exclude"
)
parser.add_argument(
     "--select-bins-by-energy", default=None
     , nargs="*", help="Select all bins corresponding to a given energy bin from the default list"
     , dest="select_bins_by_energy"
)

parser.add_argument(
     "--select-bins-by-fhit", default=None
     , nargs="*", help="Select all bins corresponding to a given fhit bin from the default list"
     , dest="select_bins_by_fhit"
)

parser.add_argument(
    "--ROI-radius", action="store", dest="roiRadius", type=float, default=3.0, help="Fix the radius of the ROI in the model"
)

parser.add_argument(
    #"--ROI-center", action="store", dest="roiCenter", type=float, nargs = 2, default=[83.63,22.01], help="Manually specify the ROI center" #crab
    #"--ROI-center", action="store", dest="roiCenter", type=float, nargs = 2, default=[304.95,36.78], help="Manually specify the ROI center" #eHWCJ2019+368
    "--ROI-center", action="store", dest="roiCenter", type=float, nargs = 2, default=[286.91,6.32], help="Manually specify the ROI center" #eHWCJ2019+368
)

parser.add_argument(
     "--src-name", dest="src_name", default="eHWCJ1907+063", help="Give a source name."
)

parser.add_argument(
    #"-M", "--map-tree", dest="map_tree", default="data/maptree-hawc300-correctAlignment.root", help="Map-tree file." #837 days
    "-M", "--map-tree", dest="map_tree", default="data/maptree-ch103-ch603.root", help="Map-tree file." #1038 days
)

parser.add_argument(
    "-D", "--det-res", dest="det_res", default="data/detRes-GP-allDecs.root", help="Detector-response file."
)

parser.add_argument(
    "-o", "--outdir", dest="outdir", default="data/", help="Directory to save the output"
)

options = parser.parse_args()
# Define the ROI
roi_radius=options.roiRadius
roi_ra, roi_dec = options.roiCenter

roi = HealpixConeROI(data_radius=roi_radius,
                     model_radius=roi_radius,
                     ra=roi_ra,
                     dec=roi_dec)

# Get the data and detector response
map_tree=options.map_tree
det_res=options.det_res

# Instance the plugin
hawc = HAL("HAWC",
           map_tree,
           det_res,
           roi)

# Choose bins depending on the declinationOA
bins=choose_bins.analysis_bins(options, dec=roi_dec)
print(bins)
hawc.set_active_measurements(bin_list=bins)

# Display information about the data loaded and the ROI
hawc.display()

# Look at the data
fig = hawc.display_stacked_image(smoothing_kernel_sigma=0.17)

# Save outputs
outdir = Path(options.outdir) / options.src_name 
if not outdir.exists():
    outdir.mkdir()

outfile = str(outdir / "stacked_image.png")
fig.savefig(outfile)

# If you want, you can save the data *within this ROI* and the response
# in hd5 files that can be used again with HAL
output_det_res = str(outdir / "response.hd5")
output_map_tree = str(outdir / "maptree.hd5")
hawc.write(output_det_res, output_map_tree)

#get common analysis and bins in data and detector response
storeDetRes = HDFStore(output_det_res)
analysisBinsDetRes = storeDetRes['/effective_area'].index.levels[1]

storeData = HDFStore(output_map_tree)
analysisBinsData = storeData['/analysis_bins'].index.levels[0]

storeDetRes.close()
storeData.close()
