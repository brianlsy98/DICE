#!/bin/bash

# User Inputs
snt_dir="./templates/sim_netlist_templates"
mosfet_spec_file="45nm_bulk.txt"
all_netlists_dir="./data/netlists"
sim_results_dir="./data/sims/txt"
train_ratio=0.8
val_ratio=0.1
test_ratio=0.1


echo ""
# Check if the directories exist
if [ ! -d "${snt_dir}" ]; then
  echo "Directory ${snt_dir} does not exist"
  exit 1
fi

# Get all folder names inside ./templates/sim_netlist_templates/
snt_folders=$(ls -d ${snt_dir}/*/)


##############################################################
# Cleanup
netlist_dir="${all_netlists_dir}"
if [ -d "$netlist_dir" ]; then
  echo "Removing all files in ${netlist_dir}"
  rm -rf ${netlist_dir}/*
else
  echo "Directory ${netlist_dir} does not exist, creating it."
  mkdir -p ${netlist_dir}
fi
sim_dir="${sim_results_dir}"
if [ -d "$sim_dir" ]; then
  echo "Removing all files in ${sim_dir}"
  rm -rf ${sim_dir}/*
else
  echo "Directory ${sim_dir} does not exist, creating it."
  mkdir -p ${sim_dir}
fi
echo "Cleanup completed."
##############################################################



# Iterate over all combinations of netlist_template_folders and sim_template_folders
for snt_folder in $snt_folders; do

    echo ""
    # Extract just the folder names from the paths (strip the directory structure)
    snt_folder_name=$(basename ${snt_folder%/})

    echo "##############################################"
    echo "## Netlist & Simulation: ${snt_folder_name}"
    echo "##############################################"



    ##############################################################
    # Generate Netlists
    python ./run_files/gen_cir_files.py --mosfet_spec=${mosfet_spec_file} --sim_netlist_name=${snt_folder_name}

    if [ $? -eq 0 ]; then
      echo "gen_cir_files.py executed successfully for ${snt_folder_name}."
    else
      echo "Error occurred while running gen_cir_files.py for ${snt_folder_name}."
    fi
    ##############################################################


    ##############################################################
    # Run Ngspice simulations
    python ./run_files/run_sim.py --sim_netlist_name=${snt_folder_name}
    
    if [ $? -eq 0 ]; then
      echo "run_sim.py executed successfully for ${snt_folder_name}."
    else
      echo "Error occurred while running run_sim.py for ${snt_folder_name}."
    fi
    ##############################################################

done


echo ""
##############################################################
# Generate pytorch dataset
python ./run_files/save_dataset.py --netlist_dir=${all_netlists_dir} --sim_result_dir=${sim_results_dir} --train_ratio=${train_ratio} --val_ratio=${val_ratio} --test_ratio=${test_ratio}
if [ $? -eq 0 ]; then
  echo "save_dataset.py executed successfully"
else
  echo "Error occurred while running save_dataset.py"
fi
##############################################################

rm bsim4v5.out temp.cir
echo ""