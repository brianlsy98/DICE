import json
import copy
import argparse
import os
import itertools


def gen_cir_files(mosfet_spec, sim_cir_params, template_dir, output_dir):
    """
    Reads the template .cir file, replaces placeholders, and writes a new .cir file.
    """
    
    # Ensure the output directories exist, create them if not
    os.makedirs(output_dir, exist_ok=True)

    # Circuit Template Load
    with open(f"{template_dir}/netlist.cir", 'r') as f:
        content = f.read()

    content = content.replace("(MOS_SPEC)", str(mosfet_spec))



    ###########################################
    # Replace placeholders with actual values #
    ###########################################
    ################# Netlist #################
    cir_params = sim_cir_params["circuit_param"]
    param_names = list(cir_params.keys())
    param_values = list(cir_params.values())

    combinations = list(itertools.product(*param_values))

    for i, combination in enumerate(combinations):
        content_i = copy.deepcopy(content)

        # Create a string that represents the parameters and their values in the format "W=... L=..."
        for param, value in zip(param_names, combination):
            content_i = content_i.replace(f"({param})", str(value))
        
        netlist_name = f"netlist"
        for param, value in zip(param_names, combination):
            netlist_name += f"_{param}-{value}"
        
    ################# Simulation #################
        sim_params = sim_cir_params["simulation_param"]

        for key, param_set in sim_params.items():
            if type(param_set) == dict:
                for k, v in param_set.items():
                    content_i = content_i.replace(f"({key}_{k})", str(v))
            else:
                content_i = content_i.replace(f"({key})", str(param_set))

        # Write the netlist to a new file
        with open(f"{output_dir}/{netlist_name}.cir", 'w') as f:
            f.write(content_i)



if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Generate circuit and simulation files from templates.")
    parser.add_argument("--mosfet_spec", default="45nm_bulk.txt",
                        type=str, help="Path to the MOSFET model file")
    parser.add_argument("--sim_netlist_name", default="sourcefollower_dc",
                        type=str, help="Name of the netlist template directory")
    args = parser.parse_args()


    mosfet_spec = f"./templates/mosfet_models/{args.mosfet_spec}"
    template_dir = f"./templates/sim_netlist_templates/{args.sim_netlist_name}"
    output_dir = f"./data/netlists/{args.sim_netlist_name}"


    # Netlist Parameters
    with open(f"{template_dir}/params.json", 'r') as file:
        sim_cir_params = json.load(file)

    # Modify the netlist file (replacing placeholders with actual values)
    gen_cir_files(mosfet_spec, sim_cir_params, template_dir, output_dir)