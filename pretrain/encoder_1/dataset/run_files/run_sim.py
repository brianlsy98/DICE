from multiprocessing import process
import subprocess
import re
import os
import sys
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(parent_dir)

from utils import parse_sim_netlist


def run_ngspice(sim_netlist_name, netlist_dir):
    """
    Runs ngspice in batch mode to simulate the circuit.
    """
    try:
        for i, netlist_f in enumerate(os.listdir(netlist_dir)):
            netlist_file = os.path.join(netlist_dir, netlist_f)
            

            with open(f"{netlist_file}", 'r') as f:
                content = f.read()


            txt_dir = f"./data/sims/txt/{sim_netlist_name}"
            os.makedirs(txt_dir, exist_ok=True)


            # (OUT_PATH)
            txt_file_name = f""
            params_split = re.split(r"[_]", str(netlist_f))[1:]
            params_split = [param for param in params_split if param]
            for i, param in enumerate(params_split):
                if i == len(params_split) - 1:
                    txt_file_name += f"{param[:-4]}"
                else: txt_file_name += f"{param}_"
            txt_file_path = f"{txt_dir}/{txt_file_name}.txt"
            content = content.replace("(OUT_PATH)", f"{txt_file_path}")


            # (OUT_DATA)
            nodes, _, _, _, _ = parse_sim_netlist(netlist_file)
            simulation_data = [f"v({n})" for n in nodes]
            simulation_data_str = " ".join(simulation_data)
            content = content.replace("(OUT_DATA)", f"{simulation_data_str}")


            # Write the modified content to a temporary .cir file
            temp_cir_file = "./temp.cir"
            with open(temp_cir_file, 'w') as temp_file:
                temp_file.write(content)

            # Run ngspice using the temporary .cir file
            result = subprocess.run(['ngspice', '-b', temp_cir_file], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE, 
                                    text=True)
            print(f"Simulation Done: {netlist_f}")
            

            # Write node names sequentially
            node_names_combined = " ".join(simulation_data)
            
            # Read the content of the file first
            with open(txt_file_path, 'r') as txt_file:
                lines = txt_file.readlines()

            processed_lines = []

            # Process each line
            for line in lines:
                elements = line.strip().split()
                unrepeated_elements = []  # Start with the first element

                # Enumerate over the rest of the elements
                for i, element in enumerate(elements):
                    if i % 2 == 1:
                        unrepeated_elements.append(element)

                # After processing the line, append the result to processed_lines
                processed_lines.append(" ".join(unrepeated_elements) + '\n')

            # Write the processed lines back to the same file
            with open(txt_file_path, 'w') as txt_file:
                txt_file.writelines(processed_lines)

            # Now, write the node names at the beginning of the file
            with open(txt_file_path, 'r+') as txt_file:
                lines = txt_file.readlines()
                txt_file.seek(0)
                txt_file.write(node_names_combined + "\n")
                txt_file.writelines(lines)


            
    except FileNotFoundError:
        print("Ngspice is not installed or not found in the system PATH.")
    except Exception as e:
        print(f"An error occurred: {e}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate circuit and simulation files from templates.")
    parser.add_argument("--sim_netlist_name", default="sourcefollower_dc",
                        type=str, help="Name of the netlist template directory")
    args = parser.parse_args()

    netlist_dir = f"./data/netlists/{args.sim_netlist_name}"

    run_ngspice(args.sim_netlist_name, netlist_dir)