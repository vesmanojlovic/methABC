import pandas as pd
import numpy as np

def write_config(
	params,
	config_template_path,
	output_path,
):
    param_dict = {param: value for param, value in params.items()}
    
    with open(config_template_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if parts[0] in param_dict:
            if type(param_dict[parts[0]]) is float:
                updated_line = f"    {parts[0]} {param_dict[parts[0]]:.6f}\n"
            else:
                updated_line = f"    {parts[0]} {param_dict[parts[0]]}\n"
        else:
            updated_line = line
        updated_lines.append(updated_line)

    with open(output_path, 'w') as file:
        file.writelines(updated_lines)

    output_parts = output_path.split('/')
    config_name = output_parts.pop()
    output_dir = '/'.join(output_parts)

    return output_dir, config_name

