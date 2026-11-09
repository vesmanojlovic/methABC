import os
import pandas as pd
import numpy as np

repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def write_config(
	params,
	config_template_path=os.path.join(repo_dir, "resources/config_template.dat"),
	output_dir="simulations/",
):
    """
    Write parameters drawn from prior into a config file based on template

    Args:
        params: sampled prior distribution (accessed as a dict)
        config_template_path: path to the template config
        output_dir: directory to write the config file to
    """
    filename = "config.dat"
    with open(config_template_path, "r") as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        parts = line.split(maxsplit=1)
        if parts and parts[0] in params.keys():
            if isinstance(params[parts[0]], float):
                updated_line = f"    {parts[0]} {params[parts[0]]:.10g}\n"
            else:
                updated_line = f"    {parts[0]} {params[parts[0]]}\n"
        else:
            updated_line = line
        updated_lines.append(updated_line)

    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w") as file:
        file.writelines(updated_lines)

    return output_path


def import_data(data_path):
    """
    Import tumour methylation data from file.
    Args:
        data_path: path to the data file
    Returns:
        data: pandas dataframe with columns rearranged
    """
    data = pd.read_csv(data_path, index_col=0).dropna()
    columns_a = sorted([col for col in data.columns if "A" in col], key=len)
    columns_b = sorted([col for col in data.columns if "B" in col], key=len)
    other_columns = [col for col in data.columns if "A" not in col and "B" not in col]
    sorted_columns = columns_a + columns_b + other_columns
    data = data[sorted_columns]
    return data


def side_counts(data):
    """
    Count the glands on each side of a tumour.
    Args:
        data: dataframe from import_data
    Returns:
        number of A glands, number of B glands
    """
    sides = [col[1] for col in data.columns]
    return sides.count("A"), sides.count("B")
