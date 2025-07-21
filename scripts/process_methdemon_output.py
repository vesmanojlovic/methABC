#!/usr/bin/env python3

import pandas as pd
import os

def process_demes_data(file_path):
    """
    Processes the final_demes.csv file to extract and transform data
    from the last generation into a specific DataFrame format.

    Args:
        file_path (str): The path to the final_demes.csv file.

    Returns:
        pd.DataFrame: A transformed DataFrame where:
                      - Columns are named 'testA{deme_number}' or 'testB{deme_number}'.
                      - Entries are values from the 'AverageArray' for the corresponding deme.
                      - The DataFrame has a 0-based index, with each row
                        corresponding to an element of the AverageArray.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame() # Return an empty DataFrame on error
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return pd.DataFrame()

    # Convert 'AverageArray' string representation to actual list of floats.
    # Now assuming the format is "float;float;float"
    def safe_parse_average_array(x):
        try:
            # Check if x is a string (e.g., "1.0;2.0;3.0")
            if isinstance(x, str):
                # Split by semicolon, strip whitespace, and convert to float
                parts = [p.strip() for p in x.split(';') if p.strip()]
                return [float(p) for p in parts]
            return x # If not a string, assume it's already in the correct format
        except (ValueError, AttributeError):
            # Catch errors during parsing (e.g., non-numeric parts, malformed string)
            print(f"Warning: Could not parse AverageArray value: '{x}'. Skipping row.")
            return [] # Return an empty list for malformed entries

    df['AverageArray'] = df['AverageArray'].apply(safe_parse_average_array)

    # 1. Take only rows from the last available generation
    if df.empty:
        print("DataFrame is empty after reading CSV.")
        return pd.DataFrame()

    last_generation = df['Generation'].max()
    df_last_gen = df[df['Generation'] == last_generation].copy()

    if df_last_gen.empty:
        print(f"No data found for the last generation ({last_generation}).")
        return pd.DataFrame()

    # Create the new deme names (e.g., 'testA1', 'testB2')
    df_last_gen['DemeName'] = df_last_gen.apply(
        lambda row: f"testA{int(row['Deme'])}" if row['Side'] == 'left' else f"testB{int(row['Deme'])}",
        axis=1
    )

    # Prepare data for the new DataFrame
    # We'll create a dictionary where keys are DemeNames and values are their AverageArrays
    deme_data = {row['DemeName']: row['AverageArray'] for _, row in df_last_gen.iterrows()}

    # Create the final DataFrame
    # pd.DataFrame.from_dict will create columns from keys, and each list
    # will become a column. If lists have different lengths, it will pad with NaN.
    transformed_df = pd.DataFrame(deme_data)

    # Sort columns as per the import_data function's logic (testA then testB, numerically)
    columns_a = sorted(
        [col for col in transformed_df.columns if "testA" in col],
        key=lambda x: int(x.replace('testA', ''))
    )
    columns_b = sorted(
        [col for col in transformed_df.columns if "testB" in col],
        key=lambda x: int(x.replace('testB', ''))
    )
    sorted_columns = columns_a + columns_b

    # Filter for only the columns that actually exist in the transformed_df
    # This prevents errors if some expected columns are missing
    final_sorted_columns = [col for col in sorted_columns if col in transformed_df.columns]

    # Reorder columns based on the sorted list
    transformed_df = transformed_df[final_sorted_columns]

    return transformed_df

if __name__ == "__main__":
    # Determine the path to the CSV file relative to the script's location
    # Assuming the script is in 'scripts/' and 'test/' is a sibling directory
    input_file_path = 'test/final_demes.csv'
    output_file_path = 'test/processed_final_demes_output.csv' # New output path

    print(f"Attempting to read data from: {input_file_path}")
    processed_dataframe = process_demes_data(input_file_path)

    if not processed_dataframe.empty:
        print("\nProcessed DataFrame:")
        print(processed_dataframe)

        # Save the processed DataFrame to the specified output file in 'test/'
        try:
            processed_dataframe.to_csv(output_file_path, index=False)
            print(f"\nProcessed data saved to: {output_file_path}")
        except Exception as e:
            print(f"Error saving processed data to CSV: {e}")
    else:
        print("No DataFrame was generated due to an error or empty data.")


