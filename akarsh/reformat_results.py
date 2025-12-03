import os
import glob
import ast # A library to safely evaluate strings containing Python literals

# --- Configuration ---
# The directory where your ArchAngel result files are located.
# The script assumes it's in the same directory it is being run from.
results_directory = "." 
# ---------------------

print(f"Searching for result files in: {os.path.abspath(results_directory)}")

# Find all the original community files from ArchAngel
file_pattern = os.path.join(results_directory, "ArchAngel_coms_t*.txt")
result_files = glob.glob(file_pattern)

# Filter out files that are already formatted to avoid processing them again
original_files = [f for f in result_files if "_formatted" not in f]

if not original_files:
    print("\nNo original ArchAngel result files found to format.")
    print("Please make sure your 'ArchAngel_coms_t*.txt' files are in this directory.")
else:
    print(f"Found {len(original_files)} files to reformat...")

# Process each file
for input_filepath in original_files:
    # Create the new filename, e.g., "ArchAngel_coms_t01_formatted.txt"
    output_filepath = input_filepath.replace(".txt", "_formatted.txt")
    
    print(f"  -> Processing '{os.path.basename(input_filepath)}'...")

    # A list to hold the newly formatted lines
    formatted_lines = []

    try:
        # Open the original result file to read
        with open(input_filepath, 'r') as infile:
            for line in infile:
                # Skip empty lines
                if not line.strip():
                    continue

                # Find the start of the list (the '[' character)
                list_start_index = line.find('[')
                if list_start_index == -1:
                    continue
                
                # Extract the string part that is the list
                list_string = line[list_start_index:]
                
                # Safely parse the string list into a real Python list of strings
                # Example: "['1', '110', '40']" becomes ['1', '110', '40']
                try:
                    nodes_as_strings = ast.literal_eval(list_string)
                except (ValueError, SyntaxError):
                    print(f"     - Warning: Could not parse a line in {os.path.basename(input_filepath)}")
                    continue
                
                # Convert the list of strings to a list of integers for sorting
                # Example: ['1', '110', '40'] becomes [1, 110, 40]
                nodes_as_integers = [int(node) for node in nodes_as_strings]
                
                # Sort the integers numerically
                # Example: [1, 110, 40] becomes [1, 40, 110]
                nodes_as_integers.sort()
                
                # Convert the sorted integers back to strings and join with a space
                # Example: [1, 40, 110] becomes "1 40 110"
                formatted_line = " ".join(map(str, nodes_as_integers))
                
                formatted_lines.append(formatted_line)

        # Write the formatted lines to the new output file
        with open(output_filepath, 'w') as outfile:
            for line in formatted_lines:
                outfile.write(line + "\n") # Write each community on a new line

    except Exception as e:
        print(f"An error occurred while processing {input_filepath}: {e}")

print("\n------------------------------------")
print("Success! Reformatting is complete.")
print(f"Your new files (ending with '_formatted.txt') are in the same directory.")
print("------------------------------------")
