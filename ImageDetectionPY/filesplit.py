def split_large_file(file_path, output_file1, output_file2):
    try:
        # Step 1: Count total number of lines in the large file
        with open(file_path, 'r', encoding='utf-8') as file:
            total_lines = sum(1 for line in file)

        # Step 2: Split and write to two new files
        midpoint = total_lines // 2
        with open(file_path, 'r', encoding='utf-8') as file, \
             open(output_file1, 'w', encoding='utf-8') as of1, \
             open(output_file2, 'w', encoding='utf-8') as of2:
            for i, line in enumerate(file):
                if i < midpoint:
                    of1.write(line)
                else:
                    of2.write(line)
        
        print(f"File '{file_path}' has been split into '{output_file1}' and '{output_file2}'.")

    except IOError as e:
        print(f"An error occurred while processing the file: {e}")

# Example usage
file_path = "path/to/large_file.txt"
output_file1 = "path/to/output_file1.txt"
output_file2 = "path/to/output_file2.txt"
split_large_file(file_path, output_file1, output_file2)
