"""
Detail the statistics of profiling_results.txt file.
"""

def parse_profiling_results(file_path):
    """Parse cProfile text output format with proper column parsing."""
    profiling_data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Skip header lines until we find the column header
    data_started = False
    for line in lines:
        # Find the header line with ncalls, tottime, etc.
        if 'ncalls' in line and 'tottime' in line and 'cumtime' in line:
            data_started = True
            continue
        
        # Skip dashes line after header
        if data_started and line.strip().startswith('---'):
            continue
        
        if data_started and line.strip():
            # Skip lines with /* markers (omitted lines)
            if line.strip().startswith('/*'):
                continue
            
            # Parse function entries
            # Format: ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            parts = line.split()
            
            if len(parts) >= 5:
                try:
                    # Handle "x/y" format for ncalls
                    ncalls_str = parts[0]
                    if '/' in ncalls_str:
                        ncalls = int(ncalls_str.split('/')[0])
                    else:
                        ncalls = int(ncalls_str)
                    
                    tottime = float(parts[1])
                    # parts[2] is percall for tottime (skip)
                    cumtime = float(parts[3])
                    # parts[4] is percall for cumtime (skip)
                    
                    # Extract function name (everything from parts[5] onward)
                    func_name = ' '.join(parts[5:]) if len(parts) > 5 else parts[-1]
                    
                    # Clean up the function name by removing the long project path
                    func_name = clean_function_name(func_name)
                    
                    profiling_data[func_name] = {
                        'ncalls': ncalls,
                        'tottime': tottime,
                        'cumtime': cumtime
                    }
                except (ValueError, IndexError):
                    continue
    
    return profiling_data

def clean_function_name(func_name):
    """Clean up function names by removing long path prefixes."""
    # Remove the .venv library path
    if '.venv/lib/python3.12/site-packages/' in func_name:
        func_name = func_name.split('.venv/lib/python3.12/site-packages/')[1]
    # Remove the project path prefix
    elif '/optimized_virtual_orbitals/./' in func_name:
        func_name = func_name.split('/optimized_virtual_orbitals/./')[1]
    elif '/optimized_virtual_orbitals/' in func_name:
        func_name = func_name.split('/optimized_virtual_orbitals/')[1]
    
    return func_name

def summarize_profiling_data(profiling_data):
    """Summarize profiling data with % metrics."""
    if not profiling_data:
        return None
    
    total_functions = len(profiling_data)
    total_calls = sum(data['ncalls'] for data in profiling_data.values())
    total_time = sum(data['tottime'] for data in profiling_data.values())
    total_cumtime = sum(data['cumtime'] for data in profiling_data.values())    

    summary = {
        'total_functions': total_functions,
        'total_calls': total_calls,
        'total_time': total_time,
        'total_cumtime': total_cumtime
    }
    return summary

def print_top_functions(profiling_data, top_n=15, output_file=None):
    """Print top functions by cumulative time with % breakdown."""
    if not profiling_data:
        msg = "No profiling data to display."
        print(msg)
        if output_file:
            output_file.write(msg + "\n")
        return
    
    sorted_data = sorted(profiling_data.items(), 
                        key=lambda x: x[1]['cumtime'], 
                        reverse=True)
    
    total_cumtime = sum(data['cumtime'] for _, data in sorted_data)
    
    header = f"\n{'Function':<60} {'Cumtime':<12} {'%':<8} {'Calls':<8} {'Func/Cumtime':<12}"
    separator = "-" * 105
    
    print(header)
    print(separator)
    if output_file:
        output_file.write(header + "\n")
        output_file.write(separator + "\n")
    
    for i, (func_name, data) in enumerate(sorted_data[:top_n], 1):
        percentage = (data['cumtime'] / total_cumtime * 100) if total_cumtime > 0 else 0
        # Truncate long function names
        display_name = func_name if len(func_name) <= 60 else func_name[:57] + "..."
        line = f"{display_name:<60} {data['cumtime']:<12.6f} {percentage:<8.2f} {data['ncalls']:<8} {data['ncalls']/data['cumtime'] if data['ncalls'] > 0 else 0:<12.6f}"
        print(line)
        if output_file:
            output_file.write(line + "\n")

def main():
    opt = 'opt_C'  # Change this to select different profiling result files
    input_file_path = 'branch/profil/profiling_results_'+opt+'.txt'
    output_file_path = 'branch/profil/profiling_summ_'+opt+'.txt'
    
    profiling_data = parse_profiling_results(input_file_path)
    
    if not profiling_data:
        print(f"Error: Could not parse profiling data from {input_file_path}")
        print("Make sure the file contains valid cProfile text output.")
        return
    
    summary = summarize_profiling_data(profiling_data)
    
    # Write to file
    with open(output_file_path, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("PROFILING SUMMARY\n")
        f.write("=" * 90 + "\n")
        f.write(f"Total Functions: {summary['total_functions']}\n")
        f.write(f"Total Calls: {summary['total_calls']}\n")
        f.write(f"Total Time (tottime): {summary['total_time']:.6f} seconds\n")
        f.write(f"Total Time (cumtime): {summary['total_cumtime']:.6f} seconds\n")
        
        print("\n" + "=" * 90)
        print("PROFILING SUMMARY")
        print("=" * 90)
        print(f"Total Functions: {summary['total_functions']}")
        print(f"Total Calls: {summary['total_calls']}")
        print(f"Total Time (tottime): {summary['total_time']:.6f} seconds")
        print(f"Total Time (cumtime): {summary['total_cumtime']:.6f} seconds")
        
        print_top_functions(profiling_data, top_n=15, output_file=f)
    
    print(f"\nResults saved to {output_file_path}")

if __name__ == "__main__":
    main()