import numpy as np
import argparse
import ast

def det_trace(matrix):
    m, n = matrix.shape
    
    if m != n:
        raise ValueError("Matrix must be square.")
    
    # Recusively caluclate determinant on first row
    def det_recurse(matrix):
        n = matrix.shape[1]
        
        if n == 1:
            return matrix[0,0]
        elif n == 2:
            return matrix[0,0] * matrix[1,1] - matrix[0,1] * matrix[1,0]
        
        det = 0.0
        for i in range(n):
            # The np.hstack statement eliminates the first row and ith column
            det += ((-1) ** i) * matrix[0,i] * det_recurse(np.hstack((matrix[1:, :i], matrix[1:, i+1:])))
            
        return det
    
    determinant = det_recurse(matrix)
    
    # Calculate trace by summing diagonal values
    trace = sum(matrix[i,i] for i in range(n))
    
    return {"determinant":determinant, "trace":trace}

def main(args):
    rows = []
    for row in args.matrix_row:
        rows.append(ast.literal_eval(row))
    
    matrix = np.array(rows)
    
    # Title    
    print("-"*100)
    print(f"Finding Determinant and Trace")
    print("-"*100 + "\n")
    
    # Call function and output
    output = det_trace(matrix)
    print(f"Determinant: {output['determinant']}")
    print(f"Trace: {output['trace']}")
    print(f"Ground truth: {np.linalg.det(matrix)}")
    print("\n" + "-"*100)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orthogonal Decomposition Program.",
                                    epilog="Example usage:\n python3 determinant.py [1,2] [3,4]")
    parser.add_argument("matrix_row", nargs='+', help="input vectors in format [x_1,x_2,...,x_n]. Each vector is a row of the matrix.")
    args=parser.parse_args()
    main(args)