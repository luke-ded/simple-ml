import numpy as np
import argparse
import ast

def gram_schmidt(basis, orthonormal=False):
    orthogonal_basis = [basis[0]]
    
    # Calculate each vector
    for vector in basis[1:]:
        new_vector = vector.copy()
        
        # Subtract projections
        for orthogonal_vector in orthogonal_basis:
            denom = (orthogonal_vector @ orthogonal_vector)
            if denom > 1e-9:
                new_vector -= ((vector @ orthogonal_vector) / denom) * orthogonal_vector
        
        orthogonal_basis.append(new_vector)
    
    # Remove zero vectors
    orthogonal_basis = [u for u in orthogonal_basis if not (np.abs(u) < 1e-9).all()]
    
    # If orthonormal basis specified, divide each vector by its l_2 norm
    if orthonormal:
        orthogonal_basis = [u / ((u @ u) ** 0.5) for u in orthogonal_basis]
            
    return orthogonal_basis 

def orthogonal_decomposition(y_vector, vector_set):
    # y = y_hat + z, where z is the orthogonal complement
    orthogonal_basis = gram_schmidt(vector_set)
    
    y_hat = np.zeros(len(orthogonal_basis[0]))
    
    # Add projections
    for orthogonal_vector in orthogonal_basis:
        y_hat += ((y_vector @ orthogonal_vector) / (orthogonal_vector @ orthogonal_vector)) * orthogonal_vector
        
    z = y_vector - y_hat
        
    return {"projection": y_vector - z, "complement": z}

# Least squares using QR decomposition
def least_squares(b, vector_set):
    n = len(vector_set)
    
    decomposition = orthogonal_decomposition(b, vector_set)
    b_hat = decomposition["projection"]
    
    orthonormal_basis = gram_schmidt(vector_set, orthonormal=True)
    Q = np.column_stack(orthonormal_basis)
    
    if len(orthonormal_basis) < n:
        return "Infinite solutions exist."
    
    # R is an upper triangular matrix a_1*e_1 ... a_n*e_1
    #                                    0    ... a_n*e_2
    #                                               ...
    #                                             a_n*e_n
    R = np.zeros((n,n))
    for i in range(n):
        for j in range(i, n):
            R[i][j] = orthonormal_basis[i] @ vector_set[j]
    
    # Solve Rx_hat = Q^Tb_hat for x_hat
    Q_Tb = Q.T @ b_hat
    x_hat = np.zeros(n)
    for i in range(n-1, -1, -1):
        x_hat[i] = (Q_Tb[i]- R[i, i+1:] @ x_hat[i+1:]) / R[i, i]
    
    return x_hat

def main(args):
    vector_set = []
    for vector in args.vector:
        vector_set.append(np.array(ast.literal_eval(vector), dtype=float))
    
    # Title    
    print("-"*100)
    print(f"Running {'Gram-Schmidt Orthogonalization' if args.mode == 'gram-schmidt' else 'Orthogonal Decomposition' if args.mode == 'decomposition' else 'Least Squares'}")
    print("-"*100 + "\n")
    
    # Call functions and output
    if args.mode == "gram-schmidt":
        print(f"{'Orthonormal' if args.orthonormal else 'Orthogonal'} Basis:\n")
        output = gram_schmidt(vector_set, args.orthonormal)
        for vector in output: print(vector)
    elif args.mode == "decomposition":
        output = orthogonal_decomposition(vector_set[0], vector_set[1:])
        print(f"Projection: {output['projection']}")
        print(f"Complement: {output['complement']}")
    else:
        output = least_squares(vector_set[-1], vector_set[:-1])
        print(f"Least-squares solution: {output}")
        
    print("\n" + "-"*100)


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{:g}'.format})
    parser = argparse.ArgumentParser(description="Gram-Schmidt Orthagonalization Program.",
                                     epilog="Example usage:\n python3 gram-schmidt.py [0,-1,2,0,2] [1,-3,1,-1,2] [-3,4,1,2,1] [-1,-3,5,0,7]")
    parser.add_argument("mode",help="`gram-schmidt` OR `decomposition OR least-squares`")
    parser.add_argument("vector", nargs='+', help="vectors in format [x_1,x_2,...,x_n]. In gram-schmidt mode: all vectors compose the set. If in decomposition mode: first vector is the vector to be decomposed; remaining vectors compose the set. If in least-squares mode: vectors are columns of A followed by b as the last vector.")
    parser.add_argument("-on", "--orthonormal", action="store_true", help="create an orthonormal basis")
    args=parser.parse_args()
    
    main(args)