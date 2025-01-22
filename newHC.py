import numpy as np

# Function to calculate the numerator for flow correction
def numerator(n, r, Q):
    return r * Q * np.abs(Q)**(n - 1)   #** operator is used for power

# Function to calculate the denominator for flow correction
def denominator(n, r, Q):
    return 2 * r * np.abs(Q)    #np.abs() is used for obtaining absolute value of Q

# Function to compute flow correction ΔQ
def flow_correction(numerator_sum, denominator_sum):
    return -numerator_sum / denominator_sum if denominator_sum != 0 else 0.0

# Function to perform one iteration of flow correction for a loop
def loop_iteration(n, pipes_r, pipes_Q, loop_signs):
    numerator_sum = 0.0
    denominator_sum = 0.0

    for j in range(len(pipes_Q)):
        numerator_sum += loop_signs[j] * numerator(n, pipes_r[j], pipes_Q[j])
        denominator_sum += denominator(n, pipes_r[j], pipes_Q[j])

    delta_Q = flow_correction(numerator_sum, denominator_sum)

    for j in range(len(pipes_Q)):
        pipes_Q[j] += loop_signs[j] * delta_Q

    return pipes_Q, delta_Q

# Hardy Cross iterative function
def hardy_cross(pipe_diameters, resistance_values, initial_flows, loops, epsilon):
    Q = initial_flows.copy()
    iteration = 0   #counter

    while True:
        max_correction = 0.0    #initializing max correction by zero
        print(f"Iteration {iteration}")

        for loop_idx, loop in enumerate(loops):
            loop_indices = np.where(loop != 0)[0]
            loop_signs = loop[loop_indices]
            pipes_r = resistance_values[loop_indices]
            pipes_Q = Q[loop_indices]

            # Perform flow correction
            pipes_Q, delta_Q = loop_iteration(2, pipes_r, pipes_Q, loop_signs)
            Q[loop_indices] = pipes_Q  # Reflect changes in global array

            max_correction = max(max_correction, np.abs(delta_Q))
            print(f"Loop {loop_idx + 1}, ΔQ = {delta_Q:.6f}")

        print(f"Updated flows: {Q}")
        print(f"Maximum correction: {max_correction}")

        if max_correction < epsilon:
            break
        iteration += 1  #updating iteration counter

    return Q

# Main program - here main thing to note besides the algorithm is the data type used by me which is float because after
# some iterations rounding off of the values may create an unnecessary variations in the output values.
def main():
    pipe_diameters = np.array([0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.2, 0.2], dtype=float)
    resistance_values = np.array([96.8, 96.8, 306, 306, 306, 3098, 3098, 3098, 306, 612, 306, 9295, 3098], dtype=float)
    initial_flows = np.array([247, -286, -100, -50, 136, 40, -40, 46, 10, 14, 100, 50, 0], dtype=float)

    # This matrix use is for predicting the direction of flow in the branches of the given pipe network
    loops = np.array([
        [1, 1, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, -1, -1, -1, 0, 0],
    ], dtype=int)

    epsilon = 0.001

    final_flows = hardy_cross(pipe_diameters, resistance_values, initial_flows, loops, epsilon)

    print("\nFinal Flow Distribution (m^3/s):")
    for i, flow in enumerate(final_flows, start=1):
        print(f"Branch {i}: {flow:.6f} m^3/s")

if __name__ == "__main__":
    main()
