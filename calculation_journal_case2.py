import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
from scipy.optimize import root

S = 15
cdp = 80
csysu = 60
delta1min = 10
delta2min = 2

GENOME_LENGTH = 15                                                      # number of possible group
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7
GENERATIONS = 1500
p_c_min = 0.6
p_c_max = 0.9
p_m_min = 0.01
p_m_max = 0.1

m = 10                                                                   # Number of repairmen
w_max = 7                                                               # Maximum number of iterations for binary search

minimal_cut_sets = [
    {1},
    {2, 3},
    {4, 7},
    {4, 8},
    {5, 6, 7},
    {5, 6, 8}, 
    {9},
    {10, 11, 12},
    {13, 14},
    {13, 15}
]

# Load the Excel file
file_path = 'parameter_journal.xlsx'
all_sheets = pd.read_excel(file_path, sheet_name=None)
data = all_sheets['data_case2']
lamda = data['lamda'].to_numpy()
beta = data['beta'].to_numpy()
cip = data['cip'].to_numpy()
cic = data['cic'].to_numpy()
wip = data['wip'].to_numpy()
tie = data['tie'].to_numpy()
pii = data['pii'].to_numpy()

# Calculation
Cip = S + cip + pii*cdp*wip
print("Cip", Cip)
Cic = S + cic + pii*csysu
print("Cic", Cic)
num = ((S + cip)*np.power(lamda, beta))/(Cic*(beta-1))
xi0 = np.power(num, 1/beta)
print("xi0", np.round(xi0, 2))

# Define a sample objective function with x as a 12-variable array
def function1(x, Cic, beta, wip, Cip, lamda):
    return np.where(x >= 0, (Cic*(beta-1)* (x**beta)) + (Cic*beta*wip*(x**(beta-1))) - (Cip*(lamda**beta)), np.nan) 

# Initial guess for x
x0 = np.ones(15)  # Start with all variables initialized to 1

# Perform optimization
result = root(function1, x0, args=(Cic, beta, wip, Cip, lamda))
# Check, Print the result
x_opt = result.x
if result.success:
    print("Solution found:")
    print(x_opt)
else:
    print("Solution not found:", result.message)
# print(Cic)
# print(Cip)
# print(wip)
# print(beta)
# print(lamda)

phi_opt = (Cip + Cic*((x_opt/lamda)**beta))/(x_opt+wip)
print("phi_opt", phi_opt)
# D0i = np.array([0, 3, 3, 3, 3, 1, 4, 5, 5, 5, 5, 5])
D0i = 0
ti1 = x_opt - tie + D0i
print("ti1", np.round(ti1, 2))
print(np.sum(phi_opt))


# initialize genome
def random_genome(length):
    return [random.randint(1, length) for _ in range(length)]

# initialize population
def init_population(population_size, genome_length):
    return [random_genome(genome_length) for _ in range(population_size)]

# evaluation
def decode(genome):
    # Dictionary to map original group to new group starting from 1
    group_mapping = {}
    new_group_number = 1

    # Create mapping from original group to new group numbers
    for group in genome:
        if group not in group_mapping:
            group_mapping[group] = new_group_number
            new_group_number += 1

    # Dictionary to store new groups and their respective activities
    group_activities = {}

    # Populate the dictionary using the new group numbers
    for activity, group in enumerate(genome, start=1):
        new_group = group_mapping[group]
        if new_group in group_activities:
            group_activities[new_group].append(activity)
        else:
            group_activities[new_group] = [activity]

    # items(): method to return the dictionary's key-value pairs
    # sorted: displaying the Keys in Sorted Order
    # for group, activities in sorted(group_activities.items()):
    #     print(f"Group {group}: Activities {activities}")

    number_of_groups = len(group_activities)
    G_activity = sorted(group_activities.items())                       # group and its activity
    return number_of_groups, G_activity

# Check if any group contains a minimal cut set
def group_contains_cut_set(group_members, cut_sets):
    for cut_set in cut_sets:
        if cut_set.issubset(set(group_members)):
            return True
    return False

# Calculate piGk
def calculate_piGk(groups_list, cut_sets):
    result_array = np.zeros(len(groups_list), dtype=int)

    for i, (_, group_members) in enumerate(groups_list):
        if group_contains_cut_set(group_members, minimal_cut_sets):
            result_array[i] = 1

    # Print results
    # print("Activities in each group:", groups_list)
    # print("Result Array:", result_array)
    return result_array

# setup cost saving
def saveup_cost_saving(G_activity, S):
    B_S = []
    for group, activity in G_activity:
        buffer = (len(activity) - 1) * S
        B_S.append(buffer)
    return B_S   

# Map others wip to groups
def map_others_to_groups(groups, durations):
    duration_groups = []
    for group_id, members in groups:
        duration_groups.append((group_id, [durations[member - 1] for member in members]))
    return duration_groups


# First Fit Decreasing (FFD) method
def first_fit_decreasing(durations, m, D):
    durations = sorted(durations, reverse=True)
    repairmen = [0] * m
    for duration in durations:
        # Find the first repairman who can take this activity
        for i in range(m):
            if repairmen[i] + duration <= D:
                repairmen[i] += duration
                break
        else:
            return False
    return repairmen


# Binary search for optimal total maintenance duration
def multifit(durations, m, w_max):
    durations = sorted(durations, reverse=True)
    D_low = max(durations[0], sum(durations) / m)
    D_up = max(durations[0], 2 * sum(durations) / m)
    
    for w in range(w_max):
        D = (D_up + D_low) / 2
        repairmen = first_fit_decreasing(durations, m, D)
        if repairmen:
            D_up = D
            min_maintenance_duration = max(repairmen)
        else:
            D_low = D
    return min_maintenance_duration


def calculate_d_Gk(G_duration, m, w_max):
    d_Gk = []
    for _, durations in G_duration:
        optimal_duration = multifit(durations, m, w_max)
        d_Gk.append(optimal_duration)
    return d_Gk


def downtime_cost_critical_group(G_duration, cdp, pii, piGk, wip, m, w_max):
    CnotGkd = []
    DGk = np.array(calculate_d_Gk(G_duration, m, w_max))
    print("DGk", DGk)
    for i in range(len(G_duration)):
        group, dur = G_duration[i]
        _, piiGk = G_pii[i]
        dur = np.array(dur)
        piiGk = np.array(piiGk)
        print(f"group: {group}, durations: {dur}, pi: {piiGk}")
        temp1 = cdp*np.sum(piiGk*dur)                               # CnotGkd for each group
        CnotGkd.append(temp1)
    CGkd = piGk*cdp*DGk
    CnotGkd = np.array(CnotGkd)
    return CnotGkd, CGkd

# penalty cost
def penalty_cost(G_x_opt, G_ti1, G_Cic, G_beta, G_lamda, G_phi_opt):
    H1 = []
    tGk = []
    for i in range(len(G_x_opt)):
        group, x_opt_Gk = G_x_opt[i]
        _, ti1_Gk = G_ti1[i]
        _, Cic_Gk = G_Cic[i]
        _, beta_Gk = G_beta[i]
        _, lamda_Gk = G_lamda[i]
        _, phi_opt_Gk = G_phi_opt[i]

        # x_opt_Gk = np.array(x_opt_Gk)
        # ti1_Gk = np.array(ti1_Gk)
        # Cic_Gk = np.array(Cic_Gk)
        # beta_Gk = np.array(beta_Gk)
        # lamda_Gk = np.array(lamda_Gk)
        # phi_opt_Gk = np.array(phi_opt_Gk)

        # Initial guess for t
        initial_guess = [0.0]
        # Perform the minimization
        result = minimize(wrapper_P_Gk, initial_guess, args=(x_opt_Gk, ti1_Gk, Cic_Gk, beta_Gk, lamda_Gk, phi_opt_Gk))
        # Print the results
        print("Minimum value of the function: ", np.round(result.fun, decimals=2))
        print("Value of t at the minimum: ", np.round(result.x, decimals=2))
        # print("---------------------------------------------------")
        H1.append(np.round(result.fun, decimals=3))
        tGk.append(np.round(result.x, decimals=3))
    H1 = np.array(H1)
    tGk = np.array(tGk)
    tGk = tGk.reshape(-1)
    return H1, tGk

# Define the piecewise function
def P_i(t, x_opt, ti1, Cic, beta, lamda, phi_opt):
    return (Cic*(((x_opt + (t-ti1))/lamda)**beta)) - (Cic*((x_opt/lamda)**beta)) - ((t-ti1)*phi_opt)

# Define the sum function P_Gk
def P_Gk(t, x_opt_Gk, ti1_Gk, Cic_Gk, beta_Gk, lamda_Gk, phi_opt_Gk):
    total_sum = 0
    for x_opt, ti1, Cic, beta, lamda, phi_opt in zip(x_opt_Gk, ti1_Gk, Cic_Gk, beta_Gk, lamda_Gk, phi_opt_Gk):
        total_sum += P_i(t, x_opt, ti1, Cic, beta, lamda, phi_opt)
    return total_sum

# Define the wrapper function for minimization
def wrapper_P_Gk(t, x_opt_Gk, ti1_Gk, Cic_Gk, beta_Gk, lamda_Gk, phi_opt_Gk):
    return P_Gk(t[0], x_opt_Gk, ti1_Gk, Cic_Gk, beta_Gk, lamda_Gk, phi_opt_Gk)


def find_critical_sets_L(minimal_cut_sets):
    # Get all components in the system
    all_components = set.union(*minimal_cut_sets)

    # Function to find critical components when a specific component is maintained
    def find_critical_components(minimal_cut_sets, maintained_component):
        critical_components = set()
        for component in all_components:
            if component != maintained_component:
                # Simulate removing the maintained component
                modified_cut_sets = [
                    cut_set - {maintained_component} for cut_set in minimal_cut_sets
                ]
                # Check if the component becomes critical
                if {component} in modified_cut_sets:
                    critical_components.add(component)
        return critical_components

    # Calculate L_i for each component
    L = []
    for component in all_components:
        critical_set = find_critical_components(minimal_cut_sets, component)
        L.append((component, list(critical_set)))

    return L

def find_critical_sets_for_groups_P(minimal_cut_sets, groups):
    """
    Find the critical components for each group G_k when maintained.

    Parameters:
    - minimal_cut_sets: List of minimal cut sets (list of sets).
    - groups: List of tuples where the first element is the order of the group, 
              and the second is a list of components in the group.

    Returns:
    - A list of tuples where the first element is the order of the group and 
      the second element is the list of critical components.
    """
    # Get all components in the system
    all_components = set.union(*minimal_cut_sets)

    # Function to find critical components when a group is maintained
    def find_critical_components(minimal_cut_sets, group_maintained):
        critical_components = set()
        for component in all_components:
            if component not in group_maintained:
                # Simulate removing the maintained group from all minimal cut sets
                modified_cut_sets = [
                    cut_set - set(group_maintained) for cut_set in minimal_cut_sets
                ]
                # Check if the component becomes critical
                if {component} in modified_cut_sets:
                    critical_components.add(component)
        return critical_components

    # Calculate critical components for each group
    critical_groups = []
    for order, group_components in groups:
        critical_set = find_critical_components(minimal_cut_sets, group_components)
        critical_groups.append((order, list(critical_set)))

    return critical_groups


print("-----------------------------")
# # Test main
# genome = random_genome(GENOME_LENGTH)
# genome = [1, 4, 6, 3, 4, 3, 6, 6, 1, 3, 6, 4, 6, 3, 1]
genome = [1, 4, 6, 3, 4, 3, 6, 6, 1, 3, 6, 4, 6, 3, 1]
N, G_activity = decode(genome)
print(f"Genome: {genome}")
print(f"Activities in each group: {G_activity}")
print(f"Number of group: {N}")
UGk = saveup_cost_saving(G_activity, S)
print(f"Setup cost saving in each group: {UGk}")
piGk = calculate_piGk(G_activity, minimal_cut_sets)
print("piGk", piGk)
# print(wip)

G_duration = map_others_to_groups(G_activity, wip)
print(f"Durations in each group: {G_duration}")
G_pii = map_others_to_groups(G_activity, pii)
print(f"pii in each group: {G_pii}")
print("pii", pii)

CnotGkd, CGkd = downtime_cost_critical_group(G_duration, cdp, pii, piGk, wip, m, w_max)
print("CnotGkd:", CnotGkd)
print("CGkd:", CGkd)



G_x_opt = map_others_to_groups(G_activity, x_opt)
G_ti1 = map_others_to_groups(G_activity, ti1)
G_Cic = map_others_to_groups(G_activity, Cic)
G_beta = map_others_to_groups(G_activity, beta)
G_lamda = map_others_to_groups(G_activity, lamda)
G_phi_opt = map_others_to_groups(G_activity, phi_opt)


H1, tGk = penalty_cost(G_x_opt, G_ti1, G_Cic, G_beta, G_lamda, G_phi_opt)
print("H1", H1)
print("tGk", tGk)
# print(np.shape(H1))
# print(np.shape(tGk))

L = find_critical_sets_L(minimal_cut_sets)
print("L:", L)
# L_duration = map_others_to_groups(L, wip)
L_beta = map_others_to_groups(L, beta)
L_lamda = map_others_to_groups(L, lamda)
L_tie = map_others_to_groups(L, tie)
L_ti1 = map_others_to_groups(L, ti1)

def calculate_CnotGks(wip, L_beta, L_lamda, L_tie, ti1, pii, csysu, G_activity):
    Cis = []
    for i in range(len(L_beta)):
        group, beta_l = L_beta[i]
        _, lamda_l = L_lamda[i]
        _, tie_l = L_tie[i]

        dur = wip[group-1]
        beta_l = np.array(beta_l)
        lamda_l = np.array(lamda_l)
        tie_l = np.array(tie_l)

        print("group", group)
        print("dur", dur)
        print("beta_l", beta_l)
        print("lamda_l", lamda_l)
        print("tie_l", tie_l)

        a_l = tie_l + ti1[group-1]
        print("a_l" ,a_l)
        temp1 = np.sum((((a_l + dur)/lamda_l)**beta_l) - ((a_l/lamda_l)**beta_l))
        temp2 = (1-pii[group-1])*csysu*temp1
        print("Cis", temp2)
        Cis.append(temp2)
    Cis = np.array(Cis)
    print(Cis)
    G_CnotGks = map_others_to_groups(G_activity, Cis)
    print(G_CnotGks)
    CnotGks = []
    for j in range(len(G_CnotGks)):
        _, temp3 = G_CnotGks[j]
        temp3 = np.array(temp3)
        temp4 = np.sum(temp3)
        CnotGks.append(temp4)
    CnotGks = np.array(CnotGks)
    return CnotGks

CnotGks = calculate_CnotGks(wip, L_beta, L_lamda, L_tie, ti1, pii, csysu, G_activity)
print(CnotGks)

P = find_critical_sets_for_groups_P(minimal_cut_sets, G_activity)
print("P:",P)
P_beta = map_others_to_groups(P, beta)
P_lamda = map_others_to_groups(P, lamda)
P_tie = map_others_to_groups(P, tie)


def calculate_CGks(P_beta, P_lamda, P_tie, csysu, m, w_max, tGk, piGk):
    DGk = np.array(calculate_d_Gk(G_duration, m, w_max))
    print("DGk", DGk)
    CGks = []
    for i in range(len(P_beta)):
        group, beta_p = P_beta[i]
        _, lamda_p = P_lamda[i]
        _, tie_p = P_tie[i]

        dur_p = DGk[group-1]
        beta_p = np.array(beta_p)
        lamda_p = np.array(lamda_p)
        tie_p = np.array(tie_p)

        print("group", group)
        print("dur_p", dur_p)
        print("beta_p", beta_p)
        print("lamda_p", lamda_p)
        print("tie_p", tie_p)
        print("tGk", tGk)

        a_p = tie_p + tGk[group-1]
        print("a_p", a_p)
        temp1 = np.sum((((a_p + dur_p)/lamda_p)**beta_p) - ((a_p/lamda_p)**beta_p))
        temp2 = (1-piGk[group-1])*csysu*temp1
        CGks.append(temp2)
    CGks = np.array(CGks)
    return CGks



CGks = calculate_CGks(P_beta, P_lamda, P_tie, csysu, m, w_max, tGk, piGk)
print(CGks)


print(f"Setup cost saving in each group: {UGk}")
print("H1", H1)
print("CnotGkd:", CnotGkd)
print("CGkd:", CGkd)
print("CnotGks", CnotGks)
print("CGks", CGks)
EPGk = UGk - (H1+CGks) -(CGkd-CnotGkd-CnotGks)
print("EPGk", EPGk)
print("EPGk double check", UGk-H1-(CGkd-CnotGkd)-(CGks-CnotGks))
EPS = np.sum(EPGk)
print("EPS", EPS)


def small_function_p(t, D, beta, lamda, tie):
    a_p = tie + t
    return ((a_p + D)/lamda)**beta - (a_p/lamda)**beta

def small_function_p_sum(t, D_p, beta_p, lamda_p, tie_p):
    total_sum = 0
    for D, beta, lamda, tie in zip(D_p, beta_p, lamda_p, tie_p):
        total_sum += small_function_p(t, D, beta, lamda, tie)
    return total_sum

def wrapper_small_function_p(t, D_p, beta_p, lamda_p, tie_p, pi_p):
    return (1-pi_p)*csysu*small_function_p_sum(t[0], D_p, beta_p, lamda_p, tie_p)

def sum_of_penalty_cost_and_CGks(t, x_opt_Gk, ti1_Gk, Cic_Gk, beta_Gk, lamda_Gk, phi_opt_Gk, D_p, beta_p, lamda_p, tie_p, pi_p):
    return wrapper_small_function_p(t, D_p, beta_p, lamda_p, tie_p, pi_p) + P_Gk(t, x_opt_Gk, ti1_Gk, Cic_Gk, beta_Gk, lamda_Gk, phi_opt_Gk)


# penalty cost
def optimize_penalty_cost_and_CGks(G_x_opt, G_ti1, G_Cic, G_beta, G_lamda, G_phi_opt, P_beta, P_lamda, P_tie, csysu, m, w_max, tGk, piGk):
    sum_H1_CGks = []
    tGk = []
    DGk = np.array(calculate_d_Gk(G_duration, m, w_max))
    print("DGk", DGk)
    P_D = result = [(item[0], [DGk[i]] * len(item[1])) for i, item in enumerate(P)]
    print("P_D", P_D)
    H1_check = []
    CGks_check = []
    for i in range(len(G_x_opt)):
        group, x_opt_Gk = G_x_opt[i]
        _, ti1_Gk = G_ti1[i]
        _, Cic_Gk = G_Cic[i]
        _, beta_Gk = G_beta[i]
        _, lamda_Gk = G_lamda[i]
        _, phi_opt_Gk = G_phi_opt[i]

        _, beta_p = P_beta[i]
        _, lamda_p = P_lamda[i]
        _, tie_p = P_tie[i]
        _, D_p = P_D[i]
        beta_p = np.array(beta_p)
        lamda_p = np.array(lamda_p)
        tie_p = np.array(tie_p)
        D_p = np.array(D_p)
        pi_p = piGk[group-1]
        print(beta_p)
        print(lamda_p)
        print(tie_p)
        print(D_p)
        print(pi_p)
        # Initial guess for t
        initial_guess = [0.0]
        # Perform the minimization
        result = minimize(sum_of_penalty_cost_and_CGks, initial_guess, args=(x_opt_Gk, ti1_Gk, Cic_Gk, beta_Gk, lamda_Gk, phi_opt_Gk, D_p, beta_p, lamda_p, tie_p, pi_p))
        # Print the results
        print("Minimum value of the function: ", np.round(result.fun, decimals=2))
        print("Value of t at the minimum: ", np.round(result.x, decimals=2))
        sum_H1_CGks.append(np.round(result.fun, decimals=3))
        tGk.append(np.round(result.x, decimals=3))

        temp5 = wrapper_P_Gk(result.x, x_opt_Gk, ti1_Gk, Cic_Gk, beta_Gk, lamda_Gk, phi_opt_Gk)
        temp6 = wrapper_small_function_p(result.x, D_p, beta_p, lamda_p, tie_p, pi_p)
        H1_check.append(np.round(temp5, decimals=3))
        CGks_check.append(np.round(temp6, decimals=3))

    print("H1_check", H1_check)
    print("CGks_check", CGks_check)
    sum_H1_CGks = np.array(sum_H1_CGks)
    tGk = np.array(tGk)
    tGk = tGk.reshape(-1)

    return sum_H1_CGks, tGk

# optimize_penalty_cost_and_CGks(G_x_opt, G_ti1, G_Cic, G_beta, G_lamda, G_phi_opt, P_beta, P_lamda, P_tie, csysu, m, w_max, tGk, piGk)

sum_H1_CGks, new_tGk = optimize_penalty_cost_and_CGks(G_x_opt, G_ti1, G_Cic, G_beta, G_lamda, G_phi_opt, P_beta, P_lamda, P_tie, csysu, m, w_max, tGk, piGk)
print(sum_H1_CGks)
print("new_tGk", new_tGk)

profit = UGk - sum_H1_CGks -(CGkd-CnotGkd-CnotGks)
print(profit)
profit_sum = np.sum(profit)
print("profit_sum", profit_sum)