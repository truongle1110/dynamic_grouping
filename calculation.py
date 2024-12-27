import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
from scipy.optimize import root

S = 15
cdp = 80
csysu = 50
delta1min = 15
delta2min = 2

GENOME_LENGTH = 12                                                      # number of possible group
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7
GENERATIONS = 1500

m = 10                                                                   # Number of repairmen
w_max = 7                                                               # Maximum number of iterations for binary search

minimal_cut_sets = [
    {1},
    {2, 3},
    {2, 4},
    {3, 4},
    {5},
    {6},
    {7},
    {8, 10},
    {9, 10},
    {11, 12}
]

# Load the Excel file
file_path = 'parameter.xlsx'
all_sheets = pd.read_excel(file_path, sheet_name=None)
data = all_sheets['data']
lamda = data['lamda'].to_numpy()
beta = data['beta'].to_numpy()
cip = data['cip'].to_numpy()
cic = data['cic'].to_numpy()
wip = data['wip'].to_numpy()
tie = data['tie'].to_numpy()
pii = data['pii'].to_numpy()

# Calculation
Cip = S + cip + pii*cdp*wip
# print(Cip)
Cic = S + cic + pii*csysu
# print(Cic)
num = ((S + cip)*np.power(lamda, beta))/(Cic*(beta-1))
xi0 = np.power(num, 1/beta)
# print(np.round(xi0, 2))

# Define a sample objective function with x as a 12-variable array
def function1(x, Cic, beta, wip, Cip, lamda):
    return np.where(x >= 0, (Cic*(beta-1)* (x**beta)) + (Cic*beta*wip*(x**(beta-1))) - (Cip*(lamda**beta)), np.nan) 

# Initial guess for x
x0 = np.ones(12)  # Start with all variables initialized to 1

# Perform optimization
result = root(function1, x0, args=(Cic, beta, wip, Cip, lamda))
# Check, Print the result
x_opt = result.x
if result.success:
    print("Solution found:")
    print(result.x)
else:
    print("Solution not found:", result.message)
# print(Cic)
# print(Cip)
# print(wip)
# print(beta)
# print(lamda)

phi_opt = (Cip + Cic*((x_opt/lamda)**beta))/(x_opt+wip)
print(phi_opt)

ti1 = x_opt - tie
print(ti1)
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
def calculate_pGk(groups_list, cut_sets):
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

# Map durations wip to groups
def map_durations_to_groups(groups, durations):
    duration_groups = []
    for group_id, members in groups:
        duration_groups.append((group_id, [durations[member - 1] for member in members]))
    return duration_groups

# Map pii to groups
def map_durations_to_groups(groups, piis):
    pii_groups = []
    for group_id, members in groups:
        pii_groups.append((group_id, [piis[member - 1] for member in members]))
    return pii_groups

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


def downtime_cost_critical_group(G_duration, cdp, pii, pGk, wip, m, w_max):
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
    CGkd = pGk*cdp*DGk
    return CnotGkd, CGkd

print("-----------------------------")
# # Test main
genome = random_genome(GENOME_LENGTH)
# genome = [11, 2, 11, 4, 6, 4, 7, 8, 11, 11, 3, 3]
N, G_activity = decode(genome)
print(f"Genome: {genome}")
print(f"Activities in each group: {G_activity}")
print(f"Number of group: {N}")
UGk = saveup_cost_saving(G_activity, S)
print(f"Setup cost saving in each group: {UGk}")
pGk = calculate_pGk(G_activity, minimal_cut_sets)
print("pGk", pGk)
# print(wip)

G_duration = map_durations_to_groups(G_activity, wip)
print(f"Durations in each group: {G_duration}")
G_pii = map_durations_to_groups(G_activity, pii)
print(f"pii in each group: {G_pii}")
print("pii", pii)

CnotGkd, CGkd = downtime_cost_critical_group(G_duration, cdp, pii, pGk, wip, m, w_max)
print("CnotGkd:", CnotGkd)
print("CGkd:", CGkd)