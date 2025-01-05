# minimal_cut_sets = [
#     {1}, {2, 3}, {2, 4}, {3, 4}, {5}, {6}, {7}, {8, 10}, {9, 10}, {11, 12}
# ]

# def find_critical_sets(minimal_cut_sets):
#     # Get all components in the system
#     all_components = set.union(*minimal_cut_sets)

#     # Function to find critical components when a specific component is maintained
#     def find_critical_components(minimal_cut_sets, maintained_component):
#         critical_components = set()
#         for component in all_components:
#             if component != maintained_component:
#                 # Simulate removing the maintained component
#                 modified_cut_sets = [
#                     cut_set - {maintained_component} for cut_set in minimal_cut_sets
#                 ]
#                 # Check if the component becomes critical
#                 if {component} in modified_cut_sets:
#                     critical_components.add(component)
#         return critical_components

#     # Calculate L_i for each component
#     L = {}
#     for component in all_components:
#         L[component] = find_critical_components(minimal_cut_sets, component)

#     return L

# critical_sets = find_critical_sets(minimal_cut_sets)
# print(critical_sets)


def find_critical_sets_for_groups(minimal_cut_sets, groups):
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

minimal_cut_sets = [
    {1}, {2, 3}, {2, 4}, {3, 4}, {5}, {6}, {7}, {8, 10}, {9, 10}, {11, 12}
]

groups = [
    (1, [1, 3, 4, 5, 6, 7]),
    (2, [2]),
    (3, [8, 9, 11]),
    (4, [10, 12])
]

critical_components_for_groups = find_critical_sets_for_groups(minimal_cut_sets, groups)
print(critical_components_for_groups)
# for order, critical_set in critical_components_for_groups:
#     print(f"Order: {order}, Critical Components: {critical_set}")
