from typing import List, Tuple

def generate_combinations(list_of_lists: List[List[int]], t: int) -> List[Tuple[any]]:
    if t == 0:
        return [tuple()]

    if not list_of_lists:
        return []

    first_list = list_of_lists[0]
    rest_lists = list_of_lists[1:]

    combinations = []
    for element in first_list:
        for sub_combination in generate_combinations(rest_lists, t - 1):
            combinations.append((element,) + sub_combination)

    return combinations

# Example usage:
list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]
list3 = [9, 10, 11, 12]
list4 = [13, 14, 15, 16]

myListOfLists = [list1, list2, list3, list4]
t = 4

combinations = generate_combinations(myListOfLists, t)
print(combinations)