def merge_sort(my_list: list):
    # implementation of mergesort

    # handle error cases
    if my_list is None or len(my_list) == 0:
        return []

    # return if one element is left
    if len(my_list) == 1:
        return my_list

    # recursively call mergesort to get sorted left and right part of list
    left = merge_sort(my_list[0 : len(my_list) // 2])
    right = merge_sort(my_list[len(my_list) // 2 :])

    # get result
    result = []
    idx_left = 0
    idx_right = 0

    # merge lists
    while idx_left < len(left) or idx_right < len(right):

        # add remaining elements from left list
        if idx_left == len(left):
            [result.append(right[x]) for x in range(idx_right, len(right))]
            break

        # add remaining elements from right list
        elif idx_right == len(right):
            [result.append(left[x]) for x in range(idx_left, len(left))]
            break

        # add smaller element
        if left[idx_left] >= right[idx_right]:
            result.append(right[idx_right])
            idx_right += 1
        elif left[idx_left] < right[idx_right]:
            result.append(left[idx_left])
            idx_left += 1

    return result


sorted_list1 = sorted(my_list)
sorted_list2 = merge_sort(my_list)
