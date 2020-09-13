def max_array_ind(arr):
    start = 0
    end = 0
    best_sum = float('-inf')
    current_sum = 0
    for i, x in enumerate(arr):
        if x > current_sum + x:
            start = i
            current_sum = x
        else:
            current_sum += x
        
        if best_sum < current_sum:
            best_sum = current_sum
            end = i + 1
    return arr[start: end]

    
def main():
    assert max_array_ind([-2,1,-3,4,-1,2,1,-5,4]) == [4, -1, 2, 1]
    assert max_array_ind([-2, -1]) == [-1]
    assert max_array_ind([5]) == [5]
    print('Ok')

if __name__ == '__main__':
    main()