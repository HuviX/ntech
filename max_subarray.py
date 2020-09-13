#Используем Kadane's algorithm для нахождений максимального подсписка в списке (сумма элементов максимальна).

def findMaxSubArray(A):
    start = 0
    end = 0
    best_sum = float('-inf') # Необходимо, чтобы алгоритм работал, если на вход подаются только отрицательные элементы. Если есть положительные,
    #то можно присвоить best_sum = 0
    current_sum = 0
    #enumerate необходим, чтобы отслеживать индекс элемента и сам элемент. В то время, как в классическом алгоритме необходимо
    #знать только значение текущего элемента.
    for i, x in enumerate(A): 
        if x > current_sum + x:
            start = i
            current_sum = x
        else:
            current_sum += x
        
        if best_sum < current_sum:
            best_sum = current_sum
            end = i + 1 # +1 нужен, чтобы сделать правильный слайс, т.к. необходимо вернуть элементы до индекса end включительно.
    return A[start: end] #В качестве результата возвращаем слайс с максимальной суммой.

def main(): 
    assert findMaxSubArray([-2,1,-3,4,-1,2,1,-5,4]) == [4, -1, 2, 1]

    assert findMaxSubArray([-2, -1]) == [-1]

    assert findMaxSubArray([5]) == [5]
    print('Ok')

    # test_array = [] #Your Input
    # ans = findMaxSubArray(test_array)
    # print(ans)
    

if __name__ == '__main__':
    main()