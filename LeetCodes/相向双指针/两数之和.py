# 输入：numbers = [2, 7, 11, 15], target = 9
# 输出：[1, 2]

class Solution:
    def twoSum(self, numbers, target):
        left = 0
        right = len(numbers) - 1
        while True:
            sum = numbers[left] + numbers[right]
            if sum == target:
                break
            if sum < target:
                left += 1
            if sum > target:
                right -= 1
        return [left+1, right+1]

    
if __name__ == "__main__":
    numbers = [2, 7, 11, 15]
    target = 9
    solution =  Solution()
    print(solution.twoSum(numbers, target))



