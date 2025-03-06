class Solution:
    def numSubarrayProductLessThanK(self, k, nums):
        if k <= 1:
            return 0
        left = 0
        prod = 1
        ans = 0
        for right, x in enumerate(nums):
            prod *= x
            while prod >= k:
                prod /= nums[left]
                left += 1
            ans += right - left + 1
        return ans
    
if __name__ == '__main__':
    solution = Solution()
    k = 100
    nums = [10,5,2,6]
    print(solution.numSubarrayProductLessThanK(k, nums))
