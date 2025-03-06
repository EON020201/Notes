class Solution:
    def minSubArrayLen(self, target, nums):
        
        n = len(nums)
        ans = n + 1
        left = 0
        sum = 0
        for right, x in enumerate(nums):
            sum += x
            while sum >= target:
                ans = min(ans, right - left + 1)
                sum -= nums[left]
                left += 1
        return ans if ans <= n else 0
    
if __name__ == '__main__':
    solution = Solution()
    target = 7
    nums = [2,3,1,2,4,3]
    print(solution.minSubArrayLen(target, nums))