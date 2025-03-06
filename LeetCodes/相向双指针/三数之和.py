class Solution:
    def threeSum(self, nums):
        nums.sort()
        n = len(nums)
        ans = []
        
        for i in range(n-2):
            x = nums[i]

            if i > 0 and x == nums[i-1]:
                continue

            if x + nums[i+1] + nums[i+2] > 0:
                break

            if x + nums[n-2] + nums[n-1] < 0:
                continue


            j = i + 1
            k = n - 1
            while j < k:
                sum = x + nums[j] + nums[k]

                if sum < 0:
                    j += 1
                if sum > 0:
                    k -= 1
                else:
                    ans.append([x, nums[j], nums[k]])

                    j += 1
                    while j < k and nums[j] == nums[j-1]:
                        continue
                    
                    k -= 1
                    while j < k and nums[k] == nums[k+1]:
                        continue

        return ans
    
if __name__ == '__main__':
    
    solution = Solution()
    nums = [-1, 0, 1, 2, -1, -4]
    print(solution.threeSum(nums))