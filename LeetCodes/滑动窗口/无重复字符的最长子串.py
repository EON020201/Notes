from collections import Counter

class Solution:
    def lengthOfLongestSubstring(self, s):
        left = ans = 0
        cnt = Counter()
        for right, x in enumerate(s):
            cnt[x] += 1
            while cnt[x] > 1:
                cnt[s[left]] -= 1
                left += 1
            ans = max(ans, right - left + 1)
        return ans
    
if __name__ =='__main__':
    solution = Solution()
    s = "abcabcbb"
    print(solution.lengthOfLongestSubstring(s))
