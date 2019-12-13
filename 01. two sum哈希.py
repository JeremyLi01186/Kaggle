'''
解法一：暴力法。
时间复杂度：O(n^2)
空间复杂度：O(1)
'''
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums_len = len(nums)
        for i in range(0, nums_len):
            for j in range(i + 1, nums_len):
                if nums[i] + nums[j] == target:
                    return [i, j]



'''
解法二：两遍哈希。第一次循环，将列表中的元素以 key = element, value = index 的形式添加到 dict 中；第二次循环，遍历 dict ，查找是否存在另一个元素满足条件：target - element in dikt。但是，该方法的不足在于无法处理列表中两个值相等的元素，如：[9, 2, 0, 8, 2, 1], target = 4。
时间复杂度：O(n)
空间复杂度：O(n)
'''
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dikt = {}
        for i in range(0, len(nums)):
            dikt[nums[i]] = i
        for k in nums.keys():
            if target - k in nums:
                return [dikt[k], nums[target - k]]


'''
解法三：一遍哈希。为了解决解法二中的无法处理的问题，对次作出改进。采用一次循环，每次将列表元素添加到 dict 中之前，先验证是否满足 target - element in dikt。如果不满足则添加到 dict 中，否则返回对应的两个 index。
时间复杂度：O(n)
空间复杂度：O(n)
'''
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dikt = {}
        for i in range(0, len(nums)):
            if target - nums[i] in dikt:
                return [i, dikt[target - nums[i]]]
            dikt[nums[i]] = i




def twoSum(self, nums, target):
        seen = {}
        for i, v in enumerate(nums):
            remaining = target - v
            if remaining in seen:
                return [seen[remaining], i]
            seen[v] = i
        return []