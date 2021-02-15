# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # initialize a ListNode and mark it as the current node as well as the start node
        curr_node = ListNode()
        start = curr_node
        curr_digit = 0
        carry_digit = 0
        end = False
 
        
        if l1 == None and l2 == None:
            return None
        
        while end == False:
            if l1 == None and l2 == None:
                curr_digit = carry_digit
            elif l1 == None:
                curr_digit = l2.val + carry_digit
                l2 = l2.next
            elif l2 == None:
                curr_digit = l1.val + carry_digit
                l1 = l1.next
            else:
                curr_digit = l1.val + l2.val + carry_digit
                l1 = l1.next
                l2 = l2.next
            
            carry_digit = curr_digit//10
            curr_digit = curr_digit%10
            
            curr_node.val = curr_digit
            if l1 == None and l2 == None and carry_digit == 0:
                end = True
            else:
                curr_node.next = ListNode()
                curr_node = curr_node.next
        
        return start

x = ListNode(2, ListNode(4, ListNode(3, ListNode(8, None))))
y = ListNode(5, ListNode(6, ListNode(4, None))) 

ans = Solution()
startNode = Solution.addTwoNumbers(ans, x, y)  