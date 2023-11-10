## STL

### vector

常用函数

1.push_back 在数组的最后添加一个数据

2.pop_back 去掉数组的最后一个数据

3.at 得到编号位置的数据

4.begin 得到数组头的指针

5.end 得到数组的最后一个单元**+1的指针**

6.front 得到数组头的**引用**

7.back 得到数组的最后一个单元的引用

8.max_size 得到vector最大可以是多大

9.capacity 当前vector分配的大小

10.size 当前使用数据的大小

11.resize 改变当前使用数据的大小，如果它比当前使用的大，则填充默认值

12.reserve 改变当前vecotr所分配空间的大小

13.erase 删除指针指向的数据项

14.clear 清空当前的vector

15.rbegin 将vector反转后的开始指针返回(其实就是原来的end-1)

16.rend 将vector反转构的结束指针返回(其实就是原来的begin-1)

17.empty 判断vector是否为空

18.swap 与另一个vector交换数据

声明迭代器：

~~~c++
vector<int> vec;
vector<int>::iterator it;
it = vec.begin();
~~~

定义二维数组：

~~~c++
int N=5, M=6; 
vector<vector<int> > obj(N, vector<int>(M)); //定义二维动态数组5行6列 
~~~

vector的插入删除都是使用迭代器进行的，且迭代器参数位置在前

~~~c++
vec.insert(vec.begin()+i,val);
vec.erase(vec.begin()+i);
vec.erase(vec.begin()+i,vec.begin()+j);
~~~





### map

插入元素

~~~c++
// 定义一个map对象
map<int, string> mapStudent;
 
// 第一种 用insert函數插入pair
mapStudent.insert(pair<int, string>(000, "student_zero"));
 
// 第二种 用insert函数插入value_type数据
mapStudent.insert(map<int, string>::value_type(001, "student_one"));
 
// 第三种 用"array"方式插入
mapStudent[123] = "student_first";
mapStudent[456] = "student_second";
~~~

查找元素

~~~c++
// find 返回迭代器指向当前查找元素的位置否则返回map::end()位置
iter = mapStudent.find("123");
 
if(iter != mapStudent.end())
       cout<<"Find, the value is"<<iter->second<<endl;
else
   cout<<"Do not Find"<<endl;
~~~

即通过返回的迭代器是否等于end()，来判断是否存在

删除元素

可以通过迭代器或者key来删除，都是使用erase函数

~~~c++
//迭代器刪除
iter = mapStudent.find("123");
mapStudent.erase(iter);
 
//用关键字刪除
int n = mapStudent.erase("123"); //如果刪除了會返回1，否則返回0
 
//用迭代器范围刪除 : 把整个map清空
mapStudent.erase(mapStudent.begin(), mapStudent.end());
//等同于mapStudent.clear()
~~~



### string

+ sub_str(int begin_idx, int len)：返回从begin_idx开始的，长度最长为len的子串（不足len则提前返回），用于截取子串时很好用

+ string中的find成员函数，返回的是int类型的**下标（而不是迭代器）**，因此结合sub_str使用如下：

  ~~~c++
  string sub=str.sub_str(0,str.find(ch));
  ~~~

  其中，find的第一个参数也可以是一个字符串，表示找到该子串第一次出现的位置，返回的也是下标
  
+ replace成员函数，接收两个迭代器代表始末位置，将这部分替换为一个新的字符串

  ~~~c++ 
  str.replace(str.begin()+i,str.end(),new_str);
  ~~~

  





### algorithm

+ reverse(iter_begin, iter_end)，将两个迭代器之间的范围内的元素值进行反转

+ swap(&a, &b)，传入两个元素，会将其值进行交换

+ unique(begin,end)，对begin和end之间的元素，消除掉相邻重复元素，并返回结尾的迭代器，可以使用`unique(begin,end)-begin()`来得到操作后的数组长度，一般在unique之前需要先排序

+ minmax_element(begin,end)，求begin和end之间的元素的最大最小值，其返回值为两个迭代器组成的pair，可以如下写：

  ~~~c++
  auto minmax = minmax_element(nums.begin(),nums.end());
  int min_val=*minmax.first, max_val=*minmax.second;
  ~~~

  

### 技巧

#### 保留n位小数

```c++
cout<<fixed<<setprecision(2)<<x<<endl;
```



#### 字符串分割

使用istringstream和getline

~~~c++
istringstream sin(str);
string temp;
vector<string> result;
while(getline(sin, temp, '.'))
    result.push_back(temp);
~~~

或者使用strtok也可以

~~~c++
char * strs = new char[str.length() + 1] ; //不要忘了
strcpy(strs, str.c_str()); 
 
char * d = new char[delim.length() + 1];
strcpy(d, delim.c_str());

char *p = strtok(strs, d);
while(p) {
	string s = p; //分割得到的字符串转换为string类型
	res.push_back(s); //存入结果数组
	p = strtok(NULL, d);
}
~~~





#### 字符串和int相互转换

string到int：使用`int a = atoi(str.c_str())`

int到string：直接使用`string s = to_string(a)`



#### 字符串和字符数组转换

char*、char[] 转为 string：直接赋值即可

~~~c++
string s;
char* ch_p = "abcd";
char ch[4] = "abc";
s = ch_p;
s = ch;
~~~

string转为char*

~~~c++
char* ch;
strcpy(ch,s.c_str());
~~~



#### 字符串的输入

若是要遇到空格就停止，则直接cin即可

~~~c++
cin>>str;
while(cin>>str){
	...
}
~~~

若是要读入一整行，则使用getline()

~~~c++
string tmp;
getline(cin,tmp);
~~~

若是`getline()`前有别的输入，需要使用`cin.ignore()`扔掉最后一个换行



#### 求全排列

注意用之前，要对整个数组进行升序排序

~~~c++
#include <algorithm>
while(next_permutation(vec.begin(),vec.end()))
~~~







## 链表

### 反转链表

#### 全部反转

通常跟链表相关的，首先排除空链表和长度为1等特殊情况，然后设置两个指针，一个指向当前节点，另一个指向父节点

下面代码中，注意pre指的是第一个需要反转的点，p指的是第二个

主循环

[链接](https://www.nowcoder.com/practice/75e878df47f24fdc9dc3e400ec6058ca?tpId=295&tqId=23286&ru=/exam/intelligent&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Fintelligent%3FquestionJobId%3D10%26tagId%3D21000)

~~~c++
/**
 * struct ListNode {
 *	int val;
 *	struct ListNode *next;
 *	ListNode(int x) : val(x), next(nullptr) {}
 * };
 */
#include <cstddef>
class Solution {
public:
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param head ListNode类 
     * @return ListNode类
     */
    ListNode* ReverseList(ListNode* head) {
        // write code here
        if(head == NULL){
            return head;
        }
        ListNode* pre = head;
        ListNode* cur = head->next;
        head->next = NULL;
        ListNode* tmp;
        while(cur!=NULL){
            tmp = cur->next;
            cur->next = pre;
            pre = cur;
           	cur = tmp;
        }
        return pre;
    }
};
~~~



#### 部分反转（m到n之间）

在全部反转的基础上修改，思路是找到最后一个不需要动的点（statical）以及第一个需要动的点（statical2），这两个在后续的链表拼接中是非常重要的（将首段和中段、中段和末段连起来）

这里有一个很重要的边界条件，若是m=1，则第一个点也需要反转，找不到最后一个不需要动的点（statical），因此可以在最开始就为链表增加一个虚假的头结点add_head，可以去掉这种情况。

[链接](https://www.nowcoder.com/practice/b58434e200a648c589ca2063f1faf58c?tpId=295&tqId=654&ru=/exam/intelligent&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Fintelligent%3FquestionJobId%3D10%26tagId%3D21000)

~~~C++
/**
 * struct ListNode {
 *	int val;
 *	struct ListNode *next;
 *	ListNode(int x) : val(x), next(nullptr) {}
 * };
 */
#include <cstddef>
class Solution {
public:
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param head ListNode类 
     * @param m int整型 
     * @param n int整型 
     * @return ListNode类
     */
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        // write code here
        if(head == NULL){
            return head;
        }
        int i = 0;
        ListNode* add_head = new ListNode(0);
        add_head->next = head;
        ListNode* p = add_head;
        for(int i=0;i<m-1;i++){
            p = p->next;
        }
        if(p==NULL || p->next==NULL){
            return head;
        }
        ListNode* statical = p; //statical是最后一个不需要动的
        ListNode* parent_p = p->next; //parent_p是第一个需要动的
        ListNode* statical2 = parent_p; //statical2是第一个需要动的(后续要更新其next)
        ListNode* tmp;
        p = parent_p->next;//p是第二个需要动的
        for(int i=0;i<(n-m);i++){
            tmp = p->next;
            p->next = parent_p;
            parent_p = p;
            p = tmp;
        }
        statical->next = parent_p;
        statical2->next = p;
        return add_head->next;
    }
};
~~~



或者有第二种未实现的思路：首先找到最后一个不需要动的点（statical），以及第一个需要动的点，然后遍历k个，每遍历到一个，就将这个元素插到statical后面



#### 分组反转，每组k个

两种思路，一种是递归的方法，一种是非递归的

[链接](https://www.nowcoder.com/practice/b49c3dc907814e9bbfa8437c251b028e?tpId=295&tqId=722&ru=/exam/intelligent&qru=/ta/format-top101/question-ranking&sourceUrl=%2Fexam%2Fintelligent%3FquestionJobId%3D10%26tagId%3D21000)

非递归的：

实现一个从起始位置开始，对接下来k个元素进行反转的函数，然后依旧是为原始链表添加一个虚拟的头（为了连接，必须记录每段之前的那个元素），然后分段反转，做好连接即可

~~~c++
/**
 * struct ListNode {
 *	int val;
 *	struct ListNode *next;
 *	ListNode(int x) : val(x), next(nullptr) {}
 * };
 */
#include <cstdio>
class Solution {
public:
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param head ListNode类 
     * @param k int整型 
     * @return ListNode类
     */
    ListNode* ReverseList(ListNode* head, int k) {
        // write code here
        if(head == NULL){
            return head;
        }
        ListNode* parent_p = head;
        ListNode* p = head->next;
        ListNode* tmp;
        for(int i=0;i<k-1;i++){
            tmp = p->next;
            p->next = parent_p;
            parent_p = p;
            p = tmp;
        }
        return parent_p;
    }

    ListNode* reverseKGroup(ListNode* head, int k) {
        if(head == NULL){
            return head;
        }
        ListNode* add_head = new ListNode(0);
        add_head->next = head;
        ListNode* parent_p = add_head;
        ListNode* p = add_head->next;
        ListNode* q = p;
        bool flag;
        while(p!=NULL){
            for(int i=0;i<k;i++){
                if(q){
                    q = q->next;
                    flag = true;
                }
                else{
                    flag = false;
                    break;
                }
            }
            // p是这一组开始的第一个，q是这一组结束后的下一个
            if(flag){
                ListNode* new_head = ReverseList(p, k);
                parent_p->next = new_head;
                p->next = q;
                parent_p = p;
                p=q;
            }
            else{
                break;
            }
        }
        return add_head->next;
    }
};
~~~

重点看下递归的方法：

这里考虑到递归，是由于递归可以从后往前实现，这样每次只需要将当前节点的next指向递归返回的新的链表头即可，连接上比较简单

~~~c++
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        //找到每次翻转的尾部
        ListNode* tail = head; 
        //遍历k次到尾部
        for(int i = 0; i < k; i++){ 
            //如果不足k到了链表尾，直接返回，不翻转
            if(tail == NULL) 
                return head;
            tail = tail->next; 
        }
        //翻转时需要的前序和当前节点
        ListNode* pre = NULL; 
        ListNode* cur = head;
        //在到达当前段尾节点前
        while(cur != tail){ 
            //翻转
            ListNode* temp = cur->next; 
            cur->next = pre;
            pre = cur;
            cur = temp;
        }
        //当前尾指向下一段要翻转的链表
        head->next = reverseKGroup(tail, k); 
        return pre;
    }
};

~~~



## 数组

### 二分查找

左闭右开区间，故初始的right定义为size()，while条件判断为小于号，right赋值为middle

~~~C++
// 版本二
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size(); // 定义target在左闭右开的区间里，即：[left, right)
        while (left < right) { // 因为left == right的时候，在[left, right)是无效的空间，所以使用 <
            int middle = left + ((right - left) >> 1);
            if (nums[middle] > target) {
                right = middle; // target 在左区间，在[left, middle)中
            } else if (nums[middle] < target) {
                left = middle + 1; // target 在右区间，在[middle + 1, right)中
            } else { // nums[middle] == target
                return middle; // 数组中找到目标值，直接返回下标
            }
        }
        // 未找到目标值
        return -1;
    }
};
~~~



### 二维数组查找

左上与右下必定为最小值与最大值，而左下与右上就有规律了：左下元素大于它上方的元素，小于它右方的元素，右上元素与之相反。每次与左下角元素比较，我们就知道目标值应该在哪部分中。

- step 1：首先获取矩阵的两个边长，判断特殊情况。
- step 2：首先以左下角为起点，若是它小于目标元素，则往右移动去找大的，若是他大于目标元素，则往上移动去找小的。
- step 3：若是移动到了矩阵边界也没找到，说明矩阵中不存在目标值。





### 寻找峰值

类似二分法的方式，每次都往高的那一侧收缩，注意此处也是使用了左闭右开的方式来定义区间

~~~c++
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        int left = 0;
        int right = nums.size() - 1;
        //二分法
        while(left < right){ 
            int mid = (left + right) / 2;
            //右边是往下，不一定有坡峰
            if(nums[mid] > nums[mid + 1])
                right = mid;
            //右边是往上，一定能找到波峰
            else
                left = mid + 1;
        }
        //其中一个波峰
        return right; 
    }
};
~~~



### 逆序对问题

也即满足arr[i]>arr[j]且i<j的数对，称为逆序对

暴力搜索O(N^2)的时间复杂度

优化算法是使用归并排序的思想，在merge过程中顺道计算逆序对数目，举例如下：

如果区间有序，比如[3,4] 和 [1,2]，如果3 > 1, 显然3后面的所有数都是大于1， 这里为 4 > 1

因此算法就是在归并排序的基础上，增加了一步关于逆序对的数目的计算，注意ret的计算只有一个分支需要进行

~~~c++
class Solution {
private:
    const int kmod = 1000000007;
public:
    int InversePairs(vector<int> data) {
        int ret = 0;
        merge_sort__(data, 0, data.size() - 1, ret);
        return ret;
    }

    void merge_sort__(vector<int> &arr, int l, int r, int &ret) {
        if (l >= r) {
            return;
        }

        int mid = l + ((r - l) >> 1);
        merge_sort__(arr, l, mid, ret);
        merge_sort__(arr, mid + 1, r, ret);
        merge__(arr, l, mid, r, ret);
    }

    void merge__(vector<int> &arr, int l, int mid, int r, int &ret) {
        vector<int> tmp(r - l + 1);
        int i = l, j = mid + 1, k = 0;

        while (i <= mid && j <= r) {
            if (arr[i] > arr[j]) {
                tmp[k++] = arr[j++];
                // 奥妙之处
                ret += (mid - i + 1);
                ret %= kmod;
            }
            else {
                tmp[k++] = arr[i++];
            }
        }

        while (i <= mid) {
            tmp[k++] = arr[i++];
        }
        while (j <= r) {
            tmp[k++] = arr[j++];
        }

        for (k = 0, i = l; i <= r; ++i, ++k) {
            arr[i] = tmp[k];
        }
    }
};
~~~

这里可以多关注一下，归并排序的实现，第一部分是比较简单的，分开归并然后merge就好，主要是merge部分的实现



### 旋转数组求最小值

使用和二分查找类似的方式，分治法进行查找，只不过此时需要增加一种情况也即边界值等于中间值的时候该如何处理

标准解法给出的方式是，此时无法判断最小值在左边还是右边，因此可以将右边那个指针right往左挪一个位子，逐步尝试

~~~c++
class Solution {
public:
    int minNumberInRotateArray(vector<int> rotateArray) {
        int left = 0;
        int right = rotateArray.size() - 1;
        while(left < right){
            int mid = (left + right) / 2;
            //最小的数字在mid右边
            if(rotateArray[mid] > rotateArray[right]) 
                left = mid + 1;
            //无法判断，一个一个试
            else if(rotateArray[mid] == rotateArray[right]) 
                right--;
            //最小数字要么是mid要么在mid左边
            else 
                right = mid;
        }
        return rotateArray[left];
    }
};
~~~



## 字符串

### 大数加法

首先补全短的字符串，补到长度一致（使用string可以方便地使用加号拼接），注意末位的进位



### 反转字符串

![image-20230621154804770](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20230621154804770.png)

最后一个是左旋转字符串，也即将前n个，放到字符串的尾部

思路如下：

1. 反转区间为前n的子串
2. 反转区间为n到末尾的子串
3. 反转整个字符串

很神奇的思路



### 字符串匹配

#### KMP算法

主要分两步，利用模式串构建next数组，利用next数组遍历一遍文本串

next数组的含义：**模式串中前后字符重复出现的个数**

![image-20230623112625017](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20230623112625017.png)

构建next数组可以采用暴力算法，O(m^2)

计算跳过的趟数（模式串指针回退的数目）：已经匹配上的字符串数目-重复字符串的长度

其中重复字符串的长度=`next[j-1]`

可以看到，在这整个过程中，文本串的指针永不回退，只会向前，所以时间复杂度是O(n)

搜索遍历的函数如下

![img](https://pic1.zhimg.com/v2-2b32c73968b32b6b7a9e91c31eb580c0_r.jpg)





字符串匹配完整代码

~~~c++
class Solution {
public:
    int strStr(string haystack, string needle) {
        vector<int> next;
        next.resize(needle.size());
        next = getNext(needle);
        int p_hay=0, p_needle=0;
        while(p_hay<haystack.size()){
            if(haystack[p_hay]==needle[p_needle]){
                p_hay++;p_needle++;
            }
            else if(p_needle!=0){
                p_needle = next[p_needle-1];
            }
            else{
                p_hay++;
            }
            if (p_needle==needle.size()){
                return p_hay-p_needle;
            }
        }
        return -1;
    }

    vector<int> getNext(string str){
        vector<int> next;
        next.push_back(0);
        int i=1;
        while(i<str.size()){
            string::iterator front=str.begin(), end=str.begin()+i+1;
            int j;
            for(j=i;j>=1;j--){
                string string_1(front,front+j);
                string string_2(end-j,end);
                if (string_1==string_2){next.push_back(j);break;}
            }
            if(j==0) next.push_back(j);
            i++;
        }
        return next;
    }
};
~~~

此处计算next数组的方式采用的是暴力算法，针对每个i，从最长的情况开始列举（相同前后缀长度为i-1）



### 重复子串

判断一个给定的字符串s，是否可以由n个相同的子串构成

[讲解链接](https://programmercarl.com/0459.%E9%87%8D%E5%A4%8D%E7%9A%84%E5%AD%90%E5%AD%97%E7%AC%A6%E4%B8%B2.html#%E7%A7%BB%E5%8A%A8%E5%8C%B9%E9%85%8D)

思路就是，利用KMP算法中的最大相同前后缀，若是前缀和后缀不重合的部分，长度刚好可以被总长度整除，则可以说明这部分就是重复子串

判断方式：`len % (len-next[len-1])==0`，也即这里例子的`8%(8-6)==0`

<img src="https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20230623150544198.png" alt="image-20230623150544198" style="zoom:80%;" />





## 二叉树

### 二叉树遍历

#### 前中后序遍历

递归方法非常简单，这里写一下非递归的方法

都是使用栈进行实现，前序比较简单，中序即每个节点要出现过一次才能输出，后序即要出现过两次

前序

~~~c++
vector<int> preorderTraversal(TreeNode* root) {
    // write code here
    TreeNode* p = root;
    vector<int> result;
    stack<TreeNode*> st;
    st.push(p);
    while(!st.empty()){
        p = st.top();
        st.pop();
        while(p!=NULL){
            result.push_back(p->val);
            st.push(p->right);
            p = p->left;
        } 
    }
    return result;
}
~~~

中序

~~~c++
vector<int> inorderTraversal(TreeNode* root) {
    // write code here
    TreeNode* p = NULL;
    vector<int> result;
    stack<TreeNode*> st;
    map<TreeNode*, int> count_map;
    st.push(root);
    while(!st.empty() || p!=NULL){
        if(p!=NULL){
            if(count_map.find(p)==count_map.end()){
                count_map[p] = 1;
                st.push(p);
                p = p->left;
            }
            else if(count_map[p] == 1){
                result.push_back(p->val);
                p = p->right;
            }
        }
        else{
            p = st.top();
            st.pop();
        }
    }
    return result;
}
~~~

后序

~~~c++
vector<int> postorderTraversal(TreeNode* root) {
    // write code here
    TreeNode* p = NULL;
    vector<int> result;
    stack<TreeNode*> st;
    map<TreeNode*, int> count_map;
    st.push(root);
    while(!st.empty() || p!=NULL){
        if(p!=NULL){
            if(count_map.find(p)==count_map.end()){
                count_map[p] = 1;
                st.push(p);
                p = p->left;
            }
            else if(count_map[p] == 1){
                count_map[p] = 2;
                st.push(p);
                p = p->right;
            }
            else {
                result.push_back(p->val);
                p = NULL;
            }
        }
        else{
            p = st.top();
            st.pop();
        }
    }
    return result;
}
~~~

#### 层次遍历

直接使用一个队列即可



### 翻转二叉树

将整棵树镜像翻转，可以采用递归，每次将左右结点调换（注意递归顺序，只能是前后序遍历，不能是中序）

~~~c++
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root == NULL) return root;
        swap(root->left, root->right);  // 中
        invertTree(root->left);         // 左
        invertTree(root->right);        // 右
        return root;
    }
};
~~~



### 对称二叉树

判断一个树是不是对称的，采用类似上一题的做法，但这里需要比较的是两个子树是否是镜像的，可以每次比较左子树的左节点和右子树的右节点，以及左子树的右节点和右子树的左节点为根的子树

~~~c++
class Solution {
public:
    bool compare(TreeNode* left, TreeNode* right) {
        // 首先排除空节点的情况
        if (left == NULL && right != NULL) return false;
        else if (left != NULL && right == NULL) return false;
        else if (left == NULL && right == NULL) return true;
        // 排除了空节点，再排除数值不相同的情况
        else if (left->val != right->val) return false;

        // 此时就是：左右节点都不为空，且数值相同的情况
        // 此时才做递归，做下一层的判断
        bool outside = compare(left->left, right->right);   // 左子树：左、 右子树：右
        bool inside = compare(left->right, right->left);    // 左子树：右、 右子树：左
        bool isSame = outside && inside;                    // 左子树：中、 右子树：中 （逻辑处理）
        return isSame;

    }
    bool isSymmetric(TreeNode* root) {
        if (root == NULL) return true;
        return compare(root->left, root->right);
    }
};
~~~



### 最大、最小深度

最大深度就直接递归，（1+max（left，right ））即可

最小深度有点讲究，因为最小深度指的是从根节点到最近的叶子结点的距离，叶子结点需要左右都为空才可以，故需要将左为空右不为空，右为空左不为空的情况都单独拎出来

~~~c++
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (root == NULL) return 0;
        if (root->left == NULL && root->right != NULL) {
            return 1 + minDepth(root->right);
        }
        if (root->left != NULL && root->right == NULL) {
            return 1 + minDepth(root->left);
        }
        return 1 + min(minDepth(root->left), minDepth(root->right));
    }
};
~~~



### 完全二叉树的节点数目

可以使用一般树的方式来进行计算，也即递归然后每遇到一个节点就返回一个

~~~c++
class Solution {
public:
    int countNodes(TreeNode* root) {
        if (root == NULL) return 0;
        return 1 + countNodes(root->left) + countNodes(root->right);
    }
};
~~~



但是可以利用完全二叉树的特性，来进一步提高算法效率

若一个完全二叉树，向左遍历和向右遍历的深度一样，则其为满二叉树，可以直接使用$n=2^{depth}+1$进行计算

~~~c++
class Solution {
public:
    int countNodes(TreeNode* root) {
        if (root == nullptr) return 0;
        TreeNode* left = root->left;
        TreeNode* right = root->right;
        int leftDepth = 0, rightDepth = 0; // 这里初始为0是有目的的，为了下面求指数方便
        while (left) {  // 求左子树深度
            left = left->left;
            leftDepth++;
        }
        while (right) { // 求右子树深度
            right = right->right;
            rightDepth++;
        }
        if (leftDepth == rightDepth) {
            return (2 << leftDepth) - 1; // 注意(2<<1) 相当于2^2，所以leftDepth初始为0
        }
        return countNodes(root->left) + countNodes(root->right) + 1;
    }
};
~~~



### 判断是否平衡二叉树

也即判断一个二叉树，左右子树的高度差是否小于1

可以使用递归的方式，计算左右高度并求差，以-1代表已经不是平衡树了，无需进一步计算

~~~c++
class Solution {
public:
    // 返回以该节点为根节点的二叉树的高度，如果不是平衡二叉树了则返回-1
    int getHeight(TreeNode* node) {
        if (node == NULL) {
            return 0;
        }
        int leftHeight = getHeight(node->left);
        if (leftHeight == -1) return -1;
        int rightHeight = getHeight(node->right);
        if (rightHeight == -1) return -1;
        return abs(leftHeight - rightHeight) > 1 ? -1 : 1 + max(leftHeight, rightHeight);
    }
    bool isBalanced(TreeNode* root) {
        return getHeight(root) == -1 ? false : true;
    }
};
~~~



### 二叉树的所有路径

利用了一点回溯法的思想，注意回溯和递归是一个相反的过程

~~~c++
class Solution {
private:

    void traversal(TreeNode* cur, string path, vector<string>& result) {
        path += to_string(cur->val); // 中
        if (cur->left == NULL && cur->right == NULL) {
            result.push_back(path);
            return;
        }
        if (cur->left) traversal(cur->left, path + "->", result); // 左
        if (cur->right) traversal(cur->right, path + "->", result); // 右
    }

public:
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<string> result;
        string path;
        if (root == NULL) return result;
        traversal(root, path, result);
        return result;

    }
};
~~~



### 路径之和

求二叉树中，是否存在一条到达叶子结点的路径，使得其值之和等于一个给定值

一个关键技巧是，不要对路径和进行求和，判断是否等于目标值，而是应该在递归时，将目标值减去当前节点值作为参数传入下一层递归

~~~c++
class Solution {

public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if(root==nullptr)return false;
        if(root->left==nullptr && root->right==nullptr && root->val==targetSum) return true;

        bool left=hasPathSum(root->left,targetSum-root->val);
        bool right=hasPathSum(root->right,targetSum-root->val);
        return left || right;
    }
};
~~~





### 使用中序和后序遍历结果，恢复一棵二叉树

其思路是，后序遍历数组的最后一个元素，即为当前层的根节点，利用该节点值，将中序遍历数组切分为两部分（左子树和右子树），然后在根据这个结果，将后续遍历数组也进行切分（数目与中序遍历数组对应），得到左右子树的中序数组和后序数组，即可进行递归

~~~c++
class Solution {
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        if(postorder.size()==0)return nullptr;
        TreeNode* root =new TreeNode(*(postorder.end()-1));
        //中序切分
        vector<int>::iterator it;
        for(it=inorder.begin();it!=inorder.end();it++){
            if(*it==root->val)break;
        }
        vector<int> left_inorder(inorder.begin(),it);
        vector<int> right_inorder(it+1,inorder.end());

        //后序切分
        vector<int> left_postorder(postorder.begin(),postorder.begin()+left_inorder.size());
        vector<int> right_postorder(postorder.begin()+left_inorder.size(),postorder.end()-1);

        root->left = buildTree(left_inorder,left_postorder);
        root->right = buildTree(right_inorder,right_postorder);

        return root;
    }
};
~~~





### 验证二叉搜索树

注意一个要点：二叉搜索树的定义是，左子树的所有节点都小于根节点，右子树的所有节点都大于根节点，因此不能在每一层单纯的判断左边小于中间小于右边

其解决的思路是，对二叉搜索树进行中序遍历，得到的是一个升序的数组（并且二叉搜索树中的元素不能有一样的）

因此下面采用递归的方式进行中序遍历，并且使用一个pre指针，来指定遍历顺序中的前一个节点的值（前一个节点值必须小于当前节点）

~~~c++
class Solution {
public:
    TreeNode* pre = NULL; // 用来记录前一个节点
    bool isValidBST(TreeNode* root) {
        if (root == NULL) return true;
        bool left = isValidBST(root->left);

        if (pre != NULL && pre->val >= root->val) return false;
        pre = root; // 记录前一个节点

        bool right = isValidBST(root->right);
        return left && right;
    }
};
~~~



### 二叉搜索树的最小绝对差

要找二叉搜索树中节点间差值的绝对值的最小值，思路依然是使用中序遍历得到递增序列，然后求相邻值的差即可，可以**像上一题一样使用一个pre指针保留前一个节点**

~~~c++ 
class Solution {
public:
    TreeNode* pre=nullptr;
    int min_difference = 1e5;
    int getMinimumDifference(TreeNode* root) {
        getMin(root);
        return min_difference;
    }

    void getMin(TreeNode* root){
        if(root==nullptr) return;
        getMin(root->left);
        if(pre!=nullptr) min_difference=min((root->val-pre->val),min_difference);
        pre=root;
        getMin(root->right);
        return;
    }
};
~~~



### 二叉搜索树的众数

依然类似上一题，可以采用中序遍历，然后用pre指针记录前一个节点，同时使用一个result的vector记录当前出现过最多次的元素，若是出现新的max_count的元素，则清空原vector，重新开始

~~~c++
class Solution {
private:
    TreeNode* pre=nullptr;
    int count = 0;
    int max_count = 0;
    vector<int> result;
    void traverse(TreeNode* root){
        if(root==nullptr)return;
        traverse(root->left);
        if(pre!=nullptr){
            if(pre->val==root->val)count++;
            else count=1;
        }
        else{
            count++;
        }
        if(count==max_count) result.push_back(root->val);
        if(count>max_count){
            max_count = count;
            result.clear();
            result.push_back(root->val);
        }
        pre = root;
        traverse(root->right);
        return;
    }
public:
    vector<int> findMode(TreeNode* root) {
        traverse(root);
        return result;
    }
};
~~~



### 二叉树的最近公共祖先

递归返回值为找到的最近公共祖先节点指针（当未遍历到两个点时，返回的是其中已经找到的那个节点指针）

~~~c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root==NULL) return NULL;
        if(root->val==p->val || root->val==q->val) return root;

        TreeNode* left = lowestCommonAncestor(root->left,p,q);
        TreeNode* right = lowestCommonAncestor(root->right,p,q);
        if(left&&right) return root;
        if(left&&!right)return left;
        else if(!left&&right)return right;
        else return NULL;
    }
};
~~~



### 二叉搜索树的最近公共祖先

相比于二叉树，二叉搜索树中寻找公共祖先非常简单，只需要找到第一个位于[p,q]范围间的节点即可

~~~c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root==NULL) return root;
        if(p->val<root->val&&q->val<root->val) return lowestCommonAncestor(root->left,p,q);
        if(p->val>root->val&&q->val>root->val) return lowestCommonAncestor(root->right,p,q);
        if((p->val>root->val&&q->val<root->val) || (p->val<root->val&&q->val>root->val)) return root;
        return root;
    }
};
~~~



### 二叉搜索树插入节点

很简单的一个题目，直接找到最低的的空节点，然后添加新节点插入即可

这里有一个比较有意思的实现：最低点的时候构建节点然后返回，在上一层的时候用`root->left`或`root->right`接住

~~~c++
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if (root == NULL) {
            TreeNode* node = new TreeNode(val);
            return node;
        }
        if (root->val > val) root->left = insertIntoBST(root->left, val);
        if (root->val < val) root->right = insertIntoBST(root->right, val);
        return root;
    }
};
~~~



### 二叉搜索树删除节点

删除节点相对比较复杂一点，分为五种情况

- 第一种情况：没找到删除的节点，遍历到空节点直接返回了
- 找到删除的节点
  - 第二种情况：左右孩子都为空（叶子节点），直接删除节点， 返回NULL为根节点
  - 第三种情况：删除节点的左孩子为空，右孩子不为空，删除节点，右孩子补位，返回右孩子为根节点
  - 第四种情况：删除节点的右孩子为空，左孩子不为空，删除节点，左孩子补位，返回左孩子为根节点
  - 第五种情况：左右孩子节点都不为空，**则将删除节点的左子树头结点（左孩子）放到删除节点的右子树的最左面节点的左孩子上，返回删除节点右孩子为新的根节点**。

注意第五种情况的解决方法



~~~c++
class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        if (root == nullptr) return root; // 第一种情况：没找到删除的节点，遍历到空节点直接返回了
        if (root->val == key) {
            // 第二种情况：左右孩子都为空（叶子节点），直接删除节点， 返回NULL为根节点
            if (root->left == nullptr && root->right == nullptr) {
                ///! 内存释放
                delete root;
                return nullptr;
            }
            // 第三种情况：其左孩子为空，右孩子不为空，删除节点，右孩子补位 ，返回右孩子为根节点
            else if (root->left == nullptr) {
                auto retNode = root->right;
                ///! 内存释放
                delete root;
                return retNode;
            }
            // 第四种情况：其右孩子为空，左孩子不为空，删除节点，左孩子补位，返回左孩子为根节点
            else if (root->right == nullptr) {
                auto retNode = root->left;
                ///! 内存释放
                delete root;
                return retNode;
            }
            // 第五种情况：左右孩子节点都不为空，则将删除节点的左子树放到删除节点的右子树的最左面节点的左孩子的位置
            // 并返回删除节点右孩子为新的根节点。
            else {
                TreeNode* cur = root->right; // 找右子树最左面的节点
                while(cur->left != nullptr) {
                    cur = cur->left;
                }
                cur->left = root->left; // 把要删除的节点（root）左子树放在cur的左孩子的位置
                TreeNode* tmp = root;   // 把root节点保存一下，下面来删除
                root = root->right;     // 返回旧root的右孩子作为新root
                delete tmp;             // 释放节点内存（这里不写也可以，但C++最好手动释放一下吧）
                return root;
            }
        }
        if (root->val > key) root->left = deleteNode(root->left, key);
        if (root->val < key) root->right = deleteNode(root->right, key);
        return root;
    }
};
~~~



### 修剪二叉搜索树

将二叉搜索树中的所有节点值，修剪到一个给定范围中，超过该范围的节点全都删掉

其思路比较简单，就是采用递归的方式，遇到不符合要求的节点，若其值过大，就将其左子树中的可用头结点用作替换，若其值过小，就将其右子树中的可用头结点用作替换

~~~c++
class Solution {
public:
    TreeNode* trimBST(TreeNode* root, int low, int high) {
        if (root == nullptr) return nullptr;
        if (root->val < low) return trimBST(root->right, low, high);
        if (root->val > high) return trimBST(root->left, low, high);
        root->left = trimBST(root->left, low, high);
        root->right = trimBST(root->right, low, high);
        return root;
    }
};
~~~



### 构造平衡二叉树

没啥好说的，就是不断取中点构造新节点，然后递归进行

~~~c++
class Solution {
private:
    TreeNode* traversal(vector<int>& nums, int left, int right) {
        if (left > right) return nullptr;
        int mid = left + ((right - left) / 2);
        TreeNode* root = new TreeNode(nums[mid]);
        root->left = traversal(nums, left, mid - 1);
        root->right = traversal(nums, mid + 1, right);
        return root;
    }
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        TreeNode* root = traversal(nums, 0, nums.size() - 1);
        return root;
    }
};
~~~





## 动态规划

动规五部曲：

1. 确定dp数组（dp table）以及下标的含义
2. 确定递推公式
3. dp数组如何初始化
4. 确定遍历顺序
5. 举例推导dp数组



### 爬楼梯、最小代价爬楼梯

根据每次可以往上爬的阶梯数，确定递推公式，最小代价爬楼梯的代码如下，本质上就是个斐波那契数

~~~c++
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        vector<int> dp(cost.size()+1);
        dp[0]=0;dp[1]=0;
        for(int i=2;i<cost.size()+1;i++){
            dp[i]=min(dp[i-1]+cost[i-1],dp[i-2]+cost[i-2]);
        }
        return dp[cost.size()];
    }
};
~~~



### 不同路径

带障碍物的m*n网格，走到终点总共有多少种方式

```cpp
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int m=obstacleGrid.size(),n=obstacleGrid[0].size();
        vector<vector<int>> dp(m,vector<int>(n));
        bool flag=0;
        for(int i=0;i<m;i++) {
            if(obstacleGrid[i][0])flag=1;
            if(!flag)dp[i][0]=1;
            else dp[i][0]=0;
        }
        flag=0;
        for(int i=0;i<n;i++) {
            if(obstacleGrid[0][i])flag=1;
            if(!flag)dp[0][i]=1;
            else dp[0][i]=0;
        }

        for(int i=1;i<m;i++){
            for(int j=1;j<n;j++){
                if(obstacleGrid[i][j]) dp[i][j]=0;
                else dp[i][j]=dp[i-1][j]+dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
};
```



### 整数拆分

将一个大于2的整数，拆分为k个正整数，要求其乘积最大，求这个最大乘积

针对每个i，可以先求比它小的数的拆分乘积情况，然后遍历所有比它小的数j，将i分解为j和另一部分，注意此时另一部分即可以是一个数，也可以是两个数，而我们这里dp数组的定义是拆分后的最大乘积值，也即默认将其视为拆分了，因此递推公式不能简单写为$max(dp[i],j\times dp[i-j])$，此时第二项默认拆分为三个数及以上，因此需要再加入一项，写为$max(dp[i],max(j\times (i-j),j\times dp[i-j]))$，考虑了拆分为两个数的情况

~~~cpp
class Solution {
public:
    int integerBreak(int n) {
        vector<int> dp(n+1,0);
        dp[1] = 1;
        for(int i=2;i<n+1;i++){
            for(int j=1;j<=i/2;j++){
                dp[i] = max(dp[i],max(j*dp[i-j],j*(i-j)));
            }
        }
        return dp[n];
    }
};
~~~



### 单词拆分

给定一个字符串s，给定一个字典vector\<string\>，代表可选用的单词，要检验s是否可以拆分为字典中单词的组合

可以从头开始遍历，不断拆分出子字符串，有如下递推式

$dp[i]=OR_j [dp[j]\ and\ (substr[j:i]\in Dict)]$，这里OR代表对所有项求或

~~~c++
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> dict(wordDict.begin(),wordDict.end());
        vector<bool> dp(s.size()+1,false);
        dp[0] = true;
        for(int j=0;j<=s.size();j++){
            for(int i=0;i<j;i++){
                string sub = s.substr(i,j-i);
                if(dict.find(sub)!=dict.end() && dp[i]==true)
                    dp[j]=true;   
            }
        }
        return dp[s.size()];
    }
};
~~~





### 不同的二叉搜索树

#### 不同二叉搜索树的数目

给定一个正整数n，计算以1…n为节点的二叉搜索树的数目，其思路为将该二叉树拆分为左中右三部分，中间为头节点，左右子树的数目可以由之前的计算来得到，因此递推公式为：

$dp[i]=\sum_j(dp[j]*dp[i-j-1])$

~~~c++
class Solution {
public:
    int numTrees(int n) {
        vector<int> dp(n+1);
        dp[0] = 1;
        for(int i=1;i<=n;i++){
            for(int j=0;j<i;j++)
                dp[i]+=dp[j]*dp[i-j-1];
        }
        return dp[n];
    }
};
~~~

#### 不同二叉搜索树

相比上一题，这里要求返回的不是数目，而是具体的树的根节点组成的数组

使用动态规划的方式，使用一个map，将(l,r)之间的元素构成的树的所有可能情况都存起来，然后若已经搜索过，则直接从map中返回即可

~~~c++
class Solution {
public:
    map<pair<int,int>,vector<TreeNode*>> m;
    vector<TreeNode*> generateTrees(int n) {
        if(m.find({1,n})!=m.end()) return m[{1,n}];
        backtrack(1,n);
        return m[{1,n}];
    }
    vector<TreeNode*> backtrack(int l, int r)
    {
        if(l>r) return {nullptr};
        if(m.find({l,r})!=m.end()) return m[{l,r}];
        vector<TreeNode*> res;
        for(int i = l; i <= r; i++)
        {
            vector<TreeNode*> left_tree = backtrack(l, i-1);
            vector<TreeNode*> right_tree = backtrack(i+1, r);
            for(TreeNode* left:left_tree)
                for(TreeNode* right:right_tree)
                    res.emplace_back(new TreeNode(i,left,right));
        }
         return m[{l,r}] = res;
    }
};
~~~



### 背包问题

#### 0-1背包

注意其dp数组的含义，dp\[i]\[j]代表在0-i个物品，背包容量为j时的最大价值，此时递推公式写为

$dp[i][j]=max(dp[i-1][j],dp[i-1][j-weight[i]]+value[i])$

也即不加入第i个物品，和加入第i个物品两种情况（加入第i个物品，要求此时背包容量要大于i的质量



写为一维dp数组的方式：

$dp[j]=max(dp[j],dp[j-weight[i]]+value[i])$

当物品的价值都为正，dp数组初始化为0即可，若存在负价值，则需要将其值初始化为INT_MIN

注意其遍历顺序，i正常从0开始遍历，j需要按背包容量从大到小逆序遍历

~~~c++
for(int i = 0; i < weight.size(); i++) { // 遍历物品
    for(int j = bagWeight; j >= weight[i]; j--) { // 遍历背包容量
        dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
    }
}
~~~



#### 分割等和子集

判断能否将一个包含正整数元素的数组，分为两个子集，使得两个子集的元素和相等

转化为：一个容量为target（target=元素和的一半）的背包，能否恰好被装满

背包容量即为所有元素的和的一半，元素的质量和价值都为元素的值

~~~c++
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int target=accumulate(nums.begin(),nums.end(),0);
        if(target%2!=0 || nums.size()<=1)return false;
        target = target/2;
        vector<int> dp(target+1,0);
        for(int i=0;i<nums.size();i++){
            for(int j=target;j>=nums[i];j--){
                dp[j] = max(dp[j],dp[j-nums[i]]+nums[i]);
            }
        }
        if(dp[target]==target)return true;
        else return false;
    }
};
~~~



#### 最后一块石头重量

所有的石头互相碰撞粉碎，剩下的石头质量为其差值，求最终留下的石头的最小可能重量

类似上一题分割等和子集，只不过此时需要额外计算一下，最终遗留下来的石头质量

$weight=sum-2\times dp[target]$



#### 目标和

给定一个非负整数数组，将每个元素赋以正号或负号，求使得其最终的和为一个给定值的赋值方式的数目

相比之前，本题的改变在于，要求所有满足要求的赋值方式的数目，而非求是否有满足要求的赋值方式，也即要求变高，变难了

对于动态规划求组合数目的题目，都可以将dp数组设计为：dp[j]表示和为j时的组合方法数，则有

$dp[j]=\sum_j dp[j-nums[i]]$，意义是比较明确的

注意初始化时，需要**将dp[0]设置为1**，否则递推后所有值都会为零**（所有求组合排列数的动态规划题都需要这样子）**

本题中，要使其最终和为一个给定值，则$x=(sum+target)/2$，背包容量可以设置为这个

总的代码如下

~~~c++
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int S) {
        int sum = 0;
        for (int i = 0; i < nums.size(); i++) sum += nums[i];
        if (abs(S) > sum) return 0; // 此时没有方案
        if ((S + sum) % 2 == 1) return 0; // 此时没有方案
        int bagSize = (S + sum) / 2;
        vector<int> dp(bagSize + 1, 0);
        dp[0] = 1;
        for (int i = 0; i < nums.size(); i++) {
            for (int j = bagSize; j >= nums[i]; j--) {
                dp[j] += dp[j - nums[i]];
            }
        }
        return dp[bagSize];
    }
};
~~~



#### 一和零

对于一个二进制字符串数组，要求其最大子集的大小，使得该子集中最多有 m 个 0 和 n 个 1 。

这其实就是个最基本的01背包问题，只不过此时物品的质量有了两个维度：0的数目和1的数目而已，物品的价值即为1（个数）

依旧是遍历所有物品，然后将dp数组设置为二维，分别逆向遍历计算即可

~~~c++
class Solution {
public:
    int findMaxForm(vector<string>& strs, int m, int n) {
        vector<vector<int>> dp(m + 1, vector<int> (n + 1, 0)); // 默认初始化0
        for (string str : strs) { // 遍历物品
            int oneNum = 0, zeroNum = 0;
            for (char c : str) {
                if (c == '0') zeroNum++;
                else oneNum++;
            }
            for (int i = m; i >= zeroNum; i--) { // 遍历背包容量且从后向前遍历！
                for (int j = n; j >= oneNum; j--) {
                    dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1);
                }
            }
        }
        return dp[m][n];
    }
};
~~~

其中i和j的遍历顺序是无所谓的



#### 完全背包

完全背包问题和01背包的区别在于，物品的数量是否是无限的，解法方面，只需要将背包容量遍历的循环，改为从前往后正序遍历即可

~~~c++
// 先遍历物品，再遍历背包
for(int i = 0; i < weight.size(); i++) { // 遍历物品
    for(int j = weight[i]; j <= bagWeight ; j++) { // 遍历背包容量
        dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
    }
}
~~~

此时也可以改为背包容量在外，物品在内的遍历，若只是计算最大价值，则结果是一样的，不过那样的写法需要增加一个下标的判断

~~~c++
// 先遍历背包，再遍历物品
for(int j = 0; j <= bagWeight; j++) { // 遍历背包容量
    for(int i = 0; i < weight.size(); i++) { // 遍历物品
        if (j - weight[i] >= 0) dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
    }
}
~~~



#### 零钱兑换

给定一个数值和一系列硬币，求有多少种组合方式

**完全背包的组合问题**

组合问题，外层循环为物品，内层为容量

而求组合方式的数目，则需要用到前面讲到的那个递推式：$dp[j]=\sum_i dp[j-[nums[i]]]$

~~~c++
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<int> dp(amount+1,0);
        dp[0]=1;
        for(int i=0;i<coins.size();i++){
            for(int j=coins[i];j<=amount;j++){
                dp[j] += dp[j-coins[i]];
            }
        }
        return dp[amount];
    }
};
~~~





零钱兑换的另一种题目：

给定不同面额的硬币 coins 和一个总金额 amount，要计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

注意现在不是要求符合要求的组合数了，而是要找到最小的硬币数

之前都是求最大，此处改为求最小，其实原理是一样的，只不过在初始化的时候需要修改一下，改为INT_MAX（而不是0），同时，初始化时dp[0]应该初始化为0

~~~c++
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        vector<int> dp(amount+1,INT_MAX);
        dp[0] = 0;
        for(int i=0;i<coins.size();i++){
            for(int j=coins[i];j<=amount;j++){
                if(dp[j-coins[i]]!=INT_MAX)
                    dp[j] = min(dp[j],dp[j-coins[i]]+1);
            }
        }
        return dp[amount]==INT_MAX?-1:dp[amount];
    }
};
~~~

注意这种求min的，由于将初始值赋值为INT_MAX，**因此在内层需要增加一个条件判断**，防止溢出







#### 组合总和 Ⅳ

给定一个正整数组成的且不含重复元素的数组，找出其和为给定值的排列数

**完全背包的排列问题**

排列问题，外层循环为容量，内层为物品，依旧是使用上一个递推式，不过需要额外对下标做一个判断

~~~c++
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        vector<int> dp(target+1,0);
        dp[0]=1;
        for(int j=0;j<=target;j++){
            for(int i=0;i<nums.size();i++){
                if(j-nums[i]>=0 && dp[j]<INT_MAX-dp[j-nums[i]]) dp[j] += dp[j-nums[i]];
            }
        }
        return dp[target];
    }
};
~~~

注意看 ，此处只是调换了循环的内外，但是最里层依旧是dp[j]的处理



### 打家劫舍

小偷要偷家，每次不能偷相邻的两个房子，问最大的偷取金额数

#### version1

最基础的版本，所有的房子排成一排

一个基础的动态规划问题，递推式为$dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])$

~~~c++
class Solution {
public:
    int rob(vector<int>& nums) {
        if(nums.size()==0)return 0;
        if(nums.size()==1)return nums[0];
        vector<int> dp(nums.size(),0);
        dp[0] = nums[0];
        dp[1] = max(nums[0],nums[1]);
        for(int i=2;i<nums.size();i++){
            dp[i] = max(dp[i-1],dp[i-2]+nums[i]);
        }
        return dp[nums.size()-1];
    }
};
~~~



#### version2

房子不再是连成一排，而是围成一圈，所以首尾不能同时偷，此时可以分解为两个子问题，去掉头和去掉尾，然后使用version1的方法分别求解，取较大的那个值即可



#### version3

房子组织成树的形状了，是一个树形dp问题，可以采用递归的方式来做，每个节点只有两个状态（偷或不偷），因此返回值可以设置为一个长度为2的数组，其中dp[0]代表不偷该节点的最大值，dp[1]表示偷该节点的最大值

~~~c++
class Solution {
    vector<int> rob_(TreeNode* root){
        if(root==NULL) return {0,0};
        
        vector<int> left=rob_(root->left);
        vector<int> right=rob_(root->right);
        int val1=max(left[0],left[1])+max(right[0],right[1]);
        int val2=root->val+left[0]+right[0];
        return {val1,val2};
    }
public:
    int rob(TreeNode* root) {
        vector<int> result=rob_(root);
        return max(result[0],result[1]);
    }
};
~~~





### 股票问题

股票问题的解决思路，就是利用一个二维dp数组（其中第二维的维数等于状态数）来代表不同状态下，该天最大的收益，以此达到遍历所有情况的目的

状态数取决于具体题目的设定，后一天的状态取决于前一天的状态

#### version1

只能买卖一次

状态：

+ 持有股票（0）
+ 不持有股票（1）

因此dp数组第二维的长度为2，递推公式如下

+ $dp[i][0]=max(dp[i-1][0],-prices[i])$，注意不是$dp[i][0]=max(dp[i-1][0],dp[i-1][1]-prices[i])$，若是使用第二种写法，则代表可以进行多次买卖
+ $dp[i][1]=max(dp[i-1][1],dp[i-1][0]+prices[i])$，这个是比较显然的，就等于前一天不持有股票或前一天持有但今天卖出这两种情况中的最大值

~~~c++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        vector<vector<int>> dp(prices.size(),vector<int>(2,0));
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        for(int i=1;i<prices.size();i++){
            dp[i][0] = max(dp[i-1][0],-prices[i]);
            dp[i][1] = max(dp[i-1][1],dp[i-1][0]+prices[i]);
        }
        return dp[prices.size()-1][1];
    }
};
~~~



#### version2

允许多次买卖，此时与version1几乎完全相同，只需要将$dp[i][0]$的递推式修改为$dp[i][0]=max(dp[i-1][0],dp[i-1][1]-prices[i])$即可



#### version3

限制最多只能完成两笔交易

此时问题就变得复杂了，需要在version1的基础上，修改状态如下

+ 未操作

+ 第一次持有股票
+ 第一次卖出股票
+ 第二次持有股票
+ 第二次卖出股票

递推公式：

+ $dp[i][0]=dp[i-1][0]$

+ $dp[i][1]=max(dp[i-1][1],dp[i-1][0]-prices[i])$
+ $dp[i][2]=max(dp[i-1][2],dp[i-1][1]+prices[i])$

+ $dp[i][3]=max(dp[i-1][3],dp[i-1][2]-prices[i])$
+ $dp[i][4]=max(dp[i-1][4],dp[i-1][3]+prices[i])$

初始化为：$dp[0][0]=0$，$dp[0][1]=-prices[0]$，$dp[0][2]=0$，$dp[0][3]=-prices[0]$，$dp[0][4]=0$

代码与version1中类似



#### version4

最多完成k笔交易，此时只需要在version3的基础上，修改代码使其第二维维度为2k+1即可

~~~c++
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        vector<vector<int>> dp(prices.size(),vector<int>(2*k+1,0));
        for(int i=1;i<2*k+1;i+=2) dp[0][i]=-prices[0];
        for(int i=1;i<prices.size();i++){
            dp[i][0] = dp[i-1][0];
            for(int j=1;j<2*k+1;j+=2){
                dp[i][j] = max(dp[i-1][j],dp[i-1][j-1]-prices[i]);
                dp[i][j+1] = max(dp[i-1][j+1],dp[i-1][j]+prices[i]);
            }
        }
        return dp[prices.size()-1][2*k];
    }
};
~~~



#### version5

此时增加了冷冻期的限制，前一天卖出了股票，无法在当天或第二天买入股票

需要在version2的基础上，增加关于冷冻期的状态

+ 持有股票（0）
+ 不持有股票但不是冷冻期（1）
+ 当天卖出股票（2）
+ 处于冷冻期（3）

递推公式

+ $dp[i][0]=max(dp[i-1][0],dp[i-1][1]-prices[i],dp[i-1][3]-prices[i])$
+ $dp[i][1]=max(dp[i-1][1],dp[i-1][3])$
+ $dp[i][2]=dp[i-1][0]+prices[i]$
+ $dp[i][3]=dp[i-1][2]$

初始化需要格外注意，$dp[0][0]=-prices[0]$是比较自然的，但是另外三个状态的初值则需要通过第二天的递推计算式来获得一个合理的值，其结果为$dp[0][1]=0$，$dp[0][2]=0$，$dp[0][3]=0$



#### version6

可以无限次买卖，但是每笔买卖都需要支付一个固定的手续费

只需要在version2的基础上，于卖出时增加一个手续费即可



### 子序列问题

子序列指的是，原数组删掉中间的某些元素后组成的序列，并不需要在原数组中连续

#### 最长上升子序列

定义dp数组：$dp[i]$表示**i之前包括i的以nums[i]结尾**的最长递增子序列的长度

递推公式：if (nums[i] > nums[j]) dp[i] = max(dp[i], dp[j] + 1);

其中j的范围从0到i-1，也即在i之前的所有元素结尾的子序列长度基础上，若是nums[i]大于该结尾元素nums[j]

，则说明最长子序列长度又增加了1

**初始化：所有位置结尾子序列的最小长度都应该初始化为1**

~~~c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        vector<int> dp(nums.size(),1);
        int max_len=1;
        for(int i=1;i<nums.size();i++){
            for(int j=0;j<i;j++){
                if(nums[i]>nums[j]) dp[i] = max(dp[i],dp[j]+1);
            }
            max_len = dp[i]>max_len?dp[i]:max_len;
        }
        return max_len;
    }
};
~~~



#### 最长重复子数组

两个数组中，要找一个最长的公共的子数组，求其长度

定义dp数组：$dp[i][j]$代表数组A以A[i]结尾的子数组，以及数组B以B[j]结尾的子数组中的最长重复子数组的长度

那么就有，$dp[i][j]=dp[i-1][j-1]+1\quad \text{if A[i]==B[j]}$

初始化方面，由于递推式中出现了i-1和j-1，因此必须要把i=0和j=0的情况进行初始化，也即A第一位和B中每一位的重合情况，以及B第一位和A中每一位的重合情况，若存在这种情况，**记得将result置为1**

只需要遍历AB数组即可，注意从1开始遍历

~~~c++
class Solution {
public:
    int findLength(vector<int>& nums1, vector<int>& nums2) {
        vector<vector<int>> dp(nums1.size(),vector<int>(nums2.size(),0));
        int result=0;
        for(int i=0;i<nums1.size();i++)if(nums1[i]==nums2[0]){dp[i][0]=1;result=1;}
        for(int j=0;j<nums2.size();j++)if(nums1[0]==nums2[j]){dp[0][j]=1;result=1;}
        for(int i=1;i<nums1.size();i++){
            for(int j=1;j<nums2.size();j++){
                if(nums1[i]==nums2[j]) dp[i][j]=dp[i-1][j-1]+1;
                result = dp[i][j]>result?dp[i][j]:result;
            }
        }
        return result;
    }
};
~~~



#### 最长公共子序列

跟上一题相比，变为不要求连续了，只要是子序列就好，子序列问题中，dp数组的每一项代表的是该项之前的数组的最长子序列长度

故在这里，定义如下：$dp[i][j]$表示以A[i]和B[j]结尾的数组的最长公共子序列长度，那么递推公式即

$\begin{aligned} dp[i][j]=\begin{cases} dp[i-1][j-1]+1 \quad &\text{,if A[i]==B[j]} \\ max(dp[i-1][j],dp[i][j-1]) &\text{,else} \end{cases}\end{aligned}$

实现类似上一题

注意这里的初始化方式，**只要出现过一样的，后面就全部置为1了**（因为是子序列）

~~~c++
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        if(text1.size()==0 || text2.size()==0)return 0;
        vector<vector<int>> dp(text1.size(),vector<int>(text2.size(),0));
        bool flag=false;
        for(int i=0;i<text1.size();i++)if(flag||text1[i]==text2[0]){dp[i][0]=1;flag=true;}
        flag=false;
        for(int j=0;j<text2.size();j++)if(flag||text1[0]==text2[j]){dp[0][j]=1;flag=true;} 
        for(int i=1;i<text1.size();i++){
            for(int j=1;j<text2.size();j++){
                if(text1[i]==text2[j]) dp[i][j]=dp[i-1][j-1]+1;
                else dp[i][j]=max(dp[i-1][j],dp[i][j-1]);
            }
        }
        return dp[text1.size()-1][text2.size()-1];        
    }
};
~~~



#### 最大子序和

其实应该叫最大子数组和，就是要求一个数组的连续子数组的最大和

依然是类似前面的dp数组的定义：$dp[i]$代表以A[i]结尾的数组中的最大子序和，因此递推式：

$dp[i]=max(dp[i-1]+nums[i],nums[i])$，即要么在前面的基础上加上当前元素值，要么就从头开始

~~~c++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        vector<int> dp(nums.size(),0);
        dp[0]=nums[0];
        int result=dp[0];
        for(int i=1;i<nums.size();i++){
            dp[i]=max(dp[i-1]+nums[i],nums[i]);
            result =dp[i]>result?dp[i]:result;
        }
        return result;
    }
};
~~~



#### 判断子序列

判断一个序列s是否是另一个序列t的子序列，最简单的思路应该就是，计算两个序列的最长公共子序列，然后判断其长度是否和s相等



#### 不同的子序列数目

给定一个序列s和t，判断s的子序列中有几个等于t

dp数组定义为$dp[i][j]$表示以s[i]和t[j]结尾的数组中，前者的子序列中等于后者的数目（子问题）

两种情况

+ 当s[i]==t[j]时，有两种情况，一是使用s[i]和t[j]，此时应当等于$dp[i-1][j-1]$，**也有另一种情况，即不使用s[i]，只使用t[j]，此时应当等于$dp[i-1][j]$**
+ 当s[i]!=t[j]时，只有不使用s[i]的情况，即等于$dp[i-1][j]$

~~~c++
class Solution {
public:
    int numDistinct(string s, string t) {
        vector<vector<double>> dp(s.size(),vector<double>(t.size(),0));
        if(s.size()<t.size())return 0;
        int count=0;
        for(int i=0;i<s.size();i++){
            if(s[i]==t[0])count++;
            dp[i][0] = count;
        }
        for(int i=1;i<s.size();i++){
            for(int j=1;j<t.size();j++){
                if(s[i]==t[j]) dp[i][j]=dp[i-1][j-1]+dp[i-1][j];
                else dp[i][j] = dp[i-1][j];
            }
        }
        return dp[s.size()-1][t.size()-1];
    }
};
~~~



#### 两个字符串的删除

计算两个字符串，需要删除多少步才能变成两个相同的字符串

只需要计算其最长公共子序列，然后分别用两个字符串的长度减掉最长公共子序列的长度，相加即可



#### 编辑距离

编辑距离指的是，两个单词，通过多少步增删替换，可以成为一样的单词

+ 当s[i]==t[j]时，此时该位已经一样了，不需要修改，应当等于$dp[i-1][j-1]$
+ 当s[i]!=t[j]时，有三种可能操作：
  + 删掉s[i]，此时等于$dp[i-1][j]+1$
  + 删掉t[j]，此时等于$dp[i][j-1]+1$
  + 修改一个元素，此时等于$dp[i-1][j-1]+1$

初始化方面，牢记定义的话，则$dp[i][0]$表示子数组和t[0]之间的距离，若存在t[0]应该初始化为i+1，否则初始化为i

$dp[0][j]$同理

~~~c++
class Solution {
public:
    int minDistance(string word1, string word2) {
        vector<vector<double>> dp(word1.size(),vector<double>(word2.size(),0));
        if(word1.size()==0)return word2.size();
        if(word2.size()==0)return word1.size();
        bool flag=false;
        for(int i=0;i<word1.size();i++){
            if(word1[i]==word2[0])flag=true;
            dp[i][0]=flag?i:i+1;
        }
        flag=false;
        for(int j=0;j<word2.size();j++){
            if(word2[j]==word1[0])flag=true;
            dp[0][j]=flag?j:j+1;
        }
        for(int i=1;i<word1.size();i++){
            for(int j=1;j<word2.size();j++){
                if(word1[i]==word2[j]) dp[i][j]=dp[i-1][j-1];
                else dp[i][j]=min({dp[i][j-1],dp[i-1][j],dp[i-1][j-1]})+1;
            }
        }
        return dp[word1.size()-1][word2.size()-1];
    }
};
~~~



### 回文

#### 回文子串

计算一个给定字符串中的回文子串的数目

一个字符本身可视作回文的，不同起始位置的相同子串也视作不同的子串

设置一个二维的dp数组，表示从i开始到j结束的字符串是否是回文的

递推公式：

+ s[i]!=s[j]，此时$dp[i][j]=false$
+ s[i]==s[j]，分两种情况：
  + j-i<=1，即长度为1或2，此时肯定是回文
  + otherwise，此时首尾已经是相等了，是否是回文取决于内部是否是回文，即$dp[i+1][j-1]$

初始化非常简单，全部都初始化为false即可

**遍历顺序**非常重要，由于计算的时候会用到$dp[i+1][j-1]$，因此i+1必须在i之前被计算，即外层的i需要逆序计算，而内层的j只需要正序即可

~~~c++
class Solution {
public:
    int countSubstrings(string s) {
        vector<vector<bool>> dp(s.size(),vector<bool>(s.size(),false));
        int result = 0;
        for(int i=s.size()-1;i>=0;i--){
            for(int j=i;j<s.size();j++){
                if(s[i]!=s[j])dp[i][j]=false;
                else if(j-i<=1) dp[i][j]=true;
                else dp[i][j] = dp[i+1][j-1];
                if(dp[i][j])result++;
            }
        }
        return result;
    }
};
~~~



#### 最长回文子串

可以将dp数组定义为i到j之间的子串为回文子串的长度（不是的话记为0），递推式只需要在上一题的基础上稍作修改即可

~~~c++
class Solution {
public:
    string longestPalindrome(string s) {
        vector<vector<int>> dp(s.size(),vector<int>(s.size(),0));
        int max_dp=0,max_i=0,max_j=0;
        for(int i=s.size()-1;i>=0;i--){
            for(int j=i;j<s.size();j++){
                if(s[i]!=s[j]) dp[i][j]=0;
                else if(j-i==0) dp[i][j]=1;
                else if(j-i==1) dp[i][j]=2;
                else dp[i][j]=dp[i+1][j-1]?dp[i+1][j-1]+2:0;

                if(dp[i][j]>max_dp){
                    max_dp=dp[i][j];
                    max_i=i;max_j=j;
                }
            }
        }
        string result=s.substr(max_i,max_j-max_i+1);
        return result;
    }
};
~~~



#### 最长回文子序列

在最长回文子串的基础上进行修改，此时若$s[i]!=s[j]$，则可以将s分别左右收缩一下，取其最大值为$dp[i][j]$

~~~c++
class Solution {
public:
    int longestPalindromeSubseq(string s) {
        vector<vector<int>> dp(s.size(),vector<int>(s.size(),0));
        int max_len=0;
        for(int i=s.size()-1;i>=0;i--){
            for(int j=i;j<s.size();j++){
                if(s[i]!=s[j]) dp[i][j]=max(dp[i+1][j],dp[i][j-1]);
                else {
                    if(j-i<=1) dp[i][j]=j-i+1;
                    else dp[i][j]=dp[i+1][j-1]+2;
                }
                max_len = dp[i][j]>max_len? dp[i][j]:max_len;
            }
        }
        return max_len;
    }
};
~~~



#### 回文总结

使用动态规划的方式，$dp[i][j]$代表i和j之间的子串子序列的长度/是否回文，不需要特别的初始化，初始化包含在了循环内j=i的情况中了，注意循环遍历的方式，外层i逆序遍历，内层j正序遍历



### 动态规划 练习

#### 删除并获得点数

给你一个整数数组 `nums` ，你可以对它进行一些操作。

每次操作中，选择任意一个 `nums[i]` ，删除它并获得 `nums[i]` 的点数。之后，你必须删除 **所有** 等于 `nums[i] - 1` 和 `nums[i] + 1` 的元素。

开始你拥有 `0` 个点数。返回你能通过这些操作获得的最大点数。



看到删除nums[i]-1和nums[i]+1，就该想到打家劫舍问题，实际实现中，这里可能存在多个相同的nums[i]，可以遍历后记录其数目，并将其合成为一个值为 n*val 的节点



#### 最大正方形

找到0-1矩阵中的最大全1正方形，定义$dp[i][j]$为以$matrix[i][j]$为右下点的正方形的最大边长，有如下递推式

$dp(i, j) = min(dp(i - 1, j), dp(i, j - 1), dp(i - 1, j - 1))+1\quad \text{if matrix[i][j]=='1'}$

![image-20230627135038345](C:\Users\15989845233\AppData\Roaming\Typora\typora-user-images\image-20230627135038345.png)



#### 两个字符串的最小删除和

计算两个字符串，成为相同字符串所需删除元素的最小ASCII值之和

~~~c++
class Solution {
public:
    int minimumDeleteSum(string s1, string s2) {
        vector<vector<int>> dp(s1.size()+1,vector<int>(s2.size()+1,INT_MAX));
        dp[0][0]=0;
        for(int i=1;i<=s1.size();i++) dp[i][0] = dp[i-1][0]+s1[i-1];
        for(int j=1;j<=s2.size();j++) dp[0][j] = dp[0][j-1]+s2[j-1];

        for(int i=1;i<=s1.size();i++){
            for(int j=1;j<=s2.size();j++){
                if(s1[i-1]==s2[j-1])dp[i][j]=dp[i-1][j-1];
                else dp[i][j] = min(dp[i-1][j]+s1[i-1],dp[i][j-1]+s2[j-1]);
            }
        }
        return dp[s1.size()][s2.size()];
    }
};
~~~



#### 最长上升子序列的数目

！！！！！！

有非常多细节需要处理，时刻需要记住在最长子序列问题中，dp[i]的含义指的是以s[i]为结尾的子序列的长度（而不是i之前的子序列的长度，即必须包含i）

~~~c++
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        vector<int> dp(nums.size(),1);
        vector<int> cnt(nums.size(),1);
        int max_len=1;
        for(int i=1;i<nums.size();i++){
            for(int j=0;j<i;j++) {
                if(nums[i]>nums[j]){
                    if(dp[j]+1>dp[i]){
                        dp[i]=dp[j]+1;
                        cnt[i]=cnt[j];
                    }
                    else if(dp[j]+1==dp[i]) {
                        cnt[i]+=cnt[j];
                    }
                }
                max_len = (dp[i]>max_len)?dp[i]:max_len;
            }
        }
        int max_count=0;
        for(int i=0;i<nums.size();i++) if(dp[i]==max_len)max_count+=cnt[i];
        return max_count;
    }
};
~~~





#### 最长数对链长度

比较元素从一个数字，变为一个数对（pair），其他完全一样，只不过需要提前对数对进行排序

注意，vector是可以进行排序的，其从头按元素大小比较，先出现较小元素的vector判为小

~~~c++
class Solution {
public:
    int findLongestChain(vector<vector<int>>& pairs) {
        sort(pairs.begin(),pairs.end());
        vector<int> dp(pairs.size(),1);
        int max_len=1;
        for(int i=1;i<pairs.size();i++){
            for(int j=0;j<i;j++) if(pairs[j][1]<pairs[i][0]) dp[i] = max(dp[i],dp[j]+1);
            max_len=(dp[i]>max_len)?dp[i]:max_len;
        }
        return max_len;
    }
};
~~~



#### 最长定差子序列

其实就是最长递增子序列的修改版本，此时不仅要求递增了，而且限制更强，要求其差值为固定值difference，因此可以简单修改代码如下

~~~c++
class Solution {
public:
    int longestSubsequence(vector<int>& arr, int difference) {
        vector<int> dp(arr.size(),1);
        int max_len=1;
        for(int i=1;i<dp.size();i++){
            for(int j=0;j<i;j++){
                if(arr[i]-arr[j]==difference)dp[i]=max(dp[i],dp[j]+1);
                max_len = (dp[i]>max_len)?dp[i]:max_len;
            }
        }
        return max_len;
    }
};
~~~

时间复杂度为O(n方），在LeetCode上超时

利用其差值固定的信息，可以使用一个map来记录dp信息，其中dp[v]表示以元素v结尾的子序列的长度

**这里其实是哈希表的写法，以后遇上等差、两数之和、两数之差这样的字眼都需要想到哈希表**

~~~c++
class Solution {
public:
    int longestSubsequence(vector<int> &arr, int difference) {
        int ans = 0;
        unordered_map<int, int> dp;
        for (int v: arr) {
            dp[v] = dp[v - difference] + 1;
            ans = max(ans, dp[v]);
        }
        return ans;
    }
};
~~~



#### 最长等差数列

利用类似最长递增子序列的思路，不过此时不同的公差需要分开考虑，因此可以将dp数组定义为一个二维数组，其中第二个维度是不同的公差

记录不同公差的数据时，比较直观的是使用一个map，但是实际操作中超时了，所以后续改为一个vector，然后遍历得到min max，偏移到下标的范围中

~~~c++
class Solution {
public:
    int longestArithSeqLength(vector<int>& nums) {
        auto minmax = minmax_element(nums.begin(),nums.end());
        int min_val=*minmax.first, max_val=*minmax.second;
        int diff=max_val-min_val;
        vector<vector<int>> dp(nums.size(),vector<int>(diff*2+1,0));
        int max_len=1;
        for(int i=1;i<nums.size();i++){
            for(int j=0;j<i;j++){
                dp[i][nums[i]-nums[j]+diff] = max(dp[i][nums[i]-nums[j]+diff],dp[j][nums[i]-nums[j]+diff] + 1);
                max_len=(dp[i][nums[i]-nums[j]+diff]>max_len)?dp[i][nums[i]-nums[j]+diff]:max_len;
            }
        }
        return max_len+1;
    }
};
~~~



#### 最大递增子序列问题的更优解法

采用贪心+二分法的方式，可以将时间复杂度压缩到O(nlogn)

具体方式为，维护一个数组dp，其中dp[i]代表长度为i的所有最长递增子序列中的末尾元素的最小值，可以证明dp[i]数组是严格递增的

依次遍历nums中的每个元素，若nums[i]>dp[end]，说明找到了一个更长的子序列；否则在dp数组中进行二分查找，找到第一个比nums[i]小的数dp[k]，并将dp[k+1]更新为nums[i]

二分查找时，可以使用标准库中的`lower_bound(begin,end,val)`函数，会进行二分查找，并返回第一个不小于val的元素的迭代器（若是不要求严格递增，则使用`upper_bound(begin,end,val)`函数）

代码如下

~~~c++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        vector<int> dp;
        for(int i=0;i<nums.size();i++){
            auto loc_it = lower_bound(dp.begin(),dp.end(),nums[i]);
            if(loc_it==dp.end()){ //表示不存在比它还大的元素，插入
                dp.push_back(nums[i]);
            }
            else{
                int loc = loc_it-dp.begin();
                dp[loc] = nums[i];
            }
        }
        return dp.size();
    }
};
~~~



#### 俄罗斯套娃信封问题

其实就是定义了一个数对的排列方式，必须要a[0]>b[0]，a[1]>b[1]时a才大于b，然后求最长递增子序列的长度，此时完全可以直接对数据进行排序，然后套用求递增子序列的方式即可

个别用例会超时

此处设计了一种排序方式，先**对首元素升序，再对第二个元素降序**，这样仅需要在第二个元素上求最长递增子序列即可

~~~c++
bool cmp(vector<int> a, vector<int> b){
    return a[0]<b[0] || (a[0]==b[0]&&a[1]>b[1]);
}

class Solution {
public:
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        sort(envelopes.begin(),envelopes.end(),cmp);
        vector<int> dp(envelopes.size(),1);
        for(int i=1;i<envelopes.size();i++){
            for(int j=0;j<i;j++){
                // if(envelopes[i][0]>envelopes[j][0] && envelopes[i][1]>envelopes[j][1])
                if(envelopes[i][1]>envelopes[j][1])
                    dp[i] = max(dp[i],dp[j]+1);
            }
        }
        return *max_element(dp.begin(),dp.end());
    }
};
~~~

可以使用动态规划+二分查找的方式

~~~c++
bool cmp(vector<int> a, vector<int> b){
    return a[0]<b[0] || (a[0]==b[0]&&a[1]>b[1]);
}

class Solution {
public:
    int maxEnvelopes(vector<vector<int>>& envelopes) {
        if (envelopes.empty()) {
            return 0;
        }
        sort(envelopes.begin(),envelopes.end(),cmp);
        vector<int> dp;
        for(int i=0;i<envelopes.size();i++){
            if(dp.empty()||envelopes[i][1]>dp.back()) dp.push_back(envelopes[i][1]);
            else {
                int loc = lower_bound(dp.begin(),dp.end(),envelopes[i][1])-dp.begin();
                dp[loc] = envelopes[i][1];
            }
        }
        return dp.size();
    }
};
~~~





#### 找出到每个位置为止最长的有效障碍赛跑路线

其实就是从求严格递增子序列，变为求递增子序列，并且要求返回每个i对应的子序列长度

注意使用动规+二分时，若没有构成更长的子序列，则每个i对应的子序列长度为**loc+1**

~~~c++
class Solution {
public:
    vector<int> longestObstacleCourseAtEachPosition(vector<int>& obstacles) {
        vector<int> dp;
        vector<int> ans;
        if(obstacles.size()==0)return {0};
        dp.push_back(obstacles[0]);
        ans.push_back(1);
        for(int i=1;i<obstacles.size();i++){
            if(obstacles[i]>=dp.back()){
                dp.push_back(obstacles[i]);
                ans.push_back(dp.size());
            } 
            else {
                int loc = upper_bound(dp.begin(),dp.end(),obstacles[i])-dp.begin();
                dp[loc] = obstacles[i];
                ans.push_back(loc+1);
            }
        }
        return ans;
    }
};
~~~





#### 让字符串成为回文串的最少插入次数

计算将一个字符串变成回文串所需要 的最少的插入次数，将问题转化为求其最长回文子序列的长度，然后只需要对于所有不在子序列中的字符，根据回文中心，在对称的位置插入一个相同的字符，即可消除该字符带来的非回文影响，即**需要插入的字符数为s的长度减掉最长回文子序列的长度**

~~~c++
class Solution {
public:
    int minInsertions(string s) {
        vector<vector<int>> dp(s.size(),vector<int>(s.size(),1));
        int result = 0;
        for(int i=s.size()-1;i>=0;i--){
            for(int j=i;j<s.size();j++){
                if(s[i]!=s[j]) dp[i][j] = max(dp[i+1][j],dp[i][j-1]);
                else{
                    if(j-i==0) dp[i][j] = 1;
                    else if(j-i==1) dp[i][j] = 2;
                    else dp[i][j] = dp[i+1][j-1] + 2;
                }
                result = (dp[i][j]>result)?dp[i][j]:result;
            }
        }
        return s.size()-result;
    }
};
~~~





#### 二叉树的最大路径和

通过一个节点及其左右孩子的路径和，可以表示为val+left_gain+right_gain，而left_gain和right_gain是可以递归计算的，所以在本题里，主要的递归函数是用来计算节点的gain，然后不断比较以该节点为中心（通过它和左右孩子）的路径和与当前最大路径和，若找到一条更大的则更新它

~~~c++
class Solution {
private:
    int maxSum = INT_MIN;

public:
    int maxGain(TreeNode* node) {
        if (node == nullptr) {
            return 0;
        }
        
        // 递归计算左右子节点的最大贡献值
        // 只有在最大贡献值大于 0 时，才会选取对应子节点
        int leftGain = max(maxGain(node->left), 0);
        int rightGain = max(maxGain(node->right), 0);

        int priceNewpath = node->val + leftGain + rightGain;
        maxSum = max(maxSum, priceNewpath);

        return node->val + max(leftGain, rightGain);
    }

    int maxPathSum(TreeNode* root) {
        maxGain(root);
        return maxSum;
    }
};
~~~







### 动态规划总结

#### 背包问题

1、常规问题，求背包中最大价值

dp数组定义为**(bag_size+1,0)**，dp[j]表示背包容量为j时的最大价值

0-1背包问题中，遍历顺序为：i从物品0开始遍历，j从bagsize开始逆序遍历到item[i]，递推式：

$dp[j] = max(dp[j],dp[j-weight[i]]+value[i])$

完全背包问题中，只是修改了遍历顺序，内部也采用正序遍历

2、组合、排列问题，求总价值满足某个条件的组合、排列数目

dp数组定义为**(bag_size+1,0)，dp[0]=1**，dp[j]表示背包容量为j时的组合、排列数

递推式：

$dp[j] = \sum_j dp[j-nums[i]]$

组合、排列的区别在于遍历方式，若是组合则先遍历物品，若是求排列则先遍历容量



#### 打家劫舍

dp数组定义为**(nums.size(),0)**，dp[i]表示下标i之前的房子序列的最大金额，初始化前两个房子，然后递推式为：

$dp[i] = max(dp[i-1],dp[i-2]+nums[i])$



当组织为树的形状时，可以定义返回值为一个长度为2的列表，分别代表偷或不偷的结果，然后采用后续遍历的方式遍历整个树，即可得到结果



#### 股票问题

dp数组维度定义为**(prices.size(),state_num)**，其中state_num为状态数，状态根据题目要求而定，其思路就是从头开始遍历，每天的状态值仅由前一天和当前的选择来决定，取max



#### 子序列问题

##### 最长递增子序列的长度

可以直接使用动规+二分查找的方式来做，dp数组的长度即为结果



##### 最长递增子序列的数目

设置一个额外的数组cnt，代表以s[i]结尾的最长子序列的数目，然后依旧使用最长子序列的算法，不同的是针对每个s[i]，**若是得到了一个更长的子序列，就将cnt重新置为1，若是得到了一个一样长的就加一**，最终取d[i]为最大值的cnt[i]之和即为结果



##### 最长等差数列

dp数组额外设置一个维度为公差大小，然后依旧是常规的最长子序列算法，只不过**不同的公差分别计算序列长度**而已，最终遍历得到最大值即为结果



##### 最长公共子数组、子序列问题

$dp[i][j]$代表序列1的i位之前，和序列2的j位之前的长度，分别从头遍历即可

递推公式

+ 子数组：$dp[i][j]=dp[i-1][j-1]+1\quad \text{if A[i]==B[j]}$
+ 子序列：$\begin{aligned} dp[i][j]=\begin{cases} dp[i-1][j-1]+1 \quad &\text{,if A[i]==B[j]} \\ max(dp[i-1][j],dp[i][j-1]) &\text{,else} \end{cases}\end{aligned}$

需要注意一下初始化





#### 回文

dp数组表示，从i开始到j结束的字符串是否是回文的，不需要特别的初始化，初始化包含在了循环内j=i的情况中了，注意循环遍历的方式，**外层i逆序遍历，内层j正序遍历**

s[i]和s[j]相同时，注意长度为1和为2的情况都是回文的



成为回文串最小插入数目：计算最长回文子序列







## SJTU

### 二次方程计算器

~~~c++
#include <iostream>
#include <bits/stdc++.h>
using namespace std;

bool is_num(char& c){
    return (c>='0'&&c<='9');
}

bool is_symbol(char& c){
    return (c=='='||c=='^'||c=='+'||c=='-');
}


int get_num(string& s, int idx){
    int result = 0;
    for(int i=idx; is_num(s[i])&&s[i]!='\0';i++){
        result+= result*10 + (s[i]-'0');
    }
    return result;
}

int main() {
    string equation;
    getline(cin,equation);
    int a=0,b=0,c=0;
    bool front_flag=true,symbol_flag=true;
    if(equation[0]=='x') equation="1"+equation;
    for(int i=0;i<equation.size();i++){
        if(is_num(equation[i])){
            int val = get_num(equation, i);
            val = (front_flag^symbol_flag)? -1*val: val;
            while(equation[i]!='\0' && is_num(equation[i])) i++;
            if(equation[i]=='x'){
                if(equation[i+1]=='^'){
                    a += val;
                    i+=2;
                }
                else{
                    b += val;
                }
            }
            else c+=val;
        }
        else if(is_symbol(equation[i])){
            if(equation[i]=='+') symbol_flag=true;
            else if(equation[i]=='-') symbol_flag=false;
            else if(equation[i]=='=') {front_flag=false;symbol_flag=true;}
        }
        else {
            int val = (front_flag^symbol_flag)? -1: 1;
            if(equation[i+1]=='^'){
                a+=val;
                i+=2;
            }
            else b+=val;
        }
    }
    
    double delta = pow(b,2)-4*a*c;
    if(delta<0){
        cout<<"No Solution";
    }
    else if(delta==0){
        printf("%.2f",-1.*b/(2.*a));
    }
    else{
        printf("%.2f %.2f",(-1.*b-sqrt(delta))/(2.*a),(-1.*b+sqrt(delta))/(2.*a));
    }
}
~~~





### Old Bills

~~~c++
#include <iostream>
#include <bits/stdc++.h>
using namespace std;

int get_num(string& str){
    int result=0;
    for(auto s:str){
        result = result*10 +(s-'0');
    }
    return result;
}

int main() {
    string tmp;
    getline(cin,tmp);
    while(true){
        if(tmp=="") break;
        int num = get_num(tmp);
        int a = 0;
        int value =0;
        for(int i=0;i<3;i++){
            cin>>a;
            value = value*10 +a;
        }
        value*=10;
        int total_value = 0;
        bool flag=false;
        for(int i=9;i>0;i--){
            for(int j=9;j>=0;j--){
                total_value = value + i*10000 + j;
                if(total_value%num==0){
                    flag = true;
                    printf("%d %d %d\n",i,j,total_value/num);
                    break;
                }
            }
            if(flag) break;
        }
        if(!flag) cout<<0<<endl;
        cin.get();
        getline(cin,tmp);
    }

    return 0;
}
~~~

这里主要是一个，关于多行字符串输入的方法

~~~c++ 
string tmp;
getline(cin,tmp);
while(true){
	if(tmp=="")break;
	/* code */
	cin.get();
	getline(cin,tmp);
}
~~~



### 棋盘问题

dfs进行搜索，注意dfs搜索时，要记录关于某个结点是否被搜索了的状态，开始搜索该节点时记录为true，搜索完后记录为False

~~~c++
#include <bits/stdc++.h>
#include <iostream>
using namespace std;

int sx,sy,ex,ey;
bool st[6][6];
int keyboard[6][6];
int res=1e8;
int x_op[4] = {0,0,-1,1},y_op[4]={1,-1,0,0};

void dfs(int x,int y,int state,int cost){
    if(x==ex&&y==ey){
        if(cost<res)res=cost;
        return;
    }
    if(cost>=res)return;
    st[x][y] = true;
    int c=0;
    for(int i=0;i<4;i++){
        int nx=x+x_op[i], ny=y+y_op[i];
        if(nx>=6||nx<0||ny>=6||ny<0||st[nx][ny]) continue;
        c=state*keyboard[nx][ny];
        dfs(nx,ny,(c%4)+1,cost+c);
        st[nx][ny]=false;
    }
}


int main(){
    for(int i=0;i<6;i++)
        for(int j=0;j<6;j++) cin>>keyboard[i][j];
    cin>>sx>>sy>>ex>>ey;
    memset(st,false,sizeof st);
    dfs(sx,sy,1,0);
    cout<<res<<endl;
    return 0;
}
~~~





### 打印路径

关键在于要对读进来的路径进行排序，同时确定每个路径分别有多少属于上一个路径（即确定初始需要打印的起点），可以通过计算每个路径与前一个路径的开头有多少重合来确定

~~~c++
#include <bits/stdc++.h>
using namespace std;

vector<vector<string>> read_paths(int n){
    string tmp;
    vector<vector<string>> paths(n,vector<string>());
    for(int i=0;i<n;i++){
        cin>>tmp;
        istringstream is(tmp);
        while(getline(is,tmp,'\\')){
            if(tmp!="")
                paths[i].push_back(tmp);
        }
    }
    return paths;
}

vector<int> diff(int n, vector<vector<string>>& paths){
    vector<int> diff_vec(n,0);
    if(n==1) return diff_vec;
    for(int i=1;i<n;i++){
        int j=0;
        for(;j<paths[i].size()&&j<paths[i-1].size();j++){
            if(paths[i][j]==paths[i-1][j]) continue;
            else break;
        }
        diff_vec[i]=j;
    }
    return diff_vec;
}

int main() {
    int n;
    cin>>n;
    cin.ignore();
    while (n!=0) {
        vector<vector<string>> paths=read_paths(n);
        sort(paths.begin(), paths.end());
        vector<int> diff_vec = diff(n,paths);
        for(int i=0;i<n;i++){
            int layer=diff_vec[i];
            for(int j=layer;j<paths[i].size();j++){
                string start(layer*2,' ');
                string out = start+paths[i][j];
                cout<<out<<endl;
                layer++;
            }
        }
        cout<<endl;
        cin>>n;
    }
    return 0;
}
~~~



### powerful calculator

加法：补齐长度，大的在前小的在后，逆序后逐位相加

减法：类似加法，逆序后逐位相减，结果需要去除前置0

乘法：逆序，先计算每一位的结果（不管进位，先存着），然后再统一处理进位，结果需要去除前置0

~~~c++
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>
#include <vector>
using namespace std;

bool judge(vector<int> a,vector<int> b){
    if(a.size()>b.size())return true;
    else if(a.size()<b.size()) return false;
    else {
        for(int i=0;i<a.size();i++){
            if(a[i]>b[i]) return true;
            else if(a[i]<b[i]) return false;
        } 
    }
    return false;
}

vector<int> add(vector<int> a,vector<int> b){
    reverse(a.begin(),a.end());
    reverse(b.begin(),b.end());
    for(int i=b.size();i<a.size();i++)b.push_back(0);
    assert(a.size()==b.size());
    vector<int> result;

    int carry=0,val=0;
    int i=0;
    for(;i<a.size();i++){
        val=a[i]+b[i]+carry;
        carry=val/10;
        val=val%10;
        result.push_back(val);
    }
    if(carry) result.push_back(1);
    reverse(result.begin(),result.end());
    return result;
}

vector<int> sub(vector<int> a,vector<int> b){
    reverse(a.begin(),a.end());
    reverse(b.begin(),b.end());
    for(int i=b.size();i<a.size();i++)b.push_back(0);
    assert(a.size()==b.size());
    vector<int> result;

    int carry=0,val=0;
    int i=0;
    for(;i<a.size();i++){
        val=a[i]-b[i]+carry;
        if(val<0) carry=-1; else carry=0;
        val=(val+10)%10;
        result.push_back(val);
    }
    for(int j=result.size()-1;result[j]==0&&j!=0;j--)result.pop_back();
    reverse(result.begin(),result.end());
    return result;
}

vector<int> mul(vector<int> a,vector<int> b){
    reverse(a.begin(),a.end());
    reverse(b.begin(),b.end());
    vector<int> result(a.size()+b.size()+1,0);
    for(int i=0;i<a.size();i++){
        for(int j=0;j<b.size();j++){
            result[i+j] += a[i]*b[j];
        }
    }
    for(int i=0;i<a.size()+b.size();i++){
        if(result[i]>10){
            result[i+1] += result[i]/10;
            result[i] = result[i]%10;
        }
    }

    for(int j=result.size()-1;result[j]==0&&j!=0;j--)result.pop_back();

    reverse(result.begin(),result.end());
    return result;
}

void print_result(vector<int> r, bool minus_flag){
    if(minus_flag) cout<<'-';
    for(auto a:r) cout<<a;
    cout<<endl;
}

vector<int> trans(string str){
    vector<int> vec;
    for(auto s:str)vec.push_back(s-'0');
    return vec;
}

int main() {
    string a_str,b_str;
    bool a_flag=true,b_flag=true;
    while(cin>>a_str>>b_str){
        if(a_str[0]=='-') {a_flag=false;a_str=a_str.substr(1);}
        if(b_str[0]=='-') {b_flag=false;b_str=b_str.substr(1);}
        vector<int> a=trans(a_str),b=trans(b_str);

        vector<int> add_result,sub_result,mul_result;


        bool add_flag=false,sub_flag=false,mul_flag=false;
        if(a_flag&&b_flag){
            add_result = (judge(a, b))?add(a,b):add(b,a);
            if(judge(b, a))sub_flag=true;
            sub_result = (judge(a, b))?sub(a,b):sub(b,a);
            mul_result = (judge(a, b))?mul(a,b):mul(b,a);
        }
        else if(!a_flag&&b_flag){
            add_result = (judge(a, b))?sub(a,b):sub(b,a);
            if(judge(a,b)) add_flag=true;
            sub_result = (judge(a, b))?add(a,b):add(b,a);
            sub_flag=true;
            mul_result = (judge(a, b))?mul(a,b):mul(b,a);
            mul_flag=true;
        }
        else if(a_flag&&!b_flag){
            add_result = (judge(a, b))?sub(a,b):sub(b,a);
            if(judge(b,a)) add_flag=true;
            sub_result = (judge(a, b))?add(a,b):add(b,a);
            sub_flag=false;
            mul_result = (judge(a, b))?mul(a,b):mul(b,a);
            mul_flag=true;
        }
        else {
            add_result = (judge(a, b))?sub(a,b):sub(b,a);
            add_flag=true;
            sub_result = (judge(a, b))?add(a,b):add(b,a);
            if(judge(a,b)) sub_flag=true;
            sub_flag=false;
            mul_result = (judge(a, b))?mul(a,b):mul(b,a);
            mul_flag=false;               
        }

        print_result(add_result,add_flag);
        print_result(sub_result,sub_flag);
        print_result(mul_result,mul_flag);
    }
    return 0;
}
~~~



### 计算表达式

表达式不含括号

#### 思路1

思路首先将所有的数据和符号分别取出来，存到两个列表中，注意数据的类型应当为double

然后遍历两次，第一次进行所有的乘除运算，同时更新数据和符号的列表，第二次从头遍历进行加减运算即可

~~~c++
#include <iostream>
#include <vector>
#include <bits/stdc++.h>
using namespace std;

bool is_num(char c){
    return (c>='0'&&c<='9');
}

int main() {
    string equation;
    while(cin>>equation){
        // preprocess
        vector<double> value;
        vector<int> op;
        for(int i=0;i<equation.size();i++){
            if(is_num(equation[i])){
                int begin=i,val=0;
                while(is_num(equation[i]))i++;
                val = atoi(equation.substr(begin,i-begin).c_str());
                value.push_back(val);
                i--;
            }
            else{
                if(equation[i]=='+')op.push_back(1);
                else if(equation[i]=='-')op.push_back(2);
                else if(equation[i]=='*')op.push_back(3);
                else if(equation[i]=='/')op.push_back(4);
            }
        }
        assert(value.size()==op.size()+1);

        //first_iter
        int i=0;
        while(i<op.size()){
            if(op[i]<=2){i++;continue;}
            double val = 0;
            if(op[i]==3)val=value[i]*value[i+1];
            else val=value[i]/value[i+1];

            op.erase(op.begin()+i);
            value.erase(value.begin()+i,value.begin()+i+2);
            value.insert(value.begin()+i,val);
            i=0;
        }

        // second_iter
        double result=value[0];
        for(int i=0;i<op.size();i++){
            if(op[i]==1) result+=value[i+1];
            else result-=value[i+1];
        }
        cout<<result<<endl;
    }
}
~~~

#### 思路2

注意符号优先级的定义：

栈中初始的压栈符号，必须是最低优先级，字符串的终止符号，优先级第二，加减符号优先级第三，乘除优先级第四

当遇到**当前优先级等于大于符号栈中最高优先级符号时，则继续压栈**；否则提出符号栈和数据栈中的元素进行运算，并从当前位置继续开始判断

~~~c++
#include <iostream>
#include <vector>
#include <bits/stdc++.h>
using namespace std;

bool is_num(char c){
    return (c>='0'&&c<='9');
}

int level(char c){
    switch (c) {
        case '#': return 0;
        case '$': return -1;
        case '+':
        case '-': return 1;
        case '*':
        case '/': return 2;
    }
    return 3;
}

double calculate(double a, double b, char c){
    switch (c) {
        case '+': return a+b;
        case '-': return a-b;
        case '*': return a*b;
        case '/': return a/b;
    }
    return 0;
}

int main() {
    string equation;
    while(cin>>equation){
        // preprocess
        vector<double> value;
        vector<char> op;
        for(int i=0;i<equation.size();i++){
            if(is_num(equation[i])){
                int begin=i,val=0;
                while(is_num(equation[i]))i++;
                val = atoi(equation.substr(begin,i-begin).c_str());
                value.push_back(val);
                i--;
            }
            else{
                if(equation[i]=='+')op.push_back('+');
                else if(equation[i]=='-')op.push_back('-');
                else if(equation[i]=='*')op.push_back('*');
                else if(equation[i]=='/')op.push_back('/');
            }
        }
        assert(value.size()==op.size()+1);

        stack<double> num_stack;
        stack<char> op_stack;
        op_stack.push('$');
        op.push_back('#');
        
        for(int i=0;i<value.size()+op.size();i++){
            // num
            if(!(i%2)){
                num_stack.push(value[i/2]);
            }
            //op
            else{
                if(level(op_stack.top())<level(op[i/2])) op_stack.push(op[i/2]);
                else {
                    char ch = op_stack.top();
                    op_stack.pop();
                    double a = num_stack.top();
                    num_stack.pop();
                    double b = num_stack.top();
                    num_stack.pop();
                    double result = calculate(b,a,ch);
                    num_stack.push(result);
                    i--;
                }
            }
        }
        cout<<num_stack.top()<<endl;
    }
    return 0;
}
~~~





### 2的幂次方

注意这里，2()代表的是2的n次方的意思

由于递归，因此可以使用一个dict保存已经出现过的结果

这道题的关键是，如何求一个数字的二进制表示（即分解为二次幂之和）

~~~c++
#include <iostream>
#include <unordered_map>
using namespace std;

unordered_map<int, string> dict;

string get_binary(int n){
    string binary;
    while(n){
        binary += (n%2 +'0');
        n/=2;
    }
    return binary;
}

void pow2(int n){
    if(dict.find(n)!=dict.end()){
        cout<<dict[n];return;
    }

    string binary = get_binary(n);
    
    int index;
    while ((index=binary.rfind('1'))!=string::npos) {
        if(index>2){
            cout<<"2(";
            pow2(index);
            binary[index]='0';
            cout<<")";
        }
        else{
            pow2(index);
            binary[index]='0';
        }
        if((index=binary.rfind('1'))!=string::npos) cout<<"+";
    }
}

int main() {
    dict[0] = "2(0)";
    dict[1] = "2";
    dict[2] = "2(2)";
    int a;
    cin>>a;
    if(a==0) {cout<<"0";return 0;}
    pow2(a);
}
~~~



### pre-post

求以pre和post作为前后序遍历的n叉树的可能数目

需要计算组合数，可以利用组合数的递推式，采用动态规划的方法计算

每一层中，总的组合数目等于n叉树取子树数目的组合，乘以子树的子树的组合数，逐层递归往上

~~~c++
#include <iostream>
#include <bits/stdc++.h>
#include <vector>
using namespace std;

vector<vector<int>> dp;

void cal_dp(int n){
    dp = vector<vector<int>>(n+1,vector<int>(n+1,1));
    for(int i=2;i<=n;i++){
        for(int j=1;j<i;j++){
            dp[i][j] = dp[i-1][j] + dp[i-1][j-1];
        }
        dp[i][i] = 1;
    }
}

int cal_tree(string pre, string post, int n){
    pre.erase(pre.begin());
    post.pop_back();
    int i=0;
    int sum=1;
    int num=0;
    while(i<pre.size()){
        for(int j=0;j<post.size();j++){
            if(pre[i]==post[j]){
                sum *= cal_tree(pre.substr(i,j-i+1), post.substr(i,j-i+1),n);
                num++;
                i = j + 1;
            }
        }
    }
    return sum * dp[n][num];
}

int main() {
    int n;
    while (cin >> n) {
        string pre,post;
        cin>>pre>>post;
        cal_dp(n);
        cout<<cal_tree(pre,post,n)<<endl;        
    }
    return 0;
}
~~~



### 整除问题

给定n，a求最大的k，使n！可以被a\^k整除但不能被a\^(k+1)整除。其思路是求出a和n的阶乘的所有质数及其次数，整除要求n中对应质数的次数要大于等于a的k次方中的次数，因此可以对a和n所有质数的次数求出来后，相除即得该质数位置处的k的最大值，对所有质数位置处的k，取最小值即为结果

求可能质数时，可以使用埃氏筛法得到小于等于a的范围内的所有质数，作为潜在的质数集合

~~~c++ 
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
#include <bits/stdc++.h>
using namespace std;

//埃氏筛法
vector<int> get_primes(int a){
    vector<int> primes;
    vector<bool> flags(a+1,true);
    for(int i=2;i<=a;i++){
        if(flags[i]){
            primes.push_back(i);
            for(int j=i*i;j<=a;j+=i) flags[j]=false;
        } 
    }
    return primes;
}

map<int, int> cal_a(int a, vector<int>& primes){
    map<int, int> mp;
    for(auto p:primes){
        while(a%p==0){
            mp[p]++;
            a/=p;
        }
    } 
    return mp;
}

map<int, int> cal_n(int n, vector<int>& primes){
    map<int, int> mp;
    for(auto p:primes){
        int base=p;
        while(n/base>0){
            mp[p] += n/base;
            base *= p;
        }
    }
    return mp;
}

int main() {
    int n,a;
    cin>>n>>a;
    vector<int> primes=get_primes(a);
    map<int, int> map_a=cal_a(a, primes);
    map<int, int> map_n=cal_n(n, primes);

    vector<int> k;
    for(auto p:primes){
        if(map_a[p]!=0){
            k.push_back(map_n[p]/map_a[p]);
        }
    }
    cout<<*min_element(k.begin(),k.end());
    return 0;
}
~~~



### 最小面积矩阵

从第i行加到第j行，然后在得到的结果数列上，求一个和大于等于k的最短连续子序列，可以使用双指针法，从头遍历一遍得到

~~~c++
#include <climits>
#include <iostream>
#include <vector>
using namespace std;

int shortest_subarr(vector<int>& arr, int k){
    int len=arr.size();
    int sum=0;
    int result = INT_MAX;
    int p=0,q=0;
    while(p<len && q<len){
        if(sum<k){
            sum+=arr[q];
            q++;
        }
        else{
            result = (q-p<result)?(q-p):result;
            sum-=arr[p];
            p++;
        }
    }
    if(p==0&&q==len) return -1;
    else return result;
}

vector<int> sum_arrs(vector<vector<int>>& m, int i, int j){
    vector<int> result;
    for(int l=0;l<m[0].size();l++){
        int val=0;
        for(int k=i;k<=j;k++) val+=m[k][l];
        result.push_back(val);
    }
    return result;
}

int main() {
    int a, b, k;
    cin>>a>>b>>k;
    vector<vector<int>> matrix(a,vector<int>(b,0));
    for(int i=0;i<a;i++){
        for(int j=0;j<b;j++)
            cin>>matrix[i][j];
    }

    vector<vector<int>> S(a,vector<int>(a,0));
    int min_s = INT_MAX;
    for(int i=0;i<a;i++){
        for(int j=i;j<a;j++){
            vector<int> arr = sum_arrs(matrix, i, j);
            int curr_s = shortest_subarr(arr, k) * (j-i+1);
            if(curr_s>=0)
                min_s = (curr_s<min_s)?curr_s:min_s;
        }
    }
    cout<<min_s;
    return 0;
}
~~~



### day of week

所有的日子，都计算其和0001年1月1日的日期差，将各个功能函数分别实现即可

注意**当前月份的天数不应当计算进入日期中**

注意vector的初始化方式

~~~c++
#include <iostream>
#include <bits/stdc++.h>
#include <vector>
using namespace std;


vector<string> MONTH={"January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"};
vector<string> WEEK={"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};

vector<vector<int>> month_day={
    {31,28,31,30,31,30,31,31,30,31,30,31},
    {31,29,31,30,31,30,31,31,30,31,30,31}
};

bool is_leapyear(int n){
    return (n%100!=0 && n%4==0)||(n%400==0);
}

int before_year(int n){
    int result=0;
    for(int i=1;i<n;i++) result+=(is_leapyear(i)?366:365);
    return result;
}

int current_year(int year, int month, int date){
    int result=0;
    for(int i=0;i<month-1;i++) result+=(is_leapyear(year)?month_day[1][i]:month_day[0][i]);
    result+=date;
    return result;
}

int main() {
    int date, year;
    string month;
    while(cin>>date>>month>>year){
        int days=0;
        int month_val = find(MONTH.begin(),MONTH.end(),month)-MONTH.begin()+1;
        days = before_year(year)+current_year(year, month_val, date);
        int result = days%7;
        cout<<WEEK[result]<<endl;
    }
    
    return 0;
}
~~~



### 日期差值

类似上一题，只不过此时计算差值而已

~~~c++
#include <cstdlib>
#include <iostream>
#include <bits/stdc++.h>
using namespace std;

vector<vector<int>> month_day = {
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
};

bool is_leapyear(int n) {
    return (n % 100 != 0 && n % 4 == 0) || (n % 400 == 0);
}

int before_year(int n){
    int result=0;
    for(int i=1;i<n;i++) result+=(is_leapyear(i)?366:365);
    return result;
}

int current_year(int year, int month, int date){
    int result=0;
    for(int i=0;i<month-1;i++) result+=(is_leapyear(year)?month_day[1][i]:month_day[0][i]);
    result+=date;
    return result;
}

int main() {
    string a, b;
    while (cin >> a >> b) {
        int year1 = atoi(a.substr(0, 4).c_str()), year2 = atoi(b.substr(0, 4).c_str());
        int month1 = atoi(a.substr(4, 2).c_str()), month2 = atoi(b.substr(4,2).c_str());
        int date1 = atoi(a.substr(6, 2).c_str()), date2 = atoi(b.substr(6, 2).c_str());

        int days1=before_year(year1)+current_year(year1, month1, date1);
        int days2=before_year(year2)+current_year(year2, month2, date2);

        cout<<(days2-days1)+1<<endl;
    }
    return 0;
}
// 64 位输出请用 printf("%lld")
~~~



### 字符串匹配

深刻记住next数组的含义，并且需要注意当长度为1时，next[p]=0，即自己和自己不算

当需要计算总的出现次数时，达到一次长度后，将p依旧设置为next[p-1]即可

~~~c++
#include <iostream>
#include <vector>
using namespace std;

int get_next(string p, int id){
    string sub = p.substr(0,id+1);
    for(int i=sub.size()-1;i>=1;i--) if(sub.substr(0,i)==sub.substr(sub.size()-i,i)) return i;
    return 0;
}

vector<int> get_next(string p){
    vector<int> next(p.size());
    for(int i=0;i<p.size();i++){
        next[i]=get_next(p, i);
    }
    return next;
}

int main() {
    string T,P;
    cin>>T>>P;
    vector<int> next = get_next(P);
    int t=0,p=0;
    int count=0;
    while(t!=T.size()){
        if(T[t]==P[p]){
            t++;p++;
        }
        else if(p){
            p = next[p-1];
        }
        else t++;
        if(p==P.size()){
            count++;
            p = next[p-1];
        }
    }
    cout<<count;
    return 0;
}
~~~



### 求最小差值的数组拆分

常规思想可以使用01背包求解，但是本例中数字太大，内存超出限制

可以使用dfs进行搜索，对所有可能的情况（和小于half）进行遍历，但是需要一些剪枝的操作

这里有一个很精妙的剪枝，对vec先进行从大到小的排序，然后当`sum+(vec.size()-idx)*vec[idx]<curr_ans`时，直接退出，意思即后续所有的数加起来都不会大于当前最佳结果了，也就没必要继续搜索了

~~~c++ 
#include <algorithm>
#include <iostream>
#include <bits/stdc++.h>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

long long curr_ans=0;

void dfs(long long idx, long long sum, vector<long long>& vec, long long& target){
    if(idx==vec.size() || sum+(vec.size()-idx)*vec[idx]<curr_ans) return;
    
    if(sum+vec[idx]<=target){
        int new_sum = sum+vec[idx];
        curr_ans = (new_sum>curr_ans)?new_sum:curr_ans;
        dfs(idx+1,new_sum,vec,target);
    }
    dfs(idx+1,sum,vec,target);
}

bool is_valid(string str){
    if(str[0]=='-') str=str.substr(1,str.size());
    for(long long i=0;i<str.size();i++) {
        if(str[i]>'9'||str[i]<'0') return false;
    }
    return true;
}

vector<long long> preprocess(string str){
    vector<long long> result;
    istringstream is(str);
    string tmp;
    while(is>>tmp){
        if(is_valid(tmp)){
            result.push_back(atoi(tmp.c_str()));
        }
        else {
            result.clear();
            return result;
        }
    }
    return result;
}

int main() {
    string input;
    while(true){
        getline(cin, input);
        if(input=="")break;
        vector<long long> vec = preprocess(input);
        if(vec.size()==0) {cout<<"ERROR"<<endl;continue;}
        long long s = accumulate(vec.begin(), vec.end(), 0);
        sort(vec.begin(),vec.end());
        reverse(vec.begin(),vec.end());
        long long target = s/2;
        dfs(0, 0, vec, target);
        cout<<s-curr_ans<<' '<<curr_ans<<endl;
        // vector<long long> dp(target+1,0);
        // for(long long i=0;i<vec.size();i++){
        //     for(long long j=target;j>=vec[i];j--){
        //         dp[j] = max(dp[j], dp[j-vec[i]]+vec[i]);
        //     }
        // }
        // cout<<s-dp[target]<<' '<<dp[target]<<endl;
        curr_ans=0;
    }

    return 0;
}
~~~



### 求连通分量数目

求连通分量，就使用并查集来实现，注意Find路径压缩的时候，记得判断father[x]!=x，否则会无限递归

~~~c++
#include <iostream>
#include <unordered_map>
using namespace std;

unordered_map<int, int> father;
unordered_map<int, int> height;

int Find(int x){
    if(father.find(x)!=father.end()){
        if(father[x]!=x)
            father[x] = Find(father[x]);
        return father[x];
    }
    else{
        father[x] = x;
        height[x] = 0;
        return x;
    }
}

void Union(int x,int y){
    x = Find(x);
    y = Find(y);
    if(x==y) return;
    else {
        if(height[x]>height[y]) father[y]=x;
        else if(height[x]<height[y]) father[x]=y;
        else{
            father[y] = x;
            height[x]++;
        }
    }
}

int main() {
    int a,b;
    while(cin>>a>>b) Union(a,b);
    int result=0;
    for(auto f:father) if(f.first==f.second) result++;
    cout<<result;
    return 0;
}
// 64 位输出请用 printf("%lld")
~~~





## 图论

### 并查集

可以使用map来进行实现

~~~c++
map<int,int> father;  
map<int,int> height;  
int find(int x){
    if(father.find(x)!=father.end()){
        if(father[x]!=x)
            father[x]=find(father[x]);  //路径压缩(最后自己通过例子模拟下过程)
    }
    else{//如果还没有出现的新节点。把father设成他自己(表示根节点)，height设成0
        father[x]=x;
        height[x]=0;
    }
    return father[x];
}
void Union(int a,int b){//合并函数
    a=find(a);
    b=find(b);
    if(a!=b){
        if(height[a]>height[b])
            father[b]=a;
        else if(height[b]>height[a])
            father[a]=b;
        else{
            father[a]=b;
            height[a]++;
        }
    }
}
~~~





### Floyd算法

全图的最短路径算法，时间复杂度n三次方，但是简单

注意最外层遍历的是中间节点即可

~~~c++
int main(){
    int n,m;
    cin>>n>>m;
    vector<vector<int>> dist(n+1,vector<int>(n+1,INT_MAX/100));
    for(int i=1;i<=n;i++) {dist[i][i]=0;}
    int a,b,c;
    while(cin>>a>>b>>c) dist[a][b] = dist[b][a] = c;    
    for(int h=1;h<=n;h++){
        for(int j=1;j<=n;j++){
            for(int k=1;k<=n;k++){
                if(dist[j][k]>dist[j][h]+dist[h][k])
                    dist[j][k]=dist[j][h]+dist[h][k];
            }
        }
    }
    cout<<dist[1][n];
}    
~~~





### Dijkstra算法

单源最短路径算法

初始只有源节点，其他节点距离设为无穷大

每次挑选一个，dist最近的未进入集合的节点，加入集合，同时根据这个新加入节点，更新所有其他节点的dist，重复以上过程直到集合大小等于节点数目

下面示例代码中，节点序号从1到N，求节点1到节点N的最短路径

**代码中需要注意，一开始不要将节点1加入used集合**，否则会缺少一次从1到相邻节点的距离更新

~~~c++
#include <bits/stdc++.h>
#include <vector>
using namespace std;

int main(){
    int n,m;
    cin>>n>>m;
    unordered_set<int> used;
    vector<vector<int>> cost(n+1,vector<int>(n+1,INT_MAX/100));
    vector<int> dist(n+1,INT_MAX/100);
    for(int i=1;i<=n;i++) {cost[i][i] = 0;}
    dist[1] = 0;
    
    int a,b,c;
    while(cin>>a>>b>>c) cost[a][b] = cost[b][a] = c;

    while(true){
        int v=-1;
        for(int u=1;u<=n;u++){
            if(used.find(u)==used.end()&&(v==-1||dist[u]<dist[v]))
                v=u;
        }
        if(v==-1) break;
        used.insert(v);
        for(int u=1;u<=n;u++){
            dist[u] = min(dist[u],dist[v]+cost[v][u]);
        }
    }
    cout<<dist[n];

}
~~~





### 拓扑排序

1. 建图
2. 算入度
3. 将入度为0的加入队列
4. 入度为0的结点队列出队，减去该结点所连接的点的入度
5. 反复循环知道所有点都已经遍历过了

```c++
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>>g(numCourses,vector<int>());
    vector<int>inDegree(numCourses,0);
    for(auto&v:prerequisites){
        g[v[0]].push_back(v[1]);
        inDegree[v[1]]++;
    }
    queue<int>q;
    for(int i=0;i<numCourses;++i){
        if(inDegree[i]==0)q.push(i);
    }
    int cnt=0;
    while(q.size()){
        cnt++;
        int now=q.front();
        q.pop();
        for(int v:g[now]){
            if(--inDegree[v]==0)q.push(v);
        }
    }
    return cnt==numCourses;
}
```



### 最小生成树

Kruscal算法，思路很简单，设置一个并查集，每次找最短的一条边，若是其左右节点不在一个集合中，则union，直到所有的节点都在一个集合中

~~~c++
#include<iostream>
#include<cstdio>
#include<vector>
#include<algorithm>
using namespace std;

class Edge {
public:
    int from, to, w;
    Edge(int f, int t, int W) {
        from = f; to = t; w = W;
    }
};

vector<int> parents;
vector<Edge> edges;

int find(int x) {
    if (parents[x] != x)
        parents[x] = find(parents[x]);
    return parents[x];
}

void Union(int x, int y) {
    int rootx = find(x), rooty = find(y);
    parents[rootx] = rooty;
}

int main(){
    int n, m;
    cin >> n >> m;
    for (int i = 0; i < n; i++) {
        int a, b, c;
        cin >> a >> b >> c;
        edges.push_back(Edge(a-1, b-1, c));
    }
    for (int i = 0; i < m; i++)
        parents.push_back(i);

    sort(edges.begin(), edges.end(), [](Edge& e1, Edge& e2){return e1.w < e2.w;});

    int cnt = 0, ans = 0, index = 0;
    while (cnt < m && index < edges.size()) {
        Edge cur = edges[index];
        if (find(cur.from) != find(cur.to)) {
            Union(cur.from, cur.to);
            ans += cur.w;
            cnt++;
        }
        index++;
    }
    cout << ans;
    return 0;
}
~~~





### 欧拉回路

判断是否存在欧拉回路的方法：所有点的出度都为偶数或者只有两个点为奇数

