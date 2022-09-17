## 运算符

+ 海象运算符：`:=`，将为变量赋值和使用变量合为一体，示例：`if (n := len(a)) > 10:`，先把len(a)赋值给n，然后再使用n的值进行判断

+ 逻辑运算符：

  ![image-20220902092739476](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220902092739476.png)

+ 成员运算符：

  ![image-20220902092813500](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220902092813500.png)

+ 身份运算符：

  ![image-20220902092837853](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220902092837853.png)



## 基本数据结构

### 字符串

使用`""`或`''`括起来的内容，不可修改，可以使用索引或切片

几个处理函数：

+ `title()`：将每个单词的首字母变为大写，其他变为小写，类似的还有：`upper()`  和 `lower()`大写和小写
+ `strip()`，删除字符串两侧的空白（制表符or空格），而`lstrip()`和`rstrip()`则分别是删除左侧和右侧的空白



### 列表list

使用方括号`[]`括起来，并用逗号隔开的元素组成一个列表，可以修改，可以使用索引或切片
索引时，负数索引表示倒数第n个元素，如`list[-1]`表示倒数第一个元素

创建空列表：`list = []`



元素的增加：

+ `append(val)`：会在列表的末尾插入一个元素val
+ `insert(val，n)`：在列表的任意位置n处插入一个元素val，后续元素自动后移



元素的删除：

+ `del`语句：`del list[n]`删除列表的第n个元素，其他元素自动前移
+ `pop()`：根据索引删除元素，不加参数的情况下默认删除最后一个元素，加上参数n即表示删除第n个元素
+ `remove(val)`：根据值删除元素，注意remove只会删除第一个val，若有多个val元素则需要循环删除



组织列表：

+ `sort()`和`sorted()`：sort对列表元素进行排序，不可恢复，sorted进行暂时排序，返回值是已排序列表，但不影响原有排序，若要逆序排序，只需要加入参数`reverse=True  `
+ `reverse()`：反转列表
+ `len()`：确定长度



创建数值列表：

+ `range( begin , end , step )`：按照step的步长，从begin到（end-1），step可省略，若只使用一个参数，range(n)表示从0到n-1
+ 使用`list()` 函数创建一个列表：`target = list(range(n))`



列表推导式：

```
[表达式 for 变量 in 列表 if 条件]
[out_exp_res for out_exp in input_list if condition]
```



切片：

+ 遍历列表的一部分：在for循环的列表后加上切片范围：`for target in targets[begin:end]:`  
+ 复制列表：复制列表不能只是简单的使用赋值符号`=`，如`new_list = list`得到的new_list和list指向的内容是一致的，是同一个列表，有点类似c++中的引用。
  若要复制列表，需采用切片的方式：`new_list = list[:]`；除此之外，也可以使用list的`copy()`方法：`new_list = list.copy()`



列表操作：

![image-20220902090005884](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220902090005884.png)

列表的比较：需要引入`operator `模块的` eq `方法：`operator.eq(a,b)`



### 元组tuple

与列表类似，区别在于不能元组的元素不可以被修改，但元组变量本身是可以被修改的（即指向一个新的元组），也可以使用+，* 等对元组进行拼接等。

元组的所有操作基本都和列表类似，不再赘述。



### 集合set

集合可以理解为就是数学上的集合的含义

创建空集合：`set()`，注意不是使用`{}`，因为这是创建空字典的方式；创建包含元素的集合：`set={x1,x2,...}`

集合可以进行一系列的集合运算



具体的方法见[这里](https://www.runoob.com/python3/python3-set.html)





### 字典dict

字典的构造方式：

~~~Python
dict{}
dict['one']=xxxx
dict[2]='xxxx'

>>> dict([('Runoob', 1), ('Google', 2), ('Taobao', 3)])
{'Runoob': 1, 'Google': 2, 'Taobao': 3}

>>> {x: x**2 for x in (2, 4, 6)}	#推导式
{2: 4, 4: 16, 6: 36}

>>> dict(Runoob=1, Google=2, Taobao=3)
{'Runoob': 1, 'Google': 2, 'Taobao': 3}
~~~

删除键值对，采用del语句：`del dict[key]`会将key对应的键值对删除掉



字典的遍历：

+ 遍历键值对：`for key,value in dict.items():`其中key和value可以是任意的变量名，中间用逗号隔开

+ 遍历值或者键：分别使用`keys()`和`values()`方法，如下：`for key in dict.keys():`，`for value in dict.values():`

以上，`items()`，`keys()`和`values()`方法都是返回一个列表



`pop()` 和`popitem()`：`pop()`接受一个参数key，会删掉倒数第一个键为key的键值对；而`popitem()`不接受参数，删掉最后一个键值对



字典推导式

~~~
{ key_expr: value_expr for value in collection }

{ key_expr: value_expr for value in collection if condition }
~~~



while语句遍历：

+ 判断条件可以直接是`while list:`，只要其不为空就会一直循环
+ 判断条件也可以是`in`和`not in`语句，对元素进行筛选



## 文件操作





## 类