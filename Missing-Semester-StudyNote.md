# Missing-Semester Learning Note

## Lecture 2: Shell Tools and Scripting 

### 变量定义

直接使用如 `foo=bar` 的赋值语句即可，不可以存在空格，存在空格的话会被解释为一个命令及其参数。后续若想使用对应的变量的值，只需要使用`$`符即可，即`$foo`会被替换为`bar`

### 字符串

shell中的字符串，使用双引号或者单引号都可，其区别在于双引号的字符串中，会对形如`$foo`的部分进行变量替换，而单引号的字符串中不进行替换

![image-20220814110740360](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220814110740360.png)

### 函数定义

新建一个.sh文件，其中输入函数定义如下：

![image-20220814111007254](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220814111007254.png)

我们创建了一个名为mcd的函数，其中包含两个命令，`"$1"`指的是该命令的第一个参数，`$2`~`$9`类比，`$0`指的是命令本身的名称
将该函数实例化，可使用`source`命令，之后即可使用如使用命令般使用该函数

### 一些常用符号

如上，`$+数字`通常指参数（两位数以上的要用{}括起来），此外：

+ `$?`会返回上一命令的返回值/错误码

+ `$_`会替换为上一命令的最后一个参数

+ `$*` 和`$@`作用一致，都是替换为该命令所有的参数

+ `$#`是替换为一个数字，表示该命令所有参数的数目

+ `$$`是替换为当前进程的pid

+ `$()`和 \` \`的作用相同，都是命令替换，括号和反引号中间加命令，会先执行该命令后将其结果进行替换

  补充：有另一个作用相似的运算符：`<()`，也是先执行括号内的命令，区别在于会先将其结果存于一个临时文件中，可用于那些需要输入参数为文件的命令，如下：
  ~~~shell
  diff <(ls foo) <(ls bar)	#比较foo和bar目录中所含文件的区别
  ~~~

+ `${}`是进行变量替换，即取出变量的值进行替换

+ `!!`会被替换为上一行命令

shell的逻辑符号：

![image-20220814160552401](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220814160552401.png)

+ `;`分号表示命令按顺序从左至右执行
+ `&&`和`||`就是逻辑与和逻辑或，用法不加赘述

其他符号：

+ `|`是管道符，前一个命令的输出会作为后一个命令的输入
+ `&`是后台运行符，将该命令进程放于后台运行



### shellcheck

是一个shell的工具，需要安装，可用来检查bash脚本的正确性



### tldr

是一个用于查找命令及其参数的对应用法的辅助性工具，包含使用例子等等

用法：`tldr 命令`



### shebang line

在shell脚本的第一行，用`#!`开头的一行，用来向shell指出该用什么解释程序来运行当前脚本，可以增强当前脚本的可移植性

~~~Python
#!/usr/local/bin/python
import sys
for arg in reversed(sys.argv[1:]):
    print(arg)
~~~

若想进一步提高可移植性，考虑到Python的安装位置不同，可以利用环境变量，将shebang改为如下：

~~~Python
#!/usr/bin/env python
~~~

其中，env是环境变量



### 脚本和函数的区别

+ 函数必须使用shell语言编写，脚本可以使用任意编程语言编写（所以脚本第一行需要shebang）
+ 函数只需要加载一次（利用`source`命令），而脚本每次都需要重新加载
+ 函数是在当前shell进程内运行的，而脚本是额外开启一个新进程来运行。因此，函数可以修改环境变量，eg.改变当前目录，而脚本不行，脚本只能使用那些已经`export`过的环境变量



### 查找文件

+ `find`命令

  可以根据文件的各种要素，如文件名称，文件路径，文件大小，修改日期，文件权限等等来查找文件，使用正则表达式，语法如下：

  ~~~shell
  find DIRECTORY OPTIONS
  ~~~

  示例如下：

  ~~~shell
  # Find all directories named src
  find . -name src -type d
  
  # Find all python files that have a folder named test in their path
  find . -path '*/test/*.py' -type f
  
  # Find all files modified in the last day
  find . -mtime -1
  
  # Find all zip files with size in range 500k to 10M
  find . -size +500k -size -10M -name '*.tar.gz'
  ~~~

  除了简单查找之外，`find`命令还有一个`-exec`选项，可以对查找所得的文件进行简单操作，如下：

  ~~~shell
  # Delete all files with .tmp extension
  find . -name '*.tmp' -exec rm {} \;
  
  # Find all PNG files and convert them to JPG
  find . -name '*.png' -exec convert {} {}.jpg \;
  ~~~

+ `fd`命令

  可理解为简化版的`find`命令，`find`命令的搜索根据路径进行匹配，而`fd`命令针对于文件名进行匹配（会递归搜索），还提供了一系列选项（如`-e`表示按后缀查找），因此使用更加简便，语法如下：
  ~~~shell
  fd PATTERN [DIRECTORY] #注意目录是可选的，默认都是当前目录，可省略
  ~~~

+ `locate`命令

  检索文件，但是不是搜索当前目录，而是在一个数据库中进行查找，所以速度很快
  需要注意的是，数据库是每日os调用定时任务进行更新的，刚创建的文件需要手动更新一下才可以搜索到（使用`updatedb`命令

  ~~~shell
  locate /etc/my 		#查找etc目录下，以my开头的文件
  /etc/my.cnf
  ~~~



### 查找代码/文件内容

`grep`命令

其作用在于查找给定的文件中是否含有特定的内容，具有一系列有用的选项，如`-C 数字`可以显示特定内容附近n行的内容，`-R`表示递归搜索，等等

有多个更好用的命令工具（需要安装，github链接见[原note](https://missing.csail.mit.edu/2020/shell-tools/)）
如`ack`，`ag`，`rg`等



### 查找历史命令

+ `history`命令
+ 可以通过`history | grep "string"`的方式进行历史命令的筛选
+ `ctrl+R`可以进入一个动态的历史命令查找的程序，在该程序中，可以利用↑和↓进行查找，在对应命令处输入`enter`进行执行，输入`Esc`输入该命令但不执行
+ `zsh`的智能推荐补全插件`zsh-autosuggestions`



### 目录漫游

+ 常用目录跳跃：
  + fasd
  + autojump
+ 目录结构
  + tree
  + broot
+ file manager
  + nnn
  + ranger



## Lecture 4: Data Wrangling

### sed命令

流编辑器，功能十分强大，可以进行文本增删，查找替换等，详见[这里](https://www.runoob.com/linux/linux-comm-sed.html)

文本替换命令：`sed 's/old(regexp)/new/'`，其中s是替换选项（substitution），假如将new一项置为空，即可将匹配的内容删除

### 正则表达式 regexp



## Lecture 5: Command-line Environment

### 信号

`ctrl+C`的终止信号，会发出SIGINT的中断信号，是可以在程序中接受并进行处理的，如果设置了这样的handler，则`ctrl+C`信号不会导致程序退出，而是触发对应的handler

有一些信号是无法被软件截获的，如SIGKILL



`ctrl+Z`会发出一个`SIGSTOP`信号，将当前进程暂停

可以使用`jobs`命令查看当前的任务，并且根据其任务序号，可以使用（1）`fg %num`或（2）`bg %num`命令重新开始第num个任务，`fg`和`bg`分别代表前台和后台运行



`kill` 命令，直接使用的话是杀死某个进程，但是其实它有多个选项，可以传递任意信号给某个进程，比如：

~~~shell
kill -STOP %1
~~~

是传递`SIGSTOP`信号给第一个进程，使其暂停



`nohup`命令，加在一个正常命令的前面，表示当前任务无法被挂起，本质上就是该进程会忽视所有的SIGHUP信号，不会在运行过程中被挂起



### terminal multiplexer

tmux

+ sessions
+ windows
+ panes