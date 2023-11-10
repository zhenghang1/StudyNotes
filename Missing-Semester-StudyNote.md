# Missing-Semester Learning Note

> 课程网站：https://missing.csail.mit.edu/2020/

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

### 正则表达式 regex

已学习并整理，详见[这里](https://github.com/zhenghang1/StudyNotes/blob/master/regex-StudyNote.md)



### 示例分析

~~~shell
ssh myserver journalctl
 | grep sshd
 | grep "Disconnected from"
 | sed -E 's/.*Disconnected from (invalid |authenticating )?user (.*) [^ ]+ port [0-9]+( \[preauth\])?$/\2/'
 | sort | uniq -c
 | sort -nk1,1 | tail -n10
 | awk '{print $2}' | paste -sd,
~~~

+ `uniq -c`：`uniq`是用来消除相同的项的工具，其中`-c`的标志表示需要`uniq`在消除相同项的同时统计其出现次数，并且放在第一列（项的前面）

+ `sort -nk1,1`：其实是`sort -n -k1,1`，合并的写法，`-n`指的是numeric，根据数字大小进行排序（而非字典序）；`-k`指的是对排序的列进行限制，从第1列到第1列（不加限制的话就是默认对一整行进行排序）

+ `tail -n10`：显示其倒数10行，`-n+数字`代表多少行

+ `awk '{print $2}'`：打印其第二列的数据

+ `paste -sd,`：`paste`命令会将文件以列对齐的方式进行合并，（即将多个文件按列对齐并输出）；此外`-s`标志表示serial，串列进行，效果是将一个文件中的每一行的内容整合到同一行中；`-d+符号`表示串列处理后，各列内容间的间隔符号（此处是,）

  ~~~shell
  cat file                  #file文件的内容  
  xiongdan 200
  lihaihui 233
  lymlrl 231
  
  paste -sd, file
  xiongdan 200,lihaihui 233,lymlrl 231
  ~~~



### awk

`awk`是一种处理文本文件的编程语言，也是一个强大的文本分析工具，其更关注于对列的处理。拥有了`awk`，甚至可以用来完成`grep`和`sed`所能完成的所有任务



其他一些数据统计方面的内容，暂无需求，不进行整理

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

+ sessions：一个组，可命名，可包含多个windows
+ windows：可命名，占满整个窗口，可分割为多个pane
+ panes：将一个窗口分为多个长方形的小窗口，每个称为一个pane

tmux中所有的命令都需要先键入前缀快捷键，默认是`Ctrl+b`，可以修改为`Ctrl+a`更加方便，详见[这里](https://blog.csdn.net/lwgkzl/article/details/100799042)

+ session

~~~shell
//创建session
tmux

//创建并指定session名字
tmux new -s $session_name

//删除session
Ctrl+b :kill-session

//删除所有session
Ctrl+b :kill-server

//删除指定session
tmux kill-session -t $session_name

//列出session
tmux ls

//进入已存在的session
tmux a -t $session_name

//临时退出session
Ctrl+b d
~~~

+ windows

~~~shell
//创建window
Ctrl+b c

//删除window
Ctrl+b &

//下一个window
Ctrl+b n

//上一个window
Ctrl+b p

//重命名window
Ctrl+b ,

//在多个window里搜索关键字
Ctrl+b f

//在相邻的两个window里切换
Ctrl+b l
~~~

+ pane

~~~shell
//创建pane
//横切split pane horizontal
Ctrl+b ” (问号的上面，shift+’)

//竖切split pane vertical
Ctrl+b % （shift+5）

//按顺序在pane之间移动
Ctrl+b o

//上下左右选择pane
Ctrl+b 方向键上下左右

//调整pane的大小
Ctrl+b :resize-pane -U #向上
Ctrl+b :resize-pane -D #向下
Ctrl+b :resize-pane -L #向左
Ctrl+b :resize-pane -R #向右
在上下左右的调整里，最后的参数可以加数字 用以控制移动的大小，例如：
Ctrl+b :resize-pane -D 50

//在同一个window里左右移动pane
Ctrl+b { （往左边，往上面）
Ctrl+b } （往右边，往下面）

//删除pane
Ctrl+b x

//更换pane排版
Ctrl+b “空格”

//移动pane至window
Ctrl+b !

//移动pane合并至某个window
Ctrl+b :join-pane -t $window_name

//显示pane编号
Ctrl+b q

//按顺序移动pane位置
Ctrl+b Ctrl+o
~~~

+ else

~~~shell
复制模式
Ctrl+b [
空格标记复制开始，回车结束复制。

//粘贴最后一个缓冲区内容
Ctrl+b ]

//选择性粘贴缓冲区
Ctrl+b =

//列出缓冲区目标
Ctrl+b :list-buffer

//查看缓冲区内容
Ctrl+b :show-buffer

//vi模式
Ctrl+b :set mode-keys vi

//显示时间
Ctrl+b t

//快捷键帮助
Ctrl+b ? (Ctrl+b :list-keys)

//tmux内置命令帮助
Ctrl+b :list-commands
~~~



### 别名alias

为一些较长的命令或较为常用的参数组合设置别名，本质上就是一个文本替换，可以使用如下的命令设置别名：

~~~shell
alias alias_name="command_to_alias arg1 arg2"
~~~

注意其中，`=`两端并没有空格，因为`alias`命令只接受一个参数

可以使用如下命令取消别名：`unalias alias_name`

使用如下命令查看当前已有别名：`alias alias_name`



### dotfiles

各种程序的配置文件通常都以dotfile的形式存在，不加赘述

为了方便在新机器上重现自己的配置或者用作分享，可以将自己的dotfiles都放在一个专门的目录并且纳入版本控制，传到github上，并使用symlink将其链接到软件要求的目录中去，使用symlink的[具体方式](https://blog.csdn.net/cumian8165/article/details/108101156)：

~~~shell
#创建符号链接，其中-s标志是软链接的标志（否则默认硬链接）
ln -s <path to the file/folder to be linked> <the path of the link to be created>

#删除符号链接，两种方式
unlink <path-to-symlink>
rm <path-to-symlink>
#删除时，注意就算是一个指向目录的链接，也不需要加上/，因为我们只关注这个链接（可视为一个文件）
~~~

一个dotfiles仓库链接：https://github.com/mathiasbynens/dotfiles



### 可移植性 portability

在dotfiles中，有些时候要使得配置文件在各个操作系统或shell中都可以适用，有些时候又想要其只在某几台特定设备上可以使用，可以使用条件判断完成这些目标：

~~~shell
if [[ "$(uname)" == "Linux" ]]; then {do_something}; fi

# Check before using shell-specific features
if [[ "$SHELL" == "zsh" ]]; then {do_something}; fi

# You can also make it machine-specific
if [[ "$(hostname)" == "myServer" ]]; then {do_something}; fi
~~~

或者可以不同的配置文件间共享相同的配置：
如在bash和zsh中共享相同的别名设置，将其独立为一个`~/.aliases`文件，并在`~/.bashrc`和`~/.zshrc`中分别加入：

~~~shell
# Test if ~/.aliases exists and source it
if [ -f ~/.aliases ]; then
    source ~/.aliases
fi
~~~



### Remote Machines

利用ssh连接远程服务器

~~~shell
ssh foo@bar.mit.edu
~~~

其中，`foo`是用户名，`bar.mit.edu`是目标服务器，也可以使用ip来表示

直接如上面连接，会进入到服务器的shell界面，如果想留在本地shell中而只是在服务器上执行，可以在后面加上待执行的命令，如果想要在服务器端执行多个命令，需要用`''`括起来，如

~~~shell
ssh foobar@server ls | grep PATTERN 	#单个命令

ssh foobar@server 'ls | grep PATTERN' 	#多个命令
~~~



接下来介绍了生成和向远程主机发送ssh keys的方式，详见[这里](https://missing.csail.mit.edu/2020/command-line/)



通过ssh发送文件的方式，主要有三种：

+ `ssh`+`tee`的方式，最简单，传输少量文件时可以这样，tee读取标准输入，并写入一个文件
+ `scp`：用来传输大量文件，因为其可以递归地传输文件/目录，命令写法如下：`scp path/to/local_file remote_host:path/to/remote_file`
+ `rsync`：与`scp`类似，但提供了更多特性，如1、可以比较本地目录和远方目录的文件差异，只复制不存在的文件；2、可以通过`--partial`的标志，重启未进行完的传输（`scp`必须一次完整传输完）；3、等等



端口转发

很多软件会监听固定的某个端口，有些时候需要用本地的某个端口去对应另一个端口，可以使用类似`ssh -L 9999:localhost:8888 foobar@remote_server`的命令来进行端口转发



`~/.ssh/config`配置文件：可以在里面配置好常用的用户名和验证文件等，便于登录操作的进行，如下：

![image-20220824170545791](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220824170545791.png)





## Lecture 6: Version Control (Git)

已学习，并结合Pro Git的内容一并整理至这份笔记[git-StudyNote](https://github.com/zhenghang1/StudyNotes/blob/master/git-StudyNote.md)



## Lecture 8: Metaprogramming

### Build systems

make的学习
