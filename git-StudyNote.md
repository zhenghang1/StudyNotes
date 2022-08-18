# git 学习笔记



## 一 基础

### 1.1 基础命令

#### 1.1.1 git diff

不加参数，表示工作目录(Working tree)和暂存区域快照(index)之间的差异，也就是修改后未暂存add的内容

加--cached（或--staged），表已经暂存add，但未commit的部分与上次commit的内容的区别



#### 1.1.2 git rm

rm会取消跟踪，并且从目录中删掉该文件

若只是想取消跟踪，则需要加上--cached

命令后面可以添加文件名或者目录名，采用后面所述的glob模式

如果删除之前修改过并且已经放到暂存区域的话，则必须要用强制删除选项 -f

  

#### 1.1.3 git mv

本质上，会进行改名，删除原名的跟踪，加上现名的跟踪三个操作



#### 1.1.4 git log

一个常用的选项是 -p，用来显示每次提交的内容差异。

也可以加上 -2 来仅显示**最近两次提交**：  

git log的一些选项

![image-20220627161442993](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220627161442993.png)

限制选项

![image-20220627162441312](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220627162441312.png)

#### 1.1.5 git reset

https://www.runoob.com/git/git-reset.html

命令格式：

~~~shell
git reset [ --soft | --mixed | --hard ] [HEAD] <file>
~~~

其中，file是可选的，HEAD是版本号

reset有三种选项：

+ hard -> 全部删除
+ soft -> 保留工作目录、暂存区
+ mixed -> 保留工作目录**（默认）**

HEAD格式：

- HEAD 表示当前版本，HEAD^ 上一个版本，以此类推
- 也可以用数字，HEAD\~0，表示当前版本，HEAD\~1表示上一个版本，以此类推

其中，git reset HEAD表示取消暂存



### 1.2 文件状态变化周期

![image-20220623133305964](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220623133305965.png)



### 1.3 gitignore

格式规范：

+ 所有空行或者以 ＃ 开头的行都会被 Git 忽略。
+  可以使用标准的 glob 模式匹配。
+  匹配模式可以以（/）开头防止递归。（表示当前目录下）
+  匹配模式可以以（/）结尾指定目录。（表示忽略指定该目录下所有文件）
+  要忽略指定模式以外的文件或目录，可以在模式前加上**惊叹号（!）取反。**  



所谓的 glob 模式是指 shell 所使用的简化了的正则表达式:

+ 星号（\*）匹配零个或多个任意字符
+ [abc] 匹配任何一个列在方括号中的字符（这个例子要么匹配一个 a，要么匹配一个 b，要么匹配一个 c）
+ 问号（?）只匹配一个任意字符
+ 如果在方括号中使用短划线分隔两个字符，表示所有在这两个字符范围内的都可以匹配（比如 [0-9] 表示匹配所有 0 到 9 的数字）
+ 使用两个星号（*) 表示匹配任意**中间目录**，比如a/\*\*/z 可以匹配 a/z, a/b/z 或 a/b/c/z等

也就是说，*表示任意多个字符，\*\*表示任意多个中间目录



github项目地址（包含多种语言的可以gitignore文件）：https://github.com/github/gitignore  



### 1.4 撤销操作

#### 1.4.1 补充提交/修改提交信息：

~~~shell
git commit --amend
~~~

比如，可以类似：

~~~shell
$ git commit -m 'initial commit'
$ git add forgotten_file
$ git commit --amend
~~~

如此，只会保留第二次提交操作，第一次会被覆盖



#### 1.4.2 取消暂存

~~~shell
git reset HEAD <file>... 
~~~

可以将一个已经add暂存的文件重新恢复为未暂存的状态，若不加\<file\>，则是对所有的都取消暂存

git reset命令加上--hard选项，则是代表版本回退，是一个比较危险的命令



#### 1.4.3 撤消对文件的修改  

本质上就是用上一个版本进行覆盖

~~~shell
git checkout -- <file> 
~~~

也可以使用reset命令

~~~shell
git reset HEAD^ <file>
~~~

表示回到上一个提交版本



### 1.5 远程仓库

#### 1.5.1 查看

~~~shell
git remote

git remote -v
~~~



#### 1.5.2 添加一个新的远程 Git 仓库

~~~shell
git remote add <shortname> <url> 
~~~



#### 1.5.3 数据拉取

+ git fetch

  获得的是所有的分支，且不会自动合并，形式如下，需要手动合并

  ![image-20220628122615448](https://cdn.staticaly.com/gh/zhenghang1/Image@main/img/image-20220628122615448.png)

  也可以如下：

  ~~~shell
  git fetch <远程主机名> <远程分支名>:<本地分支名>
  ~~~

  此时，只会将对应分支拉到本地对应分支处（不存在的话自动创建）

+ git pull

  会抓取所有分支，并且将会对应远程分支自动尝试合并到本地分支（可以先用git remote show看下具体合并情况

  同样也可以：

  ~~~shell
  git pull <远程主机名> <远程分支名>:<本地分支名>
  ~~~

  

#### 1.5.4 查看远程仓库

~~~shell
git remote show [remote-name]
~~~

会告诉你各个操作后对应的可能结果（git pull，git push等）



#### 1.5.5 远程仓库的移除与重命名 

+ git remote rm

  用来移除某个远程仓库，和git remote add作用相反

+ git remote rename

  用来重命名，语法如下：

  ~~~shell
  git remote rename <current-name> <target-name>
  ~~~

  

### 1.6 标签tag

#### 1.6.1 标签分类

+ 附注标签，类似完整版，拥有足够多的信息
+ 轻量标签，只是对一个固定commit的引用（有点类似别名，没有其他多余的信息）



#### 1.6.2 标签创建

+ 附注标签

  ~~~shell
  git tag -a v1.4 -m 'my version 1.4'
  ~~~

  -a可以理解为add，创建的意思，-m是添加标签信息，附注标签必须添加关联的信息，未加指定tag对应的提交的话，表示对当前最近一次commit打标签

  假如想为某个特定commit打标签：

  ~~~shell
  git tag -a V1.0 55d8e71fc7d0b8cefbb4cbee339beb9d987d9b81 -m '正式版本'
  ~~~

  55d8e71fc7d0b8cefbb4cbee339beb9d987d9b81也可以使用对应的部分校验和

+ 轻量标签

  不加-a，不加信息，只需要直接写上标签名和对应的commit校验和即可



#### 1.6.3 查看标签

+ 查看全部标签

  ~~~shell
  git tag
  ~~~

+ 查看某个特定标签

  ~~~shell
  git show <tag-name>
  ~~~



#### 1.6.4 删除标签

~~~shell
git tag -d <tag-name>
~~~



#### 1.6.5 标签推送

默认情况下，git push 命令并不会传送标签到远程仓库服务器上

在创建完标签后你必须显式地推送标签到共享服务器上，这个过程就像共享远程分支一样 ：

~~~shell
git push origin <tag-name>
~~~

如果想要一次性推送很多标签，也可以使用带有 --tags 选项的 git push 命令，这将会把所有不在远程仓库服务器上的标签全部传送到那里。

~~~shell 
git push origin --tags
~~~



#### 1.6.6 标签检出

标签对应的是一个commit，因此检出标签本质上就是调出对应commit时的代码，可以直接采用如下方式检出：

~~~shell
git checkout V1.0
~~~

更常见的是新建一个分支再在上面检出：

~~~shell
git checkout -b Version1 V1.0
~~~

该命令创建了一个名为Version1的分支，并将V1.0对应commit的代码检出



### 1.7 git的别名

别名是针对某个命令而言的，可以使用如下的命令来设置：

~~~shell
git config --global alias.co checkout

git config --global alias.unstage 'reset HEAD --'
~~~

表示用co来作为checkout的别名，用unstage来作为reset HEAD --的别名

