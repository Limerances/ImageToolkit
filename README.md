# ImageToolkit

### 一、使用

可以直接运行`dist/main.exe`可执行程序

### 二、环境搭建与运行

```cmd
pip install -r requirements.txt
python main.js
```

项目使用`pyinstaller`进行打包

```cmd
pyinstaller -F -w -i icon\favicon.ico --add-data "image;image" --add-data "icon:icon" main.py
```



