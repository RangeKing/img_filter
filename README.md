# 图片筛选

## 所需库
1. matplotlib
2. numpy
3. opencv_python
4. scikit_learn
5. umap_learn
6. pytorch

```
pip install -r requirements.txt
```

## 主要原理
* 初级数据筛选 img_primary_filter.py
  1. 文件类型 - 通过file_suffix参数设置
  2. 文件大小 - >10KB
  3. 图片尺寸 - 长&宽 > 256像素
  4. 图片梯度 - >50
  5. RGB三通道直方图相似度差值之和 - >0.003
* 高级数据筛选 img_advanced_filter.py

  1. 文件类型 - 通过file_suffix参数设置

  2. 降维特征图 - 利用ResNet50前7层提取图像特征图，在利用UMAP/t-SNE方法降维

## 参数设置
* root_dir_list - 输入文件路径列表

* file_suffix - 所需文件后缀名

  示例：

```
root_dir = ["../Image-Downloader-master/download_images/dog",\
			"../Image-Downloader-master/download_images/cat"]
file_suffix = "jpeg|jpg|png"
```

## 运行方式

* 运行初级数据筛选：
```
python img_primary_filter.py
```

* 运行高级数据筛选：
```
python img_advanced_filter.py
```

## 注意事项
本项目配合爬虫吗，如 [Image-Downloader](https://github.com/sczhengyabin/Image-Downloader) 项目使用（Image-Downloader中包含PHANTOMJS爬取方式，因此目前用最新4.0.0版selenium库会报错，使用旧版本如3.141.0可正常运行）。