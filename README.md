# batch_flame_fitting

### 环境创建
```
conda env create -f environment.yml
```
### data 结构
```
-- data_root
    -- subject
        -- video.mp4
```
### bash 脚本修改
```bash
subject_name=$subject
path=/path/to/your/data_root
```
### 对视频进行 fitting
```shell
bash fitting_flame.sh
```
### 渲染
```shell
python render_mesh_disk.py --data_path /path/to/your/data_root/$subject
```
