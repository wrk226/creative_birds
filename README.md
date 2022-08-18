# 2D-to-3D-Evolution-Transfer
## Visualizations
- Animated
![123](https://user-images.githubusercontent.com/111099396/185350189-9d104925-39fb-4bc1-b562-baafc3ec9378.gif)
- Texture Evolution Comparision
![compare_tex_4_page-0001](https://user-images.githubusercontent.com/111099396/185362935-5b828438-0605-45b2-95c1-c5a50d7a945c.jpg)
- Shape Evolution Comparision
![compare_shape_4_page-0001](https://user-images.githubusercontent.com/111099396/185362938-7b43b565-df43-4ddd-ac07-7af5e9a64fcd.jpg)
- Evolutionary shapes using the parameter Alpha
![alpha - 副本_page-0001](https://user-images.githubusercontent.com/111099396/185362928-a960eda7-41ce-4506-9788-e6fb671d7f06.jpg)

## Prerequisites
- Download code & pre-trained model:
Git clone the code by:
```
git clone https://github.com/anonymous-226/2D-to-3D-Evolution-Transfer $ROOTPATH
```
The pretrained model can be found from [here](https://drive.google.com/file/d/1Agf_G9OaCvXPoenRK5vpj3VckuFPGRMg/view?usp=sharing), which should be unzipped in `$ROOTPATH`.
- Install packages:
```
conda create -n evo_trans python=3.6
conda activate evo_trans
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

- Install external tools:

```
cd $ROOTPATH/2D-to-3D-Evolution-Transfer/dependency
unzip meshzoo-0.4.3.zip
cd meshzoo-0.4.3
python setup.py install
```

```
cd $ROOTPATH/2D-to-3D-Evolution-Transfer/dependency
unzip neural_renderer.zip
cd neural_renderer
python setup.py install
```

## Run the test
Run the following command from the `$ROOTPATH/2D-to-3D-Evolution-Transfer` directory:
```
python -m evo_trans.experiments.test_df2
```
The result can be found in `$ROOTPATH/2D-to-3D-Evolution-Transfer/evo_trans/cachedir/visualization` directory.
