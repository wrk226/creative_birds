# 2D-to-3D-Evolution-Transfer
## Visualizations
- Evolutionary Birds from Different Angles
![rot](https://user-images.githubusercontent.com/111099396/185560433-c4e86a75-708c-4f3d-89ea-15058945bab1.gif)
- Shape Evolution with Different Scale Factor (Alpha)
![alpha](https://user-images.githubusercontent.com/111099396/185560430-1daa4c21-b9c8-4f7f-8e02-1ef4ec3133a0.gif)
- Texture Evolution with Different Switch Gate
![tex](https://user-images.githubusercontent.com/111099396/185560429-c1525e7e-810a-426f-a2f4-affc6c35b18f.gif)

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
