# 2D-to-3D-Evolution-Transfer
## Visualizations
- Evolutionary Birds from Different Angles
![rot](https://user-images.githubusercontent.com/111099396/185530766-0dff8428-5d94-4005-858c-a3dbad581ab0.gif)
- Texture Evolution with Different Switch Gate
![tex](https://user-images.githubusercontent.com/111099396/185530776-d410b0a5-0425-4a86-bba0-9cc40d27dc6d.gif)
- Shape Evolution with Different Scale Factor (Alpha)
![alpha](https://user-images.githubusercontent.com/111099396/185532824-6f8f2d35-c44f-4ebe-9e44-f98f131b42a1.gif)

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
