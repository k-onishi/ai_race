## 実行手順

### repository clone, build

```
cd ~/catkin/src
git clone https://github.com/k-onishi/ai_race
cd ~/catkin/src
catkin clean -y
catkin build
source devel/setup.bash
```

### prepare

```
cd ~/catkin/src/ai_race/scripts
bash prepare.sh -l 1t
```

### 強化学習

```
mkdir -p ~/save_dir
cd ~/catkin/src/ai_race/ai_race/your_environment/scripts/reinforce_learning
python reinforce_learning.py #学習を終える場合は、"CTL+C"で終了させる, モデルファイルは、~/save_dir/model.pthに出力されている
```

### 推論

動かし方

```
cd ~/catkin/src/ai_race/ai_race/your_environment/scripts/reinforce_learning
python inference_from_image.py --pretrained_model (モデルファイル名)
```

(例) モデルファイル名="god_phoenix.pth"の場合

```
wget  https://github.com/k-onishi/ai_race/releases/download/v1.0/god_phoenix.pth
python inference_from_image.py --pretrained_model god_phoenix.pth
# この後暫く待って走行すればOK
```
