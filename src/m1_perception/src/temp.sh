# python -m m1_perception.speed_model_nopretrain.train --config.logdir=speed_model_logs/nopretrain/trial1
# python -m m1_perception.speed_model_nopretrain.train --config.logdir=speed_model_logs/nopretrain/trial2
# python -m m1_perception.speed_model_nopretrain.train --config.logdir=speed_model_logs/nopretrain/trial3
# python -m m1_perception.speed_model_nopretrain.train --config.logdir=speed_model_logs/nopretrain/trial4
# python -m m1_perception.speed_model_nopretrain.train --config.logdir=speed_model_logs/nopretrain/trial5

# python -m m1_perception.speed_model.train --logdir='speed_model_logs/embedding/trial1' --num_epoches=60
# python -m m1_perception.speed_model.train --logdir='speed_model_logs/embedding/trial2' --num_epoches=60
# python -m m1_perception.speed_model.train --logdir='speed_model_logs/embedding/trial3' --num_epoches=60
# python -m m1_perception.speed_model.train --logdir='speed_model_logs/embedding/trial4' --num_epoches=60
# python -m m1_perception.speed_model.train --logdir='speed_model_logs/embedding/trial5' --num_epoches=60

python -m m1_perception.speed_model_class_score.train --logdir speed_model_logs/class_score/trial_1 --num_epoches=60
python -m m1_perception.speed_model_class_score.train --logdir speed_model_logs/class_score/trial_2 --num_epoches=60
python -m m1_perception.speed_model_class_score.train --logdir speed_model_logs/class_score/trial_3 --num_epoches=60
python -m m1_perception.speed_model_class_score.train --logdir speed_model_logs/class_score/trial_4 --num_epoches=60
python -m m1_perception.speed_model_class_score.train --logdir speed_model_logs/class_score/trial_5 --num_epoches=60


python -m m1_perception.speed_model_class_category.train --logdir speed_model_logs/class_category/trial_1 --num_epoches=60
python -m m1_perception.speed_model_class_category.train --logdir speed_model_logs/class_category/trial_2 --num_epoches=60
python -m m1_perception.speed_model_class_category.train --logdir speed_model_logs/class_category/trial_3 --num_epoches=60
python -m m1_perception.speed_model_class_category.train --logdir speed_model_logs/class_category/trial_4 --num_epoches=60
python -m m1_perception.speed_model_class_category.train --logdir speed_model_logs/class_category/trial_5 --num_epoches=60
