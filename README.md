# CODE for *Isotonic Data Augmentation For Knowledge Distillation*

Here is the code for the paper *Isotonic Data Augmentation For Knowledge Distillation*

## Train Teacher Model

To pre-train a teacher-net for the downsteam **Knowledge Distillation**, run:
```shell script
python train_teacher.py --dataset <dataset> --arch <teacher_model_architecture> --batch-size <batch_size> --weight-decay <weigth_decay>
```

For example, to train a **ResNet50** teacher model, run:
```shell script
python train_teacher.py --dataset CIFAR100 --arch resnet50 --batch-size 128 --weight-decay 5e-4
```

## Knowledge Distillation Through Conventional Method
The conventional method for knowledge distillation is using the **soft label** from teacher-net and **ground truth** to train student-net.
```shell script
python kd_main.py --dataset <dataset> --teacher_arch <teacher_arch> --arch <studentj_arch> --batch_size <batch_size> --weight_decay <weight_decay> --temperature <temperature> --alpha <alpha> --lr <learning_rate>
```

For example, to distill **ResNet18** from **ResNet50**, run:
```shell script
python kd_main.py --dataset CIFAR100 --teacher-arch resnet50 --arch resnet18 --batch_size 128 --weight_decay 0.0005 --temperature 4.5 --alpha 0.95 --lr 0.1
```

## Isotonic Data Augmentation for Knowledge Distillation
To distill student-net by calibrating the soft labels when applied a data augmentation method, run:
```shell script
python kd_main.py --dataset <dataset> --num_classes <dataset_num_classes> --batch_size <batch_size> --epochs 200 --teacher_arch <teacher_arch> --arch <student_arch> --lr <lr> --weight_decay <weight_decay> --mixup_method <mixup_method> --temperature <temperature> --alpha <alpha> --calibration_method <calibration_method>  --soft_constraint_ratio <soft_constraint_ratio>
```

**Examples**
- **(KD Mixup) KD-i**, distill **ResNet18** from **ResNet50**, run:
```shell script
python kd_main.py --dataset CIFAR100 --num_classes 100 --batch_size 128 --epochs 200 --teacher_arch resnet50 --arch resnet18 --lr 0.1 --weight_decay 5e-4 --mixup_method mixup --temperature 4.5 --alpha 0.95 --calibration_method isotonic  --soft_constraint_ratio 2
```

- **(KD Mixup) KD-p**, distill **ResNet18** from **ResNet50**, run:
```shell script
python kd_main.py --dataset CIFAR100 --num_classes 100 --batch_size 128 --epochs 200 --teacher_arch resnet50 --arch resnet18 --lr 0.1 --weight_decay 5e-4 --mixup_method mixup --temperature 4.5 --alpha 0.95 --calibration_method isotonic_appr  --soft_constraint_ratio 2
```

- **(KD CutMix) KD-i**, distill **ResNet18** from **ResNet50**, run:
```shell script
python kd_main.py --dataset CIFAR100 --num_classes 100 --batch_size 128 --epochs 200 --teacher_arch resnet50 --arch resnet18 --lr 0.1 --weight_decay 5e-4 --mixup_method cutmix --temperature 4.5 --alpha 0.95 --calibration_method isotonic --soft_constraint_ratio 2
```

- **(KD CutMix) KD-p**, distill **ResNet18** from **ResNet50**, run:
```shell script
python kd_main.py --dataset CIFAR100 --num_classes 100 --batch_size 128 --epochs 200 --teacher_arch resnet50 --arch resnet18 --lr 0.1 --weight_decay 5e-4 --mixup_method cutmix --temperature 4.5 --alpha 0.95 --calibration_method isotonic_appr  --soft_constraint_ratio 2
```

**NOTE: As for CRD experiments, replace *kd_main.py* with *crd_kd.py* in the above commands.**
