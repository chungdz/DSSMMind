python build_dicts.py
python build_train.py --processes=10 --max_hist_length 30
python resplit.py --filenum 10
python build_valid.py --processes=10 --max_hist_length 30 --fsamples=valid/behaviors.small.tsv
python build_test.py --processes=40 --max_hist_length 30
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --max_hist_length=30 --epoch=4 --batch_size=256 --model=gru
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate.py --gpus=4 --max_hist_length=30 --epoch=0 --filenum=40 --batch_size=256 --model=gru
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --max_hist_length=30 --epoch=1 --model=ctr_fm --batch_size=256
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate.py --gpus=4 --max_hist_length=30 --epoch=0 --filenum=40 --model=ctr_fm --batch_size=256
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate_build_test.py --gpus=4 --max_hist_length=30 --epoch=0 --filenum=40 --model=ctr_dfm --batch_size=256

python build_dicts.py --root=MIND --max_title=15
python build_train.py --processes=10 --max_hist_length=30 --max_title=15 --root=MIND --fsamples=train_behaviors.tsv
python resplit.py --filenum 10 --fsamples=MIND/raw/train
python build_valid.py --processes=10 --max_hist_length=30 --max_title=15 --fsamples=dev_behaviors.tsv --root=MIND --ftype=dev
python build_valid.py --processes=10 --max_hist_length=30 --max_title=15 --fsamples=test_behaviors.tsv --root=MIND --ftype=test
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --max_hist_length=30 --epoch=4 --batch_size=256 --root=MIND --vtype=dev
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate.py --gpus=4 --max_hist_length=30 --epoch=11 --filenum=10 --batch_size=256 --root=MIND

python build_dicts.py --root=Adressa --max_title=15
python build_train.py --processes=10 --max_hist_length=30 --max_title=15 --root=Adressa --fsamples=train_behaviors.tsv
python resplit.py --filenum 10 --fsamples=Adressa/raw/train
python build_valid.py --processes=10 --max_hist_length=30 --max_title=15 --fsamples=dev_behaviors.tsv --root=Adressa --ftype=dev
python build_valid.py --processes=10 --max_hist_length=30 --max_title=15 --fsamples=test_behaviors.tsv --root=Adressa --ftype=test
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --max_hist_length=30 --epoch=4 --batch_size=256 --root=Adressa --vtype=dev
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate.py --gpus=4 --max_hist_length=30 --epoch=11 --filenum=10 --batch_size=256 --root=Adressa
